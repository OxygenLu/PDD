# Two-twins DDP training entry.
import os
import os.path as osp
import sys
import argparse
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from timm import create_model

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
from utils.evaluation import evaluation  
from utils.losses import MSE_loss, CosineLoss, distill_loss, KLloss, DiversityLoss
from utils.utils import get_time_stamp
from models.resnet_encoder import wide_resnet50_2

from models.mamba_decoder import pdd_decoder
from models.vmamba import VSSM

def setup_ddp():
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def weighted_metric_aggregate(device, local_metric, local_count):

    tensor = torch.tensor([local_metric * local_count, local_count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_weight = tensor[1].item()
    if total_weight == 0:
        return 0.0
    global_metric = (tensor[0].item() / total_weight)
    return global_metric


def gather_preds_labels(device, local_preds, local_labels):
    """
    Use all_gather_object to collect preds/labels of each rank (with variable-length).
    Return (all_preds_concat, all_labels_concat). Numpy arrays are returned only on rank0, while other ranks return (None, None).
    """
    world_size = dist.get_world_size()
    # all_gather_object get picklable objects
    gather_list_preds = [None for _ in range(world_size)]
    gather_list_labels = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list_preds, local_preds)
    dist.all_gather_object(gather_list_labels, local_labels)

    if dist.get_rank() == 0:
        import numpy as np
        all_preds = np.concatenate([np.asarray(x) for x in gather_list_preds], axis=0)
        all_labels = np.concatenate([np.asarray(x) for x in gather_list_labels], axis=0)
        return all_preds, all_labels
    else:
        return None, None



def evaluate_and_aggregate(encoder_ssm, encoder_res, decoder_module, res, val_loader, device, score_num, writer, epoch):
    rank = dist.get_rank()
    device_t = device

    try:
        res_all = evaluation(encoder_ssm, encoder_res, decoder_module, res, val_loader, device_t, score_num)
        if isinstance(res_all, tuple) and len(res_all) >= 3:
            auroc_local = float(res_all[0])
            aupr_local = float(res_all[1])
            f1_local   = float(res_all[2])

            # local sample count
            n_local = len(val_loader.dataset)

            # create tensors
            t_auroc = torch.tensor([auroc_local * n_local, n_local], device=device_t, dtype=torch.float64)
            t_aupr  = torch.tensor([aupr_local * n_local, n_local], device=device_t, dtype=torch.float64)
            t_f1    = torch.tensor([f1_local   * n_local, n_local], device=device_t, dtype=torch.float64)

            dist.all_reduce(t_auroc, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_aupr,  op=dist.ReduceOp.SUM)
            dist.all_reduce(t_f1,    op=dist.ReduceOp.SUM)

            if rank == 0:
                total_n_auroc = t_auroc[1].item()
                total_n_aupr  = t_aupr[1].item()
                total_n_f1    = t_f1[1].item()

                auroc_global = (t_auroc[0].item() / total_n_auroc) if total_n_auroc > 0 else 0.0
                aupr_global  = (t_aupr[0].item()  / total_n_aupr)  if total_n_aupr  > 0 else 0.0
                f1_global    = (t_f1[0].item()    / total_n_f1)    if total_n_f1    > 0 else 0.0

                if writer is not None:
                    writer.add_scalar('AUROC/test', auroc_global, epoch)
                    writer.add_scalar('AUPR/test',  aupr_global,  epoch)
                    writer.add_scalar('F1_max/test', f1_global,   epoch)

                return auroc_global, aupr_global, f1_global
            else:
                return None, None, None

        else:
            raise ValueError("Evaluation return format not recognized")

    except Exception as e:
        if rank == 0:
            print(f"[evaluate_and_aggregate] can't use evaluation, ERRO: {e}")
        return None, None, None



def train_ddp(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = (rank == 0)

    if is_main:
        time_stamp = get_time_stamp()
        stamp = f"{args.class_name}_lr{time_stamp}"
        log_dir = osp.join(PROJECT_ROOT, 'logs', 'train', str(args.class_name), stamp)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[Rank {rank}] Logging to {log_dir}")
    else:
        writer = None

    # 数据
    data_transform, _ = get_data_transforms(args.image_size, args.image_size)
    train_path = osp.join(args.data_path, 'train')
    val_path = args.data_path

    train_dataset = ImageFolder(root=train_path, transform=data_transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataset = MVTecDataset(root=val_path, transform=data_transform, phase="test")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers_val,
        pin_memory=True,
    )

    # 模型（teacher 设 eval）
    encoder_res = wide_resnet50_2(pretrained=True).to(device).eval()
    encoder_ssm = create_model("vanilla_vmamba_tiny", pretrained=True).to(device).eval()

    decoder = pdd_decoder(pretrained=False).to(device)
    decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.lr * 0.1)

    # resume
    start_epoch = 0
    if args.resume and args.ckpt_path and osp.exists(args.ckpt_path):
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(args.ckpt_path, map_location=map_location)
        decoder.module.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch-1}")

    max_auc = []
    max_auc_epoch = []

    # training loop
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        train_sampler.set_epoch(epoch)

        loss_list = []
        l_r_m_list = []
        l_f_s_list = []
        l_s_s_list = []

        loop = tqdm.tqdm(train_loader, disable=not is_main)
        for img, label in loop:
            img = img.to(device, non_blocking=True)
            l_r_m = torch.tensor(0.0, device=device)
            l_f_s = torch.tensor(0.0, device=device)
            l_s_s = torch.tensor(0.0, device=device)
            with torch.no_grad():
                inputs_ssm = encoder_ssm(img, feature_list=True)
                inputs_res = encoder_res(img, feature_list=True)
            outputs = decoder(inputs_res[3], inputs_res, inputs_ssm, res=args.res, feature_list=True)

            if args.layerloss == 1:
                loss = 0.02*MSE_loss(outputs[0:3], outputs[3:6])
            elif args.layerloss == 2:
                l_r_m = 0.02 * MSE_loss(outputs[0:3], outputs[3:6]) + 0.05 * CosineLoss(outputs[0:3], outputs[3:6])#0.05
                l_f_s = 0.02 * MSE_loss(outputs[3:6], outputs[6:9]) 
                
                l_s_s = 0.5 * CosineLoss(outputs[0:3], outputs[6:9])
                loss = l_r_m + l_f_s + l_s_s
                # loss = 0.05*MSE_loss(outputs[0:3], inputs_res[0:3]) + 0.05*MSE_loss(outputs[0:3], outputs[9:12]) + \
                #        0.05*MSE_loss(outputs[3:6], outputs[6:9]) + \
                #        0.5*MSE_loss(outputs[0:3], outputs[6:9])

            elif args.layerloss == 3:
                loss = MSE_loss(outputs[6:9], outputs[9:12]) + MSE_loss(outputs[6:9], outputs[0:3]) - CosineLoss(outputs[0:3],outputs[6:9])

            elif args.layerloss == 4:#Final loss
                l_r_m = 0.02 * MSE_loss(outputs[0:3], outputs[3:6]) + 0.05 * CosineLoss(outputs[0:3], outputs[3:6])
                l_f_s = 0.02 * MSE_loss(outputs[3:6], outputs[6:9]) 
                l_s_s = 0.5 * DiversityLoss( 
                        features_student1=outputs[0:3],  # 3 block features of student1
                        features_student2=outputs[6:9],  # 3 block features of student2
                        low_dim_layers=1,    # Diversity in block1, consistency in block2‑3
                        tau_low=0.75,         # low‑dimensional layer similarity threshold
                        tau_high=0.30,       # High‑dimensional layer similarity threshold
                    )
                loss = l_r_m + l_f_s + l_s_s
                # l_s_s = 0.5 * CosineLoss(outputs[0:3], outputs[6:9])

            else:
                # other loss 
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            loss_list.append(loss.item())
            l_r_m_list.append(l_r_m.item())
            l_f_s_list.append(l_f_s.item())
            l_s_s_list.append(l_s_s.item())

            if is_main:
                loop.set_description(f"Epoch[{epoch+1}/{args.epochs}] loss={np.mean(loss_list):.4f} l_r_m={np.mean(l_r_m_list):.4f} l_f_s={np.mean(l_f_s_list):.4f} l_s_s={np.mean(l_s_s_list):.4f}")
                # loop.set_description(f"Epoch[{epoch+1}/{args.epochs}] loss={np.mean(loss_list):.4f}")# abluation 1

        avg_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
        avg_l_r_m = float(np.mean(l_r_m_list)) if len(l_r_m_list) > 0 else 0.0
        avg_l_f_s = float(np.mean(l_f_s_list)) if len(l_f_s_list) > 0 else 0.0
        avg_l_s_s = float(np.mean(l_s_s_list)) if len(l_s_s_list) > 0 else 0.0
        scheduler.step()

        if is_main:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            if args.print_loss and (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f} l_r_m={avg_l_r_m:.4f} l_f_s={avg_l_f_s:.4f} l_s_s={avg_l_s_s:.4f}")
                # print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}")# abluation 1

        # eval & aggregate
        if (epoch + 1) % args.print_epoch == 0:
            auroc_global, aupr_global, f1_global = evaluate_and_aggregate(
                encoder_ssm, encoder_res, decoder.module, args.res, val_loader, device, args.score_num, writer, epoch
            )
            if is_main and auroc_global is not None:
                print(f"[Epoch {epoch+1}] Global AUROC: {auroc_global:.4f}, AUPR: {aupr_global:.4f}, f1_global:{f1_global:.4f}")
                #  F1: {f1_global:.4f}
                max_auc.append(auroc_global)
                max_auc_epoch.append(epoch + 1)
                if args.print_max:
                    print(f"max_auc = {max(max_auc):.4f}")
                    print(f"max_epoch = {max_auc_epoch[max_auc.index(max(max_auc))]}")

                # save weight（rank0）
                os.makedirs(args.save_path, exist_ok=True)
                save_file = osp.join(args.save_path, f"epoch_{epoch+1}_auc={auroc_global:.4f}.pth")
                torch.save({
                    'decoder': decoder.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, save_file)

    if is_main and writer is not None:
        writer.close()
    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='head_release_tao_07503')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--t_max', type=int, default=40)
    parser.add_argument('--res', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_workers_val', type=int, default=4)
    parser.add_argument('--print_epoch', type=int, default=10)
    parser.add_argument('--print_loss', type=int, default=1)
    parser.add_argument('--layerloss', type=int, default=4)
    parser.add_argument('--print_max', type=int, default=1)
    parser.add_argument('--score_num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/data0/lxj/dataset/data/head_ct')
    parser.add_argument('--save_path', type=str, default=osp.join(PROJECT_ROOT, 'checkpoints', 'head_release_tao_07503'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 scripts/train_two_twins_ddp.py
    train_ddp(args)
