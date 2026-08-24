import argparse
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from timm import create_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mamba_decoder import pdd_decoder
from models.resnet_encoder import wide_resnet50_2
from models.vmamba import VSSM
from utils.dataset_full import build_multi_class_dataset, get_data_transforms
from utils.losses import CosineLoss, DiversityLoss, MSE_loss
from utils.utils import get_time_stamp


def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def compute_f1max(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.max(f1))


def batch_stu_ano_map(fs_list, ft_list, out_size, amap_mode="mul"):
    batch_size = fs_list[0].shape[0]
    if amap_mode == "mul":
        anomaly_map = np.ones((batch_size, out_size, out_size), dtype=np.float32)
    else:
        anomaly_map = np.zeros((batch_size, out_size, out_size), dtype=np.float32)

    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[:, 0].detach().cpu().numpy()

        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map


def batch_anomaly_scores(outputs, out_size):
    ano_map_1 = batch_stu_ano_map(outputs[0:3], outputs[3:6], out_size, amap_mode="a")
    ano_map_2 = batch_stu_ano_map(outputs[6:9], outputs[3:6], out_size, amap_mode="a")
    anomaly_maps = ano_map_1 + ano_map_2

    scores = []
    for anomaly_map in anomaly_maps:
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        scores.append(float(np.max(anomaly_map)))
    return np.asarray(scores, dtype=np.float32)


def evaluate_and_aggregate_multi(
    encoder_ssm,
    encoder_res,
    decoder_module,
    res,
    val_loader,
    device,
    writer,
    epoch,
    class_names,
):
    rank = dist.get_rank()
    results_per_class = defaultdict(lambda: {"labels": [], "preds": []})

    decoder_module.eval()
    with torch.no_grad():
        for img, label, cls_name in val_loader:
            img = img.to(device, non_blocking=True)
            inputs_ssm = encoder_ssm(img, feature_list=True)
            inputs_res = encoder_res(img, feature_list=True)
            outputs = decoder_module(inputs_res[3], inputs_res, inputs_ssm, res=res, feature_list=True)
            scores = batch_anomaly_scores(outputs, img.shape[-1])

            labels = label.cpu().numpy().astype(int)
            for cls, gt, pred in zip(cls_name, labels, scores):
                results_per_class[cls]["labels"].append(int(gt))
                results_per_class[cls]["preds"].append(float(pred))

    if rank != 0:
        return None, None, None

    all_labels = []
    all_preds = []
    class_results = {}

    for cls in class_names:
        labels = np.asarray(results_per_class[cls]["labels"], dtype=np.int32)
        preds = np.asarray(results_per_class[cls]["preds"], dtype=np.float32)
        if labels.size == 0 or len(np.unique(labels)) < 2:
            continue

        auroc = float(roc_auc_score(labels, preds))
        aupr = float(average_precision_score(labels, preds))
        f1max = compute_f1max(labels, preds)
        class_results[cls] = (auroc, aupr, f1max)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    if len(np.unique(all_labels)) < 2:
        return None, None, None

    auroc_global = float(roc_auc_score(all_labels, all_preds))
    aupr_global = float(average_precision_score(all_labels, all_preds))
    f1_global = compute_f1max(all_labels, all_preds)

    if writer is not None:
        writer.add_scalar("AUROC/test", auroc_global, epoch)
        writer.add_scalar("AUPR/test", aupr_global, epoch)
        writer.add_scalar("F1_max/test", f1_global, epoch)
        for cls, metrics in class_results.items():
            writer.add_scalar(f"AUROC/{cls}", metrics[0], epoch)
            writer.add_scalar(f"AUPR/{cls}", metrics[1], epoch)
            writer.add_scalar(f"F1_max/{cls}", metrics[2], epoch)

    return auroc_global, aupr_global, f1_global


def build_dataloaders(args):
    data_transform, _ = get_data_transforms(args.image_size, args.image_size)
    meta_path = osp.join(args.data_path, "meta.json")
    class_names = [cls.strip() for cls in args.class_list.split(",") if cls.strip()]

    train_dataset = build_multi_class_dataset(
        meta_path,
        args.data_path,
        data_transform,
        phase="train",
        class_list=class_names,
    )
    val_dataset = build_multi_class_dataset(
        meta_path,
        args.data_path,
        data_transform,
        phase="test",
        class_list=class_names,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers_val,
        pin_memory=True,
        persistent_workers=args.num_workers_val > 0,
    )
    return train_loader, train_sampler, val_loader, class_names


def train_ddp_multi(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    is_main = rank == 0

    if is_main:
        time_stamp = get_time_stamp()
        class_tag = args.class_list.replace(",", "_")
        log_dir = osp.join(PROJECT_ROOT, "logs", "train", "multi", f"{class_tag}_{time_stamp}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[Rank {rank}] Logging to {log_dir}")
    else:
        writer = None

    train_loader, train_sampler, val_loader, class_names = build_dataloaders(args)

    encoder_res = wide_resnet50_2(pretrained=True).to(device).eval()
    encoder_ssm = create_model("vanilla_vmamba_tiny", pretrained=True).to(device).eval()
    decoder = pdd_decoder(pretrained=False).to(device)
    decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.t_max,
        eta_min=args.lr * 0.1,
    )

    start_epoch = 0
    if args.resume and args.ckpt_path and osp.exists(args.ckpt_path):
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        checkpoint = torch.load(args.ckpt_path, map_location=map_location)
        decoder.module.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch - 1}")

    max_auc = []
    max_auc_epoch = []

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        train_sampler.set_epoch(epoch)

        loss_list = []
        l_r_m_list = []
        l_f_s_list = []
        l_s_s_list = []

        loop = tqdm.tqdm(train_loader, disable=not is_main)
        for img, _, _ in loop:
            img = img.to(device, non_blocking=True)
            l_r_m = torch.tensor(0.0, device=device)
            l_f_s = torch.tensor(0.0, device=device)
            l_s_s = torch.tensor(0.0, device=device)

            with torch.no_grad():
                inputs_ssm = encoder_ssm(img, feature_list=True)
                inputs_res = encoder_res(img, feature_list=True)

            outputs = decoder(inputs_res[3], inputs_res, inputs_ssm, res=args.res, feature_list=True)

            if args.layerloss == 1:
                loss = 0.02 * MSE_loss(outputs[0:3], outputs[3:6])
                l_r_m = loss
            elif args.layerloss == 2:
                l_r_m = 0.02 * MSE_loss(outputs[0:3], outputs[3:6]) + 0.05 * CosineLoss(outputs[0:3], outputs[3:6])
                l_f_s = 0.02 * MSE_loss(outputs[3:6], outputs[6:9])
                l_s_s = 0.5 * CosineLoss(outputs[0:3], outputs[6:9])
                loss = l_r_m + l_f_s + l_s_s
            elif args.layerloss == 4:
                l_r_m = 0.02 * MSE_loss(outputs[0:3], outputs[3:6]) + 0.05 * CosineLoss(outputs[0:3], outputs[3:6])
                l_f_s = 0.02 * MSE_loss(outputs[3:6], outputs[6:9])
                l_s_s = 0.5 * DiversityLoss(
                    features_student1=outputs[0:3],
                    features_student2=outputs[6:9],
                    low_dim_layers=1,
                    tau_low=0.75,
                    tau_high=0.30,
                )
                loss = l_r_m + l_f_s + l_s_s
            else:
                raise ValueError(f"Unsupported layerloss={args.layerloss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            l_r_m_list.append(l_r_m.item())
            l_f_s_list.append(l_f_s.item())
            l_s_s_list.append(l_s_s.item())

            if is_main:
                loop.set_description(
                    f"Epoch[{epoch + 1}/{args.epochs}] loss={np.mean(loss_list):.4f} "
                    f"l_r_m={np.mean(l_r_m_list):.4f} l_f_s={np.mean(l_f_s_list):.4f} "
                    f"l_s_s={np.mean(l_s_s_list):.4f}"
                )

        avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
        avg_l_r_m = float(np.mean(l_r_m_list)) if l_r_m_list else 0.0
        avg_l_f_s = float(np.mean(l_f_s_list)) if l_f_s_list else 0.0
        avg_l_s_s = float(np.mean(l_s_s_list)) if l_s_s_list else 0.0
        scheduler.step()

        if is_main and writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            if args.print_loss and (epoch + 1) % 10 == 0:
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.4f} "
                    f"l_r_m={avg_l_r_m:.4f} l_f_s={avg_l_f_s:.4f} l_s_s={avg_l_s_s:.4f}"
                )

        if (epoch + 1) % args.print_epoch == 0:
            auroc_global, aupr_global, f1_global = evaluate_and_aggregate_multi(
                encoder_ssm,
                encoder_res,
                decoder.module,
                args.res,
                val_loader,
                device,
                writer,
                epoch,
                class_names,
            )

            if is_main and auroc_global is not None:
                print(
                    f"[Epoch {epoch + 1}] Global AUROC: {auroc_global:.4f}, "
                    f"AUPR: {aupr_global:.4f}, f1_global:{f1_global:.4f}"
                )
                max_auc.append(auroc_global)
                max_auc_epoch.append(epoch + 1)

                if args.print_max:
                    print(f"max_auc = {max(max_auc):.4f}")
                    print(f"max_epoch = {max_auc_epoch[max_auc.index(max(max_auc))]}")

                os.makedirs(args.save_path, exist_ok=True)
                save_file = osp.join(args.save_path, f"epoch_{epoch + 1}_auc={auroc_global:.4f}.pth")
                torch.save(
                    {
                        "decoder": decoder.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_file,
                )

    if is_main and writer is not None:
        writer.close()
    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_list", type=str, default="brain,liver,retinal")
    parser.add_argument("--data_path", type=str, default="/data0/lxj/dataset/medical")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--t_max", type=int, default=40)
    parser.add_argument("--res", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--val_batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=4)
    parser.add_argument("--print_epoch", type=int, default=10)
    parser.add_argument("--print_loss", type=int, default=1)
    parser.add_argument("--layerloss", type=int, default=4)
    parser.add_argument("--print_max", type=int, default=1)
    parser.add_argument("--score_num", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=osp.join(PROJECT_ROOT, "checkpoints", "multi_tao_0375"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ddp_multi(args)
