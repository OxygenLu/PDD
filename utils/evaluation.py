from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import torch
from scipy.ndimage import gaussian_filter
# from losses import distill_loss, KLloss, BCEloss
import numpy as np
import torch.nn.functional as F
import random
# from metrics import compute_pro

def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# f1-score
def compute_f1max(gt, pred):
    precision, recall, _ = precision_recall_curve(gt, pred)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return np.max(f1)


def cal_fusion_ano_map(fs_list, ft_list, out_size=224, amap_mode='mul'):

    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) 
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1-F.cosine_similarity(fs, ft)# 修改掉
        # a_map_mse = 1-F.mse_loss(fs, ft, reduction="none").mean(dim=1)  # 计算均方误差
        # a_map = a_map_mse 

        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)

        if amap_mode == 'mul':
            anomaly_map *= a_map 
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_stu_ano_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) 
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1-F.cosine_similarity(fs, ft)  # 计算余弦相似度
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0,:,:].to('cpu').detach().numpy()
        a_map_list.append(a_map)

        if amap_mode == 'mul':
            anomaly_map *= a_map 
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list





def evaluation(t1,t2, decoder, res, dataloader, device, score_num):
    decoder.eval()
    gt_list_sp = [] 
    pr_list_sp = []


    
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            inputs_ssm = t1(img,feature_list=True)
            inputs_res = t2(img,feature_list=True)
            outputs = decoder(inputs_res[3],inputs_res,inputs_ssm,res,feature_list=True)

            # twins t&s 
            ano_map_cos1, _ = cal_stu_ano_map(outputs[0:3], outputs[3:6], img.shape[-1], amap_mode='a')
            ano_map_cos2, _ = cal_stu_ano_map(outputs[6:9], outputs[3:6], img.shape[-1], amap_mode='a')

            anomaly_map_cosine = ano_map_cos1 + ano_map_cos2

            
            anomaly_map = gaussian_filter(anomaly_map_cosine, sigma=4)


            # sample-level
            gt_list_sp.append(np.max(label.cpu().numpy().astype(int))) 
            pr_list_sp.append(np.max(anomaly_map))


        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        aupro_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)
        f1max_px = round(compute_f1max(gt_list_sp, pr_list_sp), 4)

    return auroc_sp, aupro_sp, f1max_px




    # gt_list_px = []
    # pr_list_px = []
    # aupro_list = [] 


            # abluation 1: 2 t & 1 student, without skip feature
            # ano_map_cos, _ = cal_stu_ano_map(outputs[0:3], outputs[3:6], img.shape[-1], amap_mode='a')
            # anomaly_map_cosine = ano_map_cos

            # # abluation 1: 2 t & 1 student, without skip feature
            # ano_map_cos, _ = cal_stu_ano_map(outputs[0:3], outputs[6:9], img.shape[-1], amap_mode='a')
            # anomaly_map_cosine = ano_map_cos
