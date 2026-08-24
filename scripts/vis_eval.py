import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
from timm import create_model

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import MVTecDataset, get_data_transforms
from utils.evaluation import cal_stu_ano_map
from models.mamba_decoder import pdd_decoder
from models.resnet_encoder import wide_resnet50_2
from models.vmamba import VSSM


ckp_path = "/data0/lxj/VAD-TS/checkpoints/head_release_tao_07503/epoch_90_auc=0.9750.pth"
data_path = "/data0/lxj/dataset/data/head_ct"
img_size = 256
res = 10


def min_max_norm(array, eps=1e-8):
    array = np.asarray(array)
    return (array - array.min()) / (array.max() - array.min() + eps)


def to_heatmap(anomaly_map):
    anomaly_map = np.uint8(min_max_norm(anomaly_map) * 255)
    return cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)


def overlay_heatmap(image_bgr, heatmap_bgr, alpha):
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)


def read_original_image(img_path, image_size):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def build_models(device):
    encoder_res = wide_resnet50_2(pretrained=True).to(device).eval()
    encoder_ssm = create_model("vanilla_vmamba_tiny", pretrained=True).to(device).eval()

    checkpoint = torch.load(ckp_path, map_location=device)
    state_dict = checkpoint["decoder"] if "decoder" in checkpoint else checkpoint
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    decoder = pdd_decoder(pretrained=False).to(device)
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()
    return encoder_ssm, encoder_res, decoder


def compute_anomaly_map(encoder_ssm, encoder_res, decoder, img):
    inputs_ssm = encoder_ssm(img, feature_list=True)
    inputs_res = encoder_res(img, feature_list=True)
    outputs = decoder(
        inputs_res[3],
        inputs_res,
        inputs_ssm,
        res=res,
        feature_list=True,
    )

    ano_map_1, _ = cal_stu_ano_map(outputs[0:3], outputs[3:6], img.shape[-1], amap_mode="a")
    ano_map_2, _ = cal_stu_ano_map(outputs[6:9], outputs[3:6], img.shape[-1], amap_mode="a")
    return gaussian_filter(ano_map_1 + ano_map_2, sigma=4)


def save_visuals(image_bgr, anomaly_map, label, img_path, index, output_dir, alpha):
    img_type = osp.basename(osp.dirname(img_path))
    stem = osp.splitext(osp.basename(img_path))[0]
    prefix = f"{index:05d}_{img_type}_label{int(label)}_{stem}"

    heatmap = to_heatmap(anomaly_map)
    overlay = overlay_heatmap(image_bgr, heatmap, alpha)
    compare = np.concatenate([image_bgr, heatmap, overlay], axis=1)

    for subdir, image in (
        ("org", image_bgr),
        ("heatmap", heatmap),
        ("overlay", overlay),
        ("compare", compare),
    ):
        save_dir = osp.join(output_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(osp.join(save_dir, f"{prefix}.png"), image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize head_ct/test anomaly maps using the PDD two-twins model."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=osp.join(PROJECT_ROOT, "vis", "head_ct"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.exists(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform, _ = get_data_transforms(img_size, img_size)
    test_dataset = MVTecDataset(root=data_path, transform=data_transform, phase="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    encoder_ssm, encoder_res, decoder = build_models(device)
    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for index, (img, label) in enumerate(tqdm.tqdm(test_loader, desc="Visualizing")):
            if args.max_images is not None and saved >= args.max_images:
                break

            img = img.to(device, non_blocking=True)
            anomaly_map = compute_anomaly_map(encoder_ssm, encoder_res, decoder, img)
            image_bgr = read_original_image(test_dataset.img_paths[index], img_size)
            save_visuals(
                image_bgr=image_bgr,
                anomaly_map=anomaly_map,
                label=label.item(),
                img_path=test_dataset.img_paths[index],
                index=index,
                output_dir=args.output_dir,
                alpha=args.alpha,
            )
            saved += 1

    print(f"Saved {saved} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
