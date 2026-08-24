import json
import os

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train),
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
    ])
    return data_transforms, gt_transforms


class MetaMVTecDataset(Dataset):
    def __init__(self, meta_path, transform=None, phase="train", cls_name="brain", root_dir=""):
        self.phase = phase
        self.transform = transform
        self.root_dir = root_dir
        self.cls_name = cls_name

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if phase not in meta:
            raise ValueError(f"phase '{phase}' not found in meta.json")
        if cls_name not in meta[phase]:
            raise ValueError(f"class '{cls_name}' not found in '{phase}'")

        items = meta[phase][cls_name]
        if phase == "train":
            items = [item for item in items if item["anomaly"] == 0]
        self.items = items
        print(f"[{phase.upper()}] Class '{cls_name}' -> {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.root_dir, item["img_path"])
        label = int(item["anomaly"])

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.cls_name


def build_multi_class_dataset(meta_path, root_dir, transform, phase="train", class_list=None):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    all_classes = list(meta[phase].keys()) if class_list is None else class_list
    datasets = [
        MetaMVTecDataset(
            meta_path=meta_path,
            transform=transform,
            phase=phase,
            cls_name=cls,
            root_dir=root_dir,
        )
        for cls in all_classes
    ]
    return ConcatDataset(datasets)
