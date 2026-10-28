"""Penn-Fudan dataset download and PyTorch dataset."""
from pathlib import Path
import random
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

DATA_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
ROOT = Path("data")
EXTRACTED = ROOT / "PennFudanPed"


def download_dataset(root: Path = ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "PennFudanPed.zip"
    if not EXTRACTED.exists():
        if not target.exists():
            print(f"Downloading Penn-Fudan dataset to {target} ...")
            urllib.request.urlretrieve(DATA_URL, target)
        with zipfile.ZipFile(target) as zf:
            zf.extractall(root)
    if not (EXTRACTED / "PNGImages").exists():
        raise FileNotFoundError(f"Expected dataset at {EXTRACTED}")
    return EXTRACTED


class PennFudanDataset(Dataset):
    def __init__(self, root: str = "data/PennFudanPed", indices=None):
        self.root = Path(root)
        if not self.root.exists():
            download_dataset(self.root.parent)
        self.images = sorted((self.root / "PNGImages").glob("*.png"))
        self.masks = sorted((self.root / "PedMasks").glob("*_mask.png"))
        mask_by_stem = {p.name.replace("_mask", ""): p for p in self.masks}
        self.pairs = [(img, mask_by_stem[img.name]) for img in self.images if img.name in mask_by_stem]
        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs found in {self.root}")
        if indices is not None:
            self.pairs = [self.pairs[i] for i in indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image)
        mask_np = np.asarray(Image.open(mask_path))
        # Each pedestrian instance has a unique non-zero ID; 0 is background.
        instance_ids = np.unique(mask_np)
        instance_ids = instance_ids[(instance_ids != 0) & (instance_ids != 255)]
        masks = mask_np[None, :, :] == instance_ids[:, None, None] if len(instance_ids) else np.zeros((0, *mask_np.shape), dtype=bool)
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) and len(ys):
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # class 1 = pedestrian
        masks_t = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        image_t = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0
        return image_t, target


def make_splits(root="data/PennFudanPed", test_fraction=0.2, seed=42):
    dataset = PennFudanDataset(root)
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(n * test_fraction))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    path = download_dataset()
    ds = PennFudanDataset(str(path))
    print(f"Dataset: {path}")
    print(f"Image/mask pairs: {len(ds)}")
    image, target = ds[0]
    print(f"First image shape: {tuple(image.shape)}")
    print(f"Pedestrians in first image: {len(target['boxes'])}")
