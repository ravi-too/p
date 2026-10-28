import torch
from src.dataset import PennFudanDataset


def test_dataset_sample_shapes():
    ds = PennFudanDataset()
    image, target = ds[0]
    assert image.ndim == 3 and image.shape[0] == 3
    assert image.dtype == torch.float32
    assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert len(target["boxes"]) == len(target["labels"]) == len(target["masks"])
    if len(target["boxes"]):
        assert torch.all(target["boxes"][:, 2] > target["boxes"][:, 0])
        assert torch.all(target["boxes"][:, 3] > target["boxes"][:, 1])
