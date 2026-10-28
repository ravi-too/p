"""Model construction and checkpoint loading."""
from pathlib import Path
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + pedestrian


def build_model(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, NUM_CLASSES)
    return model


def load_model(checkpoint=None, device=None, pretrained_if_missing=True):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(pretrained=pretrained_if_missing if checkpoint is None else False)
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}. Train first or pass a valid path.")
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)
    model.eval()
    return model, device
