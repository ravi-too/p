"""Evaluate detections with greedy one-to-one matching at a chosen IoU threshold."""
import argparse
import time
import torch
from torch.utils.data import DataLoader
from src.dataset import make_splits, collate_fn
from src.model import load_model


def box_iou(box_a, box_b):
    if len(box_a) == 0 or len(box_b) == 0:
        return torch.zeros((len(box_a), len(box_b)))
    lt = torch.maximum(box_a[:, None, :2], box_b[None, :, :2])
    rb = torch.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]).clamp(min=0) *
              (box_a[:, 3] - box_a[:, 1]).clamp(min=0))
    area_b = ((box_b[:, 2] - box_b[:, 0]).clamp(min=0) *
              (box_b[:, 3] - box_b[:, 1]).clamp(min=0))
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


@torch.inference_mode()
def evaluate(model, loader, device, score_threshold=0.5, iou_threshold=0.5):
    tp = fp = fn = 0
    durations = []
    n_images = 0
    model.eval()
    for images, targets in loader:
        images = [x.to(device) for x in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000 / len(images))
        for output, target in zip(outputs, targets):
            keep = output["scores"].detach().cpu() >= score_threshold
            pred_boxes = output["boxes"].detach().cpu()[keep]
            gt_boxes = target["boxes"].cpu()
            ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            image_tp = 0
            # Greedy matching in descending confidence order (model outputs are score-sorted).
            for p_idx in range(len(pred_boxes)):
                if ious.shape[1] == 0:
                    break
                values = ious[p_idx].clone()
                for gt_idx in matched_gt:
                    values[gt_idx] = -1
                best_iou, best_gt = values.max(dim=0)
                if best_iou.item() >= iou_threshold:
                    image_tp += 1
                    matched_gt.add(int(best_gt))
            tp += image_tp
            fp += len(pred_boxes) - image_tp
            fn += len(gt_boxes) - image_tp
            n_images += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"images": n_images, "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "iou_threshold": iou_threshold, "score_threshold": score_threshold,
            "avg_latency_ms": sum(durations) / max(1, len(durations))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pedestrian_maskrcnn.pth")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_ds, test_ds = make_splits(seed=args.seed)
    ds = test_ds if args.split == "test" else train_ds
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model, device = load_model(args.checkpoint)
    results = evaluate(model, loader, device, args.score_threshold, args.iou_threshold)
    print(f"Device: {device}")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
