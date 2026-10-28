"""Fine-tune Mask R-CNN on Penn-Fudan."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import make_splits, collate_fn
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/pedestrian_maskrcnn.pth")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = make_splits(seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn)
    model = build_model(pretrained=True).to(device)
    # Fine-tune the detection and mask heads plus backbone.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        mean_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss={mean_loss:.4f} - lr={scheduler.get_last_lr()[0]:.6g}")
        # Validation loss is not calculated here because torchvision detection models
        # return losses in train mode and predictions in eval mode. Use evaluate.py
        # for held-out detection metrics after training.
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                    "seed": args.seed, "num_classes": 2}, out)
    print(f"Saved checkpoint: {out.resolve()}")
    print(f"Validation images available: {len(val_ds)}")


if __name__ == "__main__":
    main()
