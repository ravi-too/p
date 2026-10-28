"""Run inference on one image and save an annotated result."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor
from src.model import load_model


def predict_image(model, device, image, score_threshold=0.5):
    tensor = pil_to_tensor(image.convert("RGB")).float() / 255.0
    with torch.inference_mode():
        output = model([tensor.to(device)])[0]
    keep = output["scores"].detach().cpu() >= score_threshold
    return {
        "boxes": output["boxes"].detach().cpu()[keep].tolist(),
        "scores": output["scores"].detach().cpu()[keep].tolist(),
        "labels": output["labels"].detach().cpu()[keep].tolist(),
    }


def annotate(image, result):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), f"pedestrian {score:.2f}", fill="red")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pedestrian_maskrcnn.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/prediction.jpg")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    model, device = load_model(args.checkpoint)
    result = predict_image(model, device, image, args.score_threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotate(image, result).save(out)
    print(f"Detections: {len(result['boxes'])}")
    print(f"Saved annotated image: {out.resolve()}")


if __name__ == "__main__":
    main()
