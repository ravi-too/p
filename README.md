# Pedestrian Vision — End-to-End Computer Vision Project

**Task:** pedestrian detection (bounding boxes) with optional instance masks  
**Dataset:** Penn-Fudan Pedestrian Dataset  
**Model:** Torchvision Mask R-CNN with a pretrained COCO backbone, fine-tuned for one class  
**Interfaces:** command-line inference, FastAPI, Streamlit demo

> This repository is a reproducible project template. It does not pretend that training has already been run: metrics and model weights are created by running the commands below. Pretrained weights may be downloaded by Torchvision on first use.

## 1. Why this project?

Pedestrian detection is a practical computer-vision task. Penn-Fudan is small enough for a free CPU environment, while Mask R-CNN lets us train on instance masks and use its bounding-box predictions for detection. We compare a pretrained baseline with a fine-tuned model and measure detection quality and latency.

## 2. Project structure

```text
pedestrian-vision/
├── src/
│   ├── dataset.py       # download, parse masks, dataset
│   ├── model.py         # model creation and checkpoint loading
│   ├── train.py         # fine-tuning loop
│   ├── evaluate.py      # IoU-based precision/recall and latency
│   ├── predict.py       # inference and visualisation
│   ├── api.py            # FastAPI endpoint
│   └── app.py            # Streamlit demo
├── tests/
│   └── test_dataset.py
├── requirements.txt
└── .gitignore
```

## 3. Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support, install the PyTorch/Torchvision build appropriate for your CUDA version from the official PyTorch installation selector.

## 4. Download and inspect data

The dataset is downloaded automatically into `data/` on first run.

```bash
python -m src.dataset
```

Penn-Fudan provides pedestrian instance masks. The code derives bounding boxes from each instance mask, so it does not require a separate annotation conversion step.

## 5. Train

```bash
python -m src.train --epochs 5 --batch-size 2 --output checkpoints/pedestrian_maskrcnn.pth
```

For a quick smoke test, use `--epochs 1`. On CPU, training can take a while; a free GPU notebook environment is recommended if available. The backbone is initialized from COCO pretrained weights and the prediction heads are adapted to background + pedestrian classes.

## 6. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/pedestrian_maskrcnn.pth --split test
```

The script reports precision, recall and F1 at IoU=0.5, plus average inference latency. The provided split is a deterministic random split of Penn-Fudan images. For a more rigorous evaluation, keep the test split untouched during all tuning and run several seeds.

## 7. Predict on a new image

```bash
python -m src.predict --checkpoint checkpoints/pedestrian_maskrcnn.pth --image path/to/image.jpg --output outputs/prediction.jpg
```

## 8. Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Send an image to `POST /predict`. The endpoint returns boxes, confidence scores and an annotated image encoded as base64.

## 9. Run the web demo

```bash
streamlit run src/app.py
```

Upload an image and inspect the detections. Set `MODEL_CHECKPOINT` to change the checkpoint path if needed.

## 10. Experiments to report

Run and record at least these experiments:

1. **Pretrained baseline vs fine-tuned model:** evaluate both on the same held-out test split.
2. **Training duration:** compare 1, 3 and 5 epochs, keeping the split fixed.
3. **Confidence threshold:** compare 0.3, 0.5 and 0.7; record precision/recall/F1.
4. **Inference resize:** compare image sizes 640 and 480; record latency and detection metrics.

Do not fill in results before running experiments. Save each run's terminal output or redirect it to a file. Example:

```bash
python -m src.evaluate --checkpoint checkpoints/pedestrian_maskrcnn.pth --score-threshold 0.5 | tee results_finetuned.txt
```

### Results table template

| Experiment | Setting | Precision | Recall | F1 | Avg latency (ms) |
|---|---|---:|---:|---:|---:|
| Baseline | COCO pretrained | — | — | — | — |
| Fine-tuning | 1 epoch | — | — | — | — |
| Fine-tuning | 3 epochs | — | — | — | — |
| Fine-tuning | 5 epochs | — | — | — | — |
| Threshold | 0.3 / 0.5 / 0.7 | — | — | — | — |
| Resize | 640 vs 480 | — | — | — | — |

## 11. Design choices and limitations

- **Why Mask R-CNN?** It supports instance segmentation and object detection in a single model. We use the boxes for the core task and can display masks too.
- **Why transfer learning?** Penn-Fudan is small; starting from COCO pretrained weights is more data-efficient than training from scratch.
- **Why IoU-based metrics?** A prediction counts as a true positive only when its box overlaps a ground-truth box sufficiently (IoU >= 0.5), with one-to-one matching.
- **Limitations:** small dataset, one object class, limited scene diversity, possible domain shift in night/rain/crowded scenes. Metrics from this dataset do not guarantee performance in production.
- **Optimization:** score threshold changes precision/recall; resize changes the latency/quality trade-off. Measure on the same hardware and split.

## 12. Technical report outline

1. Problem and use case
2. Dataset and preprocessing
3. Model architecture and transfer learning
4. Experimental setup and reproducibility
5. Metrics and error analysis (false positives, false negatives, poor localization)
6. Latency/resize trade-off
7. Strengths, weaknesses and future improvements
8. AI tools used: disclose tools and describe how code/claims were checked with tests, documentation and experiments.

## 13. Interview preparation

Be ready to explain:
- what an instance mask is and how bounding boxes are extracted from it;
- why the model has two classes (background and pedestrian);
- train/eval mode, optimizer, learning rate and transfer learning;
- IoU, precision, recall and F1;
- why changing the confidence threshold trades precision against recall;
- how you prevented test-set leakage;
- the difference between model accuracy and inference latency.

## License and data

Check the dataset's own terms before redistribution. This repository downloads data on demand and does not include the dataset or pretrained model weights.
