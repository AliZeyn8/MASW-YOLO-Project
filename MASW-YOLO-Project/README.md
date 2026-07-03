# MASW-YOLO Project

**Multi-Scale Attention and Shrewd Weighting YOLO (MASW-YOLO)** — An improved YOLOv8 architecture for **UAV small object detection**.

## Overview

MASW-YOLO enhances YOLOv8 with three key innovations for detecting small objects in Unmanned Aerial Vehicle (UAV) imagery:

1. **MSCA (Multi-Scale Convolution Attention)** — A lightweight attention module inserted into the backbone to enhance feature extraction at multiple scales using strip convolutions.
2. **AFPN (Asymptotic Feature Pyramid Network)** — Replaces the original PAN-FPN neck with a progressive adjacent-level fusion strategy that reduces semantic gaps between non-adjacent feature layers.
3. **Wise-IoU Loss** — A dynamic non-monotonic focusing mechanism for bounding box regression that improves localization for small objects.
4. **Soft-NMS** — Decays rather than discards overlapping detection scores, retaining true positives in dense object scenarios.

## Project Structure

```
MASW-YOLO-Project/
├── data/                       # Dataset files
│   ├── raw/                    # Raw dataset (VisDrone, etc.)
│   ├── processed/              # Preprocessed data
│   └── dataset.yaml            # Dataset configuration
├── src/                        # Source code
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── yolov8n.yaml        # Base YOLOv8n config
│   │   └── modules/            # Custom modules
│   │       ├── __init__.py
│   │       ├── msca.py         # Multi-Scale Convolution Attention
│   │       └── afpn.py         # Asymptotic Feature Pyramid Network
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── loss.py             # Wise-IoU loss
│   │   ├── nms.py              # Soft-NMS
│   │   └── metrics.py          # Evaluation metrics
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── configs/                    # Configuration files
│   ├── base_config.yaml        # Default hyperparameters
│   └── experiment_1.yaml       # Experiment-specific config
├── runs/                       # Training and evaluation outputs
│   └── train/
├── notebooks/                  # Jupyter notebooks for analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py --config configs/base_config.yaml
```

### Evaluation

```bash
python src/evaluate.py --weights runs/train/MASW-YOLO/weights/best.pt --data data/dataset.yaml
```

## Dataset

Designed for the VisDrone dataset (10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor).

## Reference

This project implements concepts from:
- MASW-YOLO (Improved YOLOv8 for UAV Small Object Detection)
- Wise-IoU (arXiv:2301.10051)
- Soft-NMS (arXiv:1704.04503)

## License

MIT
