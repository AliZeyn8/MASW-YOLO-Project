"""
MASW-YOLO Evaluation Script.

Evaluates a trained MASW-YOLO model on a validation/test dataset.
Computes precision, recall, and mAP metrics.

Usage:
    python evaluate.py --weights runs/train/MASW-YOLO/weights/best.pt --data data/dataset.yaml
"""

import argparse
import os
import sys

from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model(weights_path: str, data_path: str, imgsz: int = 640) -> None:
    """
    Evaluate a trained MASW-YOLO model.

    Args:
        weights_path (str): Path to the trained model weights (.pt file).
        data_path (str): Path to the dataset configuration file.
        imgsz (int): Input image size for evaluation. Default: 640.
    """
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)

    print(f"Evaluating on dataset: {data_path}")
    print("Evaluation started...")

    # Run validation
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=16,
        device=0,
        project="runs/eval",
        name="MASW-YOLO-eval",
        exist_ok=True,
    )

    print(f"Evaluation completed.")
    if results:
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.p:.4f}")
        print(f"Recall: {results.box.r:.4f}")


def main():
    """Parse command-line arguments and launch evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate MASW-YOLO model")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    args = parser.parse_args()

    evaluate_model(args.weights, args.data, args.imgsz)


if __name__ == "__main__":
    main()
