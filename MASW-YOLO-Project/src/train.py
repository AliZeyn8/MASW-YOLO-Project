"""
MASW-YOLO Training Script.

Main entry point for training the full MASW-YOLO model (MSCA backbone +
AFPN neck + WiseIoU loss + Soft-NMS).

Usage:
    python train.py --config configs/base_config.yaml
"""

import argparse
import os
import sys

import yaml

# Add project root to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_trainer import MASWTrainer


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path: str) -> None:
    """
    Train the MASW-YOLO model.

    Loads hyperparameters from the YAML config and passes them as overrides
    to the MASWTrainer. The trainer handles:
        - Custom model building (MASWDetectionModel with MSCA / AFPN layers)
        - WiseIoU loss for bounding-box regression
        - Soft-NMS during validation

    Args:
        config_path (str): Path to the training configuration file.
    """
    config = load_config(config_path)

    # Prepare overrides dict for the trainer
    overrides = {
        "model": os.path.join(
            os.path.dirname(__file__), "models", config.get("model", "yolov8n.yaml")
        ),
        "data": config.get("data", "data/dataset.yaml"),
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch", 16),
        "imgsz": config.get("imgsz", 640),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 3.0),
        "warmup_momentum": config.get("warmup_momentum", 0.8),
        "warmup_bias_lr": config.get("warmup_bias_lr", 0.1),
        "optimizer": config.get("optimizer", "SGD"),
        "device": config.get("device", 0),
        "workers": config.get("workers", 8),
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 0.5),
        "dfl": config.get("dfl", 1.5),
        "project": config.get("project", "runs/train"),
        "name": config.get("name", "MASW-YOLO"),
        "exist_ok": True,
        # Wise-IoU parameters
        "wiou_version": config.get("wiou_version", "v3"),
        "wiou_beta": config.get("wiou_beta", 1.0),
        "wiou_delta": config.get("wiou_delta", 0.5),
        # Soft-NMS parameters
        "soft_sigma": config.get("soft_sigma", 0.5),
        "soft_score_thr": config.get("soft_score_thr", 0.001),
    }

    print(f"Starting MASW-YOLO training with config: {config_path}")
    print(f"Model: {overrides['model']}")
    print(f"Epochs: {overrides['epochs']} | Batch: {overrides['batch']} | ImgSz: {overrides['imgsz']}")

    # Instantiate the MASW trainer and start training
    trainer = MASWTrainer(overrides=overrides)
    trainer.train()

    print(f"Training completed. Results saved to: {trainer.save_dir}")


def main():
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train MASW-YOLO model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    train_model(args.config)


if __name__ == "__main__":
    main()
