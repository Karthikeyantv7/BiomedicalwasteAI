"""
Biomedical Waste Detection - Model Training Script
Project: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation
Model: Ultralytics YOLO11n (Baseline)

Usage (Default GPU/Auto):
    python src/train.py

Custom arguments example:
    python src/train.py --epochs 100 --batch 32 --device 0 --imgsz 640
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model on Biomedical Waste Dataset"
    )
    # Model architecture / weights
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model checkpoint (default: yolo11n.pt)",
    )
    # Dataset configuration path
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to dataset YAML file (default: dataset/data.yaml)",
    )
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50; good baseline for convergence)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8; sensible baseline for 6GB-8GB+ VRAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640)",
    )
    # Hardware & Performance
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0' for GPU 0, '0,1' for multi-GPU, 'cpu' for CPU. "
             "Default automatically selects GPU if available, else CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of DataLoader worker threads (default: 4; set to 2 or 0 on Windows if encountering shared memory issues)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience: stop if no mAP improvement for N epochs (default: 15)",
    )
    # Output structure
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save training run logs and checkpoints (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="biomedical_waste_baseline",
        help="Experiment run name (default: biomedical_waste_baseline)",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=False,
        help="Overwrite existing project/name experiment folder without incrementing",
    )
    return parser.parse_args()


def detect_device(requested_device: str | None) -> str:
    """
    Selects compute device. Defaults to CUDA GPU '0' if available, otherwise 'cpu'.
    """
    if requested_device is not None and requested_device.strip() != "":
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"[Device] CUDA is available: {gpu_count} GPU(s) detected.")
        print(f"[Device] Using primary GPU: {gpu_name}")
        return "0"
    else:
        print("[Device] CUDA is NOT available. Falling back to CPU.")
        print("[Warning] Training YOLO on CPU will be extremely slow. Dedicated GPU is strongly recommended.")
        return "cpu"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_path = (project_root / args.data).resolve()

    print("=" * 70)
    print("BIOMEDICAL WASTE DETECTION - MODEL TRAINING")
    print("=" * 70)

    # Verify dataset YAML configuration
    if not data_path.exists():
        print(f"[Error] Dataset configuration not found at: {data_path}")
        print("Please verify dataset directory structure and ensure data.yaml exists.")
        sys.exit(1)

    device = detect_device(args.device)

    print(f"Dataset Config   : {data_path}")
    print(f"Base Model       : {args.model}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch Size       : {args.batch}")
    print(f"Image Size       : {args.imgsz}")
    print(f"Device           : {device}")
    print(f"Workers          : {args.workers}")
    print(f"Project Output   : {Path(args.project) / args.name}")
    print("=" * 70)

    # Initialize YOLO model with pretrained weights
    print(f"[Init] Loading pretrained weights: {args.model}...")
    model = YOLO(args.model)

    # Start training
    print("[Training] Starting training run...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        plots=True,
        verbose=True,
    )

    print("=" * 70)
    print("TRAINING FINISHED")
    print(f"Results and plots saved to : {Path(args.project) / args.name}")
    print(f"Best model weights saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("To use the trained weights for inference:")
    print(f"  python src/predict.py --weights {Path(args.project) / args.name / 'weights' / 'best.pt'} --source dataset/test/images")
    print("=" * 70)


if __name__ == "__main__":
    main()
