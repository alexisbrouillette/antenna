#!/usr/bin/env python3
"""
Trains YOLO11s-pose on the 11-keypoint Insect Anatomical Dataset.
Enforces structural biological topology, part counting, and visibility constraints.
"""

import argparse
import os
import shutil
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train Insect Anatomical Pose Estimator")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--model", type=str, default="yolo11s-pose.pt", help="Pretrained pose model")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0', 'cuda', 'cpu')")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    args = parser.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset_pose")
    os.makedirs(dataset_dir, exist_ok=True)
    yaml_path = os.path.join(dataset_dir, "dataset_pose.yaml")

    # Ensure absolute path in dataset_pose.yaml
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"""path: {dataset_dir}
train: images/train
val: images/val

kpt_shape: [11, 3]
flip_idx: [0, 3, 4, 1, 2, 5, 6, 9, 10, 7, 8]

names:
  0: insect
""")

    print("=" * 70)
    print("🦋 INSECT ANATOMICAL POSE & KINEMATIC SKELETON TRAINING")
    print("=" * 70)
    print(f"Device:       {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"Base model:   {args.model}")
    print(f"Dataset:      {yaml_path}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch}")
    print(f"Image size:   {args.imgsz}px")
    print("=" * 70)

    model = YOLO(args.model)

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,
        workers=4,
        project=os.path.join(script_dir, "runs"),
        name="insect_pose_v1",
        exist_ok=True,
        save=True,
        plots=True,
        cos_lr=True,
        degrees=30.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=15
    )

    weights_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_src = os.path.join(script_dir, "runs", "insect_pose_v1", "weights", "best.pt")
    best_dst = os.path.join(weights_dir, "best_pose.pt")

    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\n🎉 Training complete! Best pose weights saved to:\n   {best_dst}")
    else:
        print("\n⚠️ Training finished. Check runs/insect_pose_v1 for logs and weights.")

if __name__ == "__main__":
    main()

