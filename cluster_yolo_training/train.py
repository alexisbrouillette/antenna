#!/usr/bin/env python3
"""
Robust YOLO-seg fine-tuning script.
Incorporates synthetic field-mimic degradation and domain adaptation.
"""

import argparse
import os
import shutil
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train robust YOLO segmentation on insect parts")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16 on GPU, 4-8 on CPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640 matches field crop scale)")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt", help="Pretrained base model (e.g. yolov8s-seg.pt or yolo11s-seg.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: '0', 'cuda', or 'cpu'")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 65)
    print("🦋 INSECT ANATOMY YOLO-SEG TRAINING (FIELD-ADAPTED v2)")
    print("=" * 65)
    print(f"Device:       {device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and device != "cpu":
        print(f"GPU Name:     {torch.cuda.get_device_name(0)}")
    print(f"Base model:   {args.model} (~11.2M params for fine anatomical structures)")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch}")
    print(f"Image size:   {args.imgsz}px (calibrated for camera trap crops)")
    print("=" * 65)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")

    # Resolve model path locally if available in script directory
    if not os.path.exists(args.model) and os.path.exists(os.path.join(script_dir, args.model)):
        args.model = os.path.join(script_dir, args.model)

    # Ensure dataset.yaml points to current location (works anywhere folder is copied)
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"""path: {dataset_dir}
train: images/train
val: images/val

names:
  0: full-body
  1: wing
  2: leg
  3: antenna
  4: head
  5: body
""")

    # Load pretrained model
    model = YOLO(args.model)

    # Train with domain adaptation augmentations (lighting, scale, rotation)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,
        workers=4,
        project=os.path.join(script_dir, "runs"),
        name="insect_anatomy_v2",
        exist_ok=True,
        save=True,
        plots=True,
        cos_lr=True,            # Cosine learning rate schedule for smoother convergence
        # --- Data Augmentation for Natural Lighting & Camera Trap Diversity ---
        hsv_h=0.015,            # Subtle color hue variation
        hsv_s=0.7,              # Saturation variation (dull sheet vs vibrant outdoor light)
        hsv_v=0.4,              # Brightness / contrast variation (harsh sun vs flash vs shadows)
        degrees=30.0,           # Random rotation up to +/- 30 deg (moths land at any angle)
        translate=0.1,          # Translation jitter
        scale=0.5,              # Scale variation (+/- 50% zoom for crops vs full view)
        shear=2.0,              # Slight perspective shear
        fliplr=0.5,             # Horizontal flip (50%)
        flipud=0.5,             # Vertical flip (50%)
        mosaic=1.0,             # 4-image mosaic for multi-scale and contextual diversity
        mixup=0.15,             # Mixup blending for semi-transparent wings / complex backgrounds
        close_mosaic=15         # Turn off mosaic last 15 epochs for clean real-boundary learning
    )

    # Copy best weights to root weights/ directory for easy access
    weights_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_src = os.path.join(script_dir, "runs", "insect_anatomy_v2", "weights", "best.pt")
    if not os.path.exists(best_src):
        best_src = os.path.join(script_dir, "runs", "insect_anatomy", "weights", "best.pt")
    best_dst = os.path.join(weights_dir, "best.pt")

    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\n🎉 Training complete! Best weights saved to:\n   {best_dst}")
        
        # Export to ONNX (for AnyLabeling or CPU deployment)
        try:
            print("\n📦 Exporting best model to ONNX format...")
            best_model = YOLO(best_dst)
            onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)
            print(f"✔ ONNX model exported: {onnx_path}")
        except Exception as e:
            print(f"⚠️ Note: ONNX export skipped ({e})")
    else:
        print("\n⚠️ Training finished, check runs/insect_anatomy_v2 for results.")

if __name__ == "__main__":
    main()
