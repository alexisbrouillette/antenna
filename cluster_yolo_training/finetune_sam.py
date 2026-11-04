#!/usr/bin/env python3
"""
Fine-tunes MobileSAM's Mask Decoder on Insect Anatomical Masks.

Key Architecture Strategy:
- Image Encoder (TinyViT, 6.0M params): FROZEN (Preserves massive generic vision representations).
- Mask Decoder (4.0M params) & Prompt Encoder: TRAINABLE.
- Prompts: Bounding boxes around insect parts (wings, legs, body, head, antennae).
- Loss Function: Binary Cross Entropy + 2.0 * Dice Loss (optimized for sharp boundaries and thin structures).
- Output: cluster_yolo_training/weights/mobile_sam_finetuned.pt
"""

import argparse
import glob
import os
import random
import time

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ultralytics import SAM


class InsectSAMDataset(Dataset):
    """
    Loads insect camera trap images and their ground-truth polygon annotations,
    letterboxing to 1024x1024 and producing bounding box prompts with target 256x256 masks.
    """
    def __init__(self, img_dir, lbl_dir, imgsz=1024):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.lbl_dir = lbl_dir
        self.imgsz = imgsz
        self.samples = []

        for p in self.img_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl_file = os.path.join(lbl_dir, f"{stem}.txt")
            if os.path.exists(lbl_file) and os.path.getsize(lbl_file) > 0:
                self.samples.append((p, lbl_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black canvas
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        resized_img = cv2.resize(img, (new_w, new_h))
        canvas[:new_h, :new_w] = resized_img

        # Convert BGR to RGB and normalize to [0, 1] then standardize for SAM
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm_img = (canvas_rgb - pixel_mean) / pixel_std
        img_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, 1024, 1024)

        boxes = []
        masks = []

        with open(lbl_path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = [float(x) for x in parts[1:]]
                pts = []
                for i in range(0, len(coords) - 1, 2):
                    px = int(round(coords[i] * w * scale))
                    py = int(round(coords[i + 1] * h * scale))
                    pts.append([px, py])

                if len(pts) >= 3:
                    pts_arr = np.array(pts, dtype=np.int32)
                    # Bounding box with small padding
                    xmin = max(0, pts_arr[:, 0].min() - 2)
                    ymin = max(0, pts_arr[:, 1].min() - 2)
                    xmax = min(self.imgsz, pts_arr[:, 0].max() + 2)
                    ymax = min(self.imgsz, pts_arr[:, 1].max() + 2)
                    boxes.append([xmin, ymin, xmax, ymax])

                    # Binary ground truth mask (at 256x256 SAM low-res mask resolution)
                    m1024 = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
                    cv2.fillPoly(m1024, [pts_arr], 1)
                    m256 = cv2.resize(m1024, (256, 256), interpolation=cv2.INTER_NEAREST)
                    masks.append(m256)

        if not boxes:
            # Fallback dummy box to prevent zero-shape crashes
            boxes.append([0, 0, 10, 10])
            masks.append(np.zeros((256, 256), dtype=np.uint8))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)       # (N, 4)
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32)  # (N, 256, 256)

        return img_tensor, boxes_tensor, masks_tensor, os.path.basename(img_path)


def collate_fn(batch):
    # Custom collate because each image has variable number of insect parts (N)
    img_tensors, boxes_list, masks_list, names = zip(*batch)
    return torch.stack(img_tensors, 0), list(boxes_list), list(masks_list), list(names)


def dice_loss(pred_logits, targets, smooth=1.0):
    """Computes soft Dice loss for binary mask predictions."""
    probs = torch.sigmoid(pred_logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    return 1.0 - dice


def compute_miou(pred_logits, targets, thresh=0.0):
    """Computes mean IoU between predicted logits and binary targets."""
    preds = (pred_logits > thresh).float()
    inter = (preds * targets).sum().item()
    union = ((preds + targets) > 0).float().sum().item()
    return (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileSAM Mask Decoder on Insect Parts")
    parser.add_argument("--sam-base", type=str, default="cluster_yolo_training/mobile_sam.pt", help="Path to base MobileSAM weights")
    parser.add_argument("--dataset", type=str, default="cluster_yolo_training/dataset", help="Path to reconstructed YOLO dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', '0', 'cpu')")
    parser.add_argument("--output-dir", type=str, default="cluster_yolo_training/weights", help="Directory to save checkpoint")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    # Resolve paths relative to script directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.sam_base) and os.path.exists(os.path.join(script_dir, os.path.basename(args.sam_base))):
        args.sam_base = os.path.join(script_dir, os.path.basename(args.sam_base))
    if not os.path.exists(args.dataset) and os.path.exists(os.path.join(script_dir, "dataset")):
        args.dataset = os.path.join(script_dir, "dataset")
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, "weights")

    print("=" * 80)
    print("🦋 FINE-TUNING MOBILESAM MASK DECODER ON INSECT ANATOMY")
    print("=" * 80)
    print(f"Device:       {device}")
    if torch.cuda.is_available() and device.type != "cpu":
        print(f"GPU Name:     {torch.cuda.get_device_name(0)}")
    print(f"Base SAM:     {args.sam_base}")
    print(f"Dataset:      {args.dataset}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Learning Rate:{args.lr}")
    print("=" * 80)

    # 1. Load SAM model
    assert os.path.exists(args.sam_base), f"❌ SAM base weights not found at: {args.sam_base}"
    sam_wrapper = SAM(args.sam_base)
    model = sam_wrapper.model.to(device)

    # 2. Freeze Image Encoder & Set Decoder Trainable
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    
    trainable_params = []
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
        trainable_params.append(p)
    for p in model.prompt_encoder.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"✔ Frozen parameters (Image Encoder): {num_frozen:,}")
    print(f"✔ Trainable parameters (Mask Decoder): {num_trainable:,} (lightweight & fast!)\n")

    # 3. Create Datasets & DataLoaders
    train_img_dir = os.path.join(args.dataset, "images", "train")
    train_lbl_dir = os.path.join(args.dataset, "labels", "train")
    val_img_dir = os.path.join(args.dataset, "images", "val")
    val_lbl_dir = os.path.join(args.dataset, "labels", "val")

    train_ds = InsectSAMDataset(train_img_dir, train_lbl_dir)
    val_ds = InsectSAMDataset(val_img_dir, val_lbl_dir)

    print(f"📦 Training samples:   {len(train_ds)} images")
    print(f"📦 Validation samples: {len(val_ds)} images\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_val_miou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    os.makedirs(args.output_dir, exist_ok=True)
    best_weights_path = os.path.join(args.output_dir, "mobile_sam_finetuned.pt")

    # 5. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.mask_decoder.train()
        model.prompt_encoder.train()
        model.image_encoder.eval()

        epoch_loss = 0.0
        total_batches = 0

        for imgs, boxes_list, masks_list, _ in train_loader:
            imgs = imgs.to(device)

            # A. Extract frozen image embeddings
            with torch.no_grad():
                image_embeddings = model.image_encoder(imgs)  # (B, 256, 64, 64)

            image_pe = model.prompt_encoder.get_dense_pe()
            batch_loss = 0.0
            valid_items = 0

            # B. Forward through prompt encoder + mask decoder per sample in batch
            for b_idx in range(len(imgs)):
                boxes = boxes_list[b_idx].to(device)      # (N, 4)
                gt_masks = masks_list[b_idx].to(device)   # (N, 256, 256)

                if boxes.shape[0] == 0:
                    continue

                # Pass boxes of shape (N, 4) directly
                sparse_pe, dense_pe = model.prompt_encoder(points=None, boxes=boxes, masks=None)

                # Decode masks: (N, 1, 256, 256)
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings[b_idx:b_idx+1],
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_pe,
                    dense_prompt_embeddings=dense_pe,
                    multimask_output=False
                )

                # Squeeze channel dim -> (N, 256, 256)
                pred_masks = low_res_masks.squeeze(1)

                # C. Compute Combined Loss
                bce = bce_loss_fn(pred_masks, gt_masks)
                dice = dice_loss(pred_masks, gt_masks)
                item_loss = bce + 2.0 * dice

                batch_loss = batch_loss + item_loss
                valid_items += 1

            if valid_items > 0:
                loss = batch_loss / valid_items
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, total_batches)

        # 6. Validation Evaluation
        model.mask_decoder.eval()
        model.prompt_encoder.eval()

        val_loss = 0.0
        val_miou_list = []
        val_batches = 0

        with torch.no_grad():
            for imgs, boxes_list, masks_list, _ in val_loader:
                imgs = imgs.to(device)
                image_embeddings = model.image_encoder(imgs)
                image_pe = model.prompt_encoder.get_dense_pe()

                for b_idx in range(len(imgs)):
                    boxes = boxes_list[b_idx].to(device)
                    gt_masks = masks_list[b_idx].to(device)
                    if boxes.shape[0] == 0:
                        continue

                    sparse_pe, dense_pe = model.prompt_encoder(points=None, boxes=boxes, masks=None)
                    low_res_masks, _ = model.mask_decoder(
                        image_embeddings=image_embeddings[b_idx:b_idx+1],
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_pe,
                        dense_prompt_embeddings=dense_pe,
                        multimask_output=False
                    )
                    pred_masks = low_res_masks.squeeze(1)

                    bce = bce_loss_fn(pred_masks, gt_masks)
                    dice = dice_loss(pred_masks, gt_masks)
                    val_loss += (bce + 2.0 * dice).item()
                    val_miou_list.append(compute_miou(pred_masks, gt_masks))
                    val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_miou = np.mean(val_miou_list) if val_miou_list else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_miou"].append(avg_val_miou)

        is_best = avg_val_miou > best_val_miou
        if is_best:
            best_val_miou = avg_val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "mask_decoder_state_dict": model.mask_decoder.state_dict(),
                "prompt_encoder_state_dict": model.prompt_encoder.state_dict(),
                "val_miou": best_val_miou,
                "val_loss": avg_val_loss,
            }, best_weights_path)

        best_marker = "⭐ BEST" if is_best else ""
        print(f"Epoch [{epoch:>2}/{args.epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:>6.1%} {best_marker}")

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 80)
    print("🎉 FINE-TUNING COMPLETED!")
    print("=" * 80)
    print(f"• Total Training Time: {total_time:.1f} minutes")
    print(f"• Peak Validation mIoU: {best_val_miou:.1%}")
    print(f"• Trained Checkpoint:  {best_weights_path}")
    print("=" * 80)

    # Plot & Save Loss Curve
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss (BCE + 2*Dice)", color="tab:blue")
    ax1.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss", color="tab:orange", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(history["val_miou"]) + 1), [x * 100 for x in history["val_miou"]], label="Val mIoU (%)", color="tab:green", linewidth=2)
    ax2.set_ylabel("mIoU (%)")
    ax2.legend(loc="upper right")

    plt.title("MobileSAM Insect Anatomy Fine-Tuning Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    curve_path = os.path.join(os.path.dirname(best_weights_path), "sam_finetune_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"✔ Training curve plot saved: {curve_path}\n")


if __name__ == "__main__":
    main()
