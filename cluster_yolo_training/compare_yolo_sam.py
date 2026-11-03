#!/usr/bin/env python3
"""
Comparative Evaluation: Pure YOLO vs. YOLO + SAM on Insect Anatomy Validation Set.

- Evaluates validation images against ground-truth polygon annotations.
- Computes mean IoU per anatomical class for:
    1) Pure YOLO-seg
    2) YOLO + SAM (bounding-box prompted MobileSAM)
- Renders 4-panel visual comparison figures:
    [Original Image] | [Ground Truth] | [Pure YOLO] | [YOLO + SAM]
"""

import argparse
import glob
import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO, SAM

CLASS_NAMES = ["full-body", "wing", "leg", "antenna", "head", "body"]
CLASS_COLORS = {
    "full-body": (255, 0, 0),      # Blue
    "wing":      (0, 165, 255),    # Orange
    "leg":       (0, 255, 0),      # Green
    "antenna":   (255, 255, 0),    # Cyan
    "head":      (0, 0, 255),      # Red
    "body":      (255, 0, 255),    # Magenta
}


def load_ground_truth(txt_path, img_w, img_h):
    """Parses normalized YOLO polygon annotations from a .txt file."""
    gt_by_class = defaultdict(list)
    if not os.path.exists(txt_path):
        return gt_by_class

    with open(txt_path, "r", encoding="utf-8") as fp:
        for line in fp:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            cls_id = int(tokens[0])
            coords = [float(x) for x in tokens[1:]]
            pts = []
            for i in range(0, len(coords) - 1, 2):
                px = int(round(coords[i] * img_w))
                py = int(round(coords[i + 1] * img_h))
                pts.append([px, py])
            if len(pts) >= 3:
                cname = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                gt_by_class[cname].append(np.array(pts, dtype=np.int32))

    return gt_by_class


def polygons_to_class_mask(polygons, img_shape):
    """Converts a list of polygon contours into a binary mask."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [poly], 1)
    return mask


def compute_iou(mask1, mask2):
    """Calculates Intersection-over-Union between two binary masks."""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def draw_polygons_overlay(img_bgr, polygons_by_class, alpha=0.45):
    """Draws semi-transparent colored masks and borders on an image."""
    overlay = img_bgr.copy()
    borders = img_bgr.copy()

    for cname, polys in polygons_by_class.items():
        color = CLASS_COLORS.get(cname, (200, 200, 200))
        for poly in polys:
            if len(poly) >= 3:
                cv2.fillPoly(overlay, [poly], color)
                cv2.polylines(borders, [poly], isClosed=True, color=color, thickness=2)

    blended = cv2.addWeighted(overlay, alpha, borders, 1.0 - alpha, 0)
    return blended


def evaluate_sample(img_path, txt_path, yolo_model, sam_model, conf=0.25):
    """Evaluates a single validation image with both Pure YOLO and YOLO + SAM."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]

    # 1. Ground Truth
    gt_by_class = load_ground_truth(txt_path, w, h)

    # 2. Pure YOLO Prediction
    yolo_res = yolo_model.predict(img_path, conf=conf, verbose=False)[0]
    yolo_polys = defaultdict(list)
    boxes = []
    box_classes = []

    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
        cls_ids = yolo_res.boxes.cls.cpu().numpy()
        boxes = yolo_res.boxes.xyxy.cpu().numpy()
        box_classes = [yolo_model.names[int(c)] for c in cls_ids]

        if yolo_res.masks is not None:
            for poly, cname in zip(yolo_res.masks.xy, box_classes):
                if len(poly) >= 3:
                    yolo_polys[cname].append(np.array(poly, dtype=np.int32))

    # 3. YOLO + SAM Prediction
    sam_polys = defaultdict(list)
    if len(boxes) > 0 and sam_model is not None:
        sam_res = sam_model(img_path, bboxes=boxes, verbose=False)[0]
        if sam_res.masks is not None:
            for poly, cname in zip(sam_res.masks.xy, box_classes):
                if len(poly) >= 3:
                    sam_polys[cname].append(np.array(poly, dtype=np.int32))

    # 4. Compute per-class IoU against Ground Truth
    yolo_ious = {}
    sam_ious = {}

    all_present_classes = set(gt_by_class.keys()).union(yolo_polys.keys()).union(sam_polys.keys())
    for cname in all_present_classes:
        gt_mask = polygons_to_class_mask(gt_by_class.get(cname, []), (h, w))
        yolo_mask = polygons_to_class_mask(yolo_polys.get(cname, []), (h, w))
        sam_mask = polygons_to_class_mask(sam_polys.get(cname, []), (h, w))

        # Only evaluate IoU if class is present in GT or predictions
        if gt_mask.sum() > 0 or yolo_mask.sum() > 0:
            yolo_ious[cname] = compute_iou(gt_mask, yolo_mask)
        if gt_mask.sum() > 0 or sam_mask.sum() > 0:
            sam_ious[cname] = compute_iou(gt_mask, sam_mask)

    return {
        "img_bgr": img_bgr,
        "gt_polys": gt_by_class,
        "yolo_polys": yolo_polys,
        "sam_polys": sam_polys,
        "yolo_ious": yolo_ious,
        "sam_ious": sam_ious,
        "filename": os.path.basename(img_path),
    }


def load_sam_model(sam_path):
    """Loads base MobileSAM and optionally applies fine-tuned decoder weights."""
    base_sam_path = "cluster_yolo_training/mobile_sam.pt"
    if not os.path.exists(base_sam_path):
        base_sam_path = "mobile_sam.pt"
    sam_model = SAM(base_sam_path)

    if sam_path and os.path.exists(sam_path) and sam_path != base_sam_path:
        import torch
        ckpt = torch.load(sam_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sam_model.model.load_state_dict(ckpt["model_state_dict"])
            print(f"✔ Successfully loaded fine-tuned SAM weights from: {sam_path}")
        elif isinstance(ckpt, dict) and "mask_decoder_state_dict" in ckpt:
            sam_model.model.mask_decoder.load_state_dict(ckpt["mask_decoder_state_dict"])
            print(f"✔ Successfully loaded fine-tuned Mask Decoder from: {sam_path}")
    return sam_model


def main():
    parser = argparse.ArgumentParser(description="Compare Pure YOLO vs YOLO+SAM")
    parser.add_argument("--yolo", type=str, default="cluster_yolo_training/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--sam", type=str, default="cluster_yolo_training/mobile_sam.pt", help="Path to SAM weights (base or fine-tuned)")
    parser.add_argument("--val-dir", type=str, default="cluster_yolo_training/dataset/images/val", help="Validation images directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of validation images to evaluate and render (default: 10)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--output-dir", type=str, default="cluster_yolo_training/runs/yolo_vs_sam", help="Directory to save visual comparison plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 COMPARATIVE BENCHMARK: PURE YOLO vs. YOLO + SAM")
    print("=" * 80)
    print(f"• YOLO Model:   {args.yolo}")
    print(f"• SAM Model:    {args.sam}")
    print(f"• Validation:   {args.val_dir}")
    print(f"• Output plots: {args.output_dir}")
    print("=" * 80)

    yolo_model = YOLO(args.yolo)
    sam_model = load_sam_model(args.sam)

    val_imgs = sorted(glob.glob(os.path.join(args.val_dir, "*.jpg")))
    if not val_imgs:
        print(f"❌ No validation images found in {args.val_dir}")
        return

    eval_samples = val_imgs[:args.num_samples] if args.num_samples > 0 else val_imgs
    print(f"Evaluating {len(eval_samples)} validation samples...\n")

    all_yolo_ious = defaultdict(list)
    all_sam_ious = defaultdict(list)

    for idx, img_path in enumerate(eval_samples):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")

        res = evaluate_sample(img_path, txt_path, yolo_model, sam_model, conf=args.conf)
        if res is None:
            continue

        # Accumulate metrics
        for cname, score in res["yolo_ious"].items():
            all_yolo_ious[cname].append(score)
        for cname, score in res["sam_ious"].items():
            all_sam_ious[cname].append(score)

        # Render 4-panel visual comparison
        orig_rgb = cv2.cvtColor(res["img_bgr"], cv2.COLOR_BGR2RGB)
        gt_vis = cv2.cvtColor(draw_polygons_overlay(res["img_bgr"], res["gt_polys"]), cv2.COLOR_BGR2RGB)
        yolo_vis = cv2.cvtColor(draw_polygons_overlay(res["img_bgr"], res["yolo_polys"]), cv2.COLOR_BGR2RGB)
        sam_vis = cv2.cvtColor(draw_polygons_overlay(res["img_bgr"], res["sam_polys"]), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original: {res['filename']}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_vis)
        axes[1].set_title("Ground Truth (Human)", fontsize=11, fontweight="bold", color="darkgreen")
        axes[1].axis("off")

        mean_yolo_iou = np.mean(list(res["yolo_ious"].values())) if res["yolo_ious"] else 0.0
        axes[2].imshow(yolo_vis)
        axes[2].set_title(f"Pure YOLO-seg (mIoU: {mean_yolo_iou:.1%})", fontsize=11, fontweight="bold", color="darkblue")
        axes[2].axis("off")

        mean_sam_iou = np.mean(list(res["sam_ious"].values())) if res["sam_ious"] else 0.0
        axes[3].imshow(sam_vis)
        axes[3].set_title(f"YOLO + SAM (mIoU: {mean_sam_iou:.1%})", fontsize=11, fontweight="bold", color="purple")
        axes[3].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"cmp_{idx:02d}_{stem}.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[{idx+1:>2}/{len(eval_samples)}] {res['filename']:<35} -> Pure YOLO: {mean_yolo_iou:>5.1%} | YOLO+SAM: {mean_sam_iou:>5.1%}")

    # Final Quantitative Benchmark Table
    print("\n" + "=" * 80)
    print("📊 QUANTITATIVE BENCHMARK: PURE YOLO vs. YOLO + SAM (mIoU on Validation Set)")
    print("=" * 80)
    print(f"{'ANATOMICAL CLASS':<18} | {'PURE YOLO mIoU':<16} | {'YOLO + SAM mIoU':<16} | {'DELTA':<10}")
    print("-" * 80)

    class_deltas = []
    for cname in CLASS_NAMES:
        y_scores = all_yolo_ious.get(cname, [])
        s_scores = all_sam_ious.get(cname, [])
        y_mean = np.mean(y_scores) if y_scores else 0.0
        s_mean = np.mean(s_scores) if s_scores else 0.0
        delta = s_mean - y_mean
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.1%}"
        print(f"{cname:<18} | {y_mean:>14.1%} | {s_mean:>14.1%} | {delta_str:>10}")
        if y_scores or s_scores:
            class_deltas.append(delta)

    print("=" * 80)
    overall_yolo = np.mean([np.mean(v) for v in all_yolo_ious.values()]) if all_yolo_ious else 0.0
    overall_sam = np.mean([np.mean(v) for v in all_sam_ious.values()]) if all_sam_ious else 0.0
    overall_delta = overall_sam - overall_yolo
    delta_sym = "+" if overall_delta >= 0 else ""
    print(f"{'OVERALL mIoU':<18} | {overall_yolo:>14.1%} | {overall_sam:>14.1%} | {delta_sym}{overall_delta:>9.1%}")
    print("=" * 80)
    print(f"✔ Visual comparison plots saved in: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
