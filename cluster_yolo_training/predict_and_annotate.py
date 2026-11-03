#!/usr/bin/env python3
"""
Batch Pre-Annotator:
Runs the trained YOLO-seg model on unannotated images and generates
LabelMe / AnyLabeling .json files with predicted polygon masks.

Rules:
1. NEVER overwrites existing manual annotations.
2. Clearly tags AI-generated annotations in both file flags and shape descriptions.
3. Supports testing on specific species or tracks (field / inat).
"""

import argparse
import glob
import json
import os
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from ultralytics import YOLO

def is_protected_annotation(json_path):
    """
    Returns True if an annotation is:
    - Manual (not ai_generated)
    - Human-verified (checked=True or status='verified')
    - Human-modified (shape description edited or added by human)
    These must NEVER be overwritten!
    """
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        # 1. Human verified / checked
        if data.get("checked") is True:
            return True
        flags = data.get("flags", {})
        if flags and flags.get("status") == "verified":
            return True
        if not flags or not flags.get("ai_generated"):
            return True
        # 2. Check if user added or modified any shape
        shapes = data.get("shapes", [])
        if any(not s.get("description", "").startswith("🤖 AI") for s in shapes):
            return True
        return False
    except Exception:
        return True  # Erring on the side of caution


def score_prediction(result):
    """
    Computes an automated quality score (0-100) based on biological sanity checks:
    - 1 full-body mask is mandatory (penalty -30 if missing, -30 if >2)
    - Wing count <= 4 (penalty -25 if >4)
    - Leg count <= 6 (penalty -20 if >6)
    - Antenna count <= 2 (penalty -15 if >2)
    - Head count <= 2 (penalty -20 if >2)
    - Penalize low-confidence detections (<0.40)
    """
    if result.masks is None or len(result.masks) == 0:
        return 0, "hard", ["No masks detected"]

    classes = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()]
    scores = result.boxes.conf.cpu().numpy()

    mean_conf = float(np.mean(scores))
    counts = Counter(classes)
    penalty = 0
    reasons = []

    # 1. Full-body sanity check
    fb = counts.get("full-body", 0)
    if fb == 1:
        pass
    elif fb == 0:
        penalty += 30
        reasons.append("Missing full-body")
    elif fb > 2:
        penalty += 30
        reasons.append(f"Multiple full-bodies ({fb})")

    # 2. Implausible part counts
    if counts.get("wing", 0) > 4:
        penalty += 25
        reasons.append(f"Too many wings ({counts['wing']})")
    if counts.get("leg", 0) > 6:
        penalty += 20
        reasons.append(f"Too many legs ({counts['leg']})")
    if counts.get("antenna", 0) > 2:
        penalty += 15
        reasons.append(f"Too many antennae ({counts['antenna']})")
    if counts.get("head", 0) > 2:
        penalty += 20
        reasons.append(f"Multiple heads ({counts['head']})")

    # 3. Weak detections (<0.40)
    low_confs = sum(1 for s in scores if s < 0.40)
    if low_confs > 0:
        penalty += min(25, low_confs * 8)
        reasons.append(f"{low_confs} weak parts")

    final_score = max(0, min(100, int(mean_conf * 100 - penalty)))
    tier = "clean" if final_score >= 70 and penalty == 0 else ("review" if final_score >= 45 else "hard")
    return final_score, tier, reasons


def annotate_images(model, img_files, conf=0.25, iou=0.45, overwrite_ai=False, weights_path="best.pt", sam_model=None):
    annotated_count = 0
    skipped_manual = 0
    skipped_existing_ai = 0

    stats = defaultdict(lambda: defaultdict(int))
    species_report = []

    for img_path in img_files:
        json_path = os.path.splitext(img_path)[0] + ".json"
        
        # 1. Protect manual and human-verified annotations
        if os.path.exists(json_path):
            if is_protected_annotation(json_path):
                skipped_manual += 1
                continue
            elif not overwrite_ai:
                skipped_existing_ai += 1
                continue

        # Get species and track from path
        parts = os.path.normpath(img_path).split(os.sep)
        track = "unknown"
        species = "unknown"
        if "field" in parts:
            track = "field"
            species = parts[parts.index("field") + 1]
        elif "inat" in parts:
            track = "inat"
            species = parts[parts.index("inat") + 1]

        # Get image dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        # Run inference
        results = model.predict(source=img_path, conf=conf, iou=iou, max_det=30, verbose=False)
        result = results[0]

        shapes = []
        part_counts = defaultdict(int)

        polygons = []
        classes = []
        confs = []

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            if sam_model is not None:
                # Use SAM to refine masks from YOLO bounding box prompts
                sam_res = sam_model(img_path, bboxes=boxes, verbose=False)[0]
                if sam_res.masks is not None:
                    polygons = sam_res.masks.xy
            
            if not polygons and result.masks is not None:
                polygons = result.masks.xy

            for poly, cls_id, score in zip(polygons, classes, confs):
                if len(poly) < 3:
                    continue
                label_name = result.names[int(cls_id)]
                part_counts[label_name] += 1
                stats[track][label_name] += 1

                points = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly]

                shapes.append({
                    "label": label_name,
                    "score": round(float(score), 4),
                    "points": points,
                    "group_id": None,
                    "description": f"🤖 AI ({label_name} {score:.0%})",
                    "difficult": False,
                    "shape_type": "polygon",
                    "flags": {"ai_predicted": True},
                    "attributes": {},
                    "kie_linking": []
                })

        score, tier, reasons = score_prediction(result)
        tier_label = "clean_to_approve" if tier == "clean" else "hand_annotate"

        labelme_data = {
            "version": "5.0.1",
            "flags": {
                "ai_generated": True,
                "quality_score": score,
                "quality_tier": tier_label,
                "reasons": reasons,
                "model": os.path.basename(weights_path),
                "status": "needs_human_review"
            },
            "checked": False,
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(labelme_data, fp, indent=2)

        annotated_count += 1
        species_report.append({
            "species": species,
            "track": track,
            "image": os.path.basename(img_path),
            "shapes": len(shapes),
            "parts": dict(part_counts),
            "score": score,
            "tier": tier
        })

    return annotated_count, skipped_manual, skipped_existing_ai, stats, species_report


def main():
    parser = argparse.ArgumentParser(description="Auto-generate LabelMe JSON masks using trained YOLO-seg / YOLO+SAM")
    parser.add_argument("--weights", type=str, default="cluster_yolo_training/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--target", type=str, default="data/active_annotation_set", help="Target root folder")
    parser.add_argument("--track", type=str, choices=["all", "field", "inat"], default="all", help="Which track to annotate")
    parser.add_argument("--species", type=str, default=None, help="Filter to specific species folder name")
    parser.add_argument("--limit-per-species", type=int, default=None, help="Limit number of images per species for quick testing")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
    parser.add_argument("--overwrite-ai", action="store_true", help="Overwrite existing AI-generated JSONs (manual JSONs are ALWAYS kept safe)")
    parser.add_argument("--use-sam", action="store_true", help="Use MobileSAM to refine mask contours from YOLO bounding box prompts")
    parser.add_argument("--sam-weights", type=str, default="cluster_yolo_training/mobile_sam.pt", help="Path to SAM weights")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        alt = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
        if os.path.exists(alt):
            args.weights = alt
        else:
            print(f"❌ Weights not found at: {args.weights}")
            return

    sam_model = None
    if args.use_sam:
        from ultralytics import SAM
        import torch
        base_path = os.path.join(os.path.dirname(__file__), "mobile_sam.pt")
        if not os.path.exists(base_path):
            base_path = "cluster_yolo_training/mobile_sam.pt"
        if not os.path.exists(base_path):
            base_path = "mobile_sam.pt"

        sam_path = args.sam_weights if os.path.exists(args.sam_weights) else os.path.join(os.path.dirname(__file__), "weights", "mobile_sam_finetuned.pt")
        if not os.path.exists(sam_path):
            sam_path = base_path

        if os.path.exists(base_path):
            print(f"🧩 Initializing SAM from: {base_path}")
            sam_model = SAM(base_path)
            if sam_path != base_path and os.path.exists(sam_path):
                ckpt = torch.load(sam_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    sam_model.model.load_state_dict(ckpt["model_state_dict"])
                    print(f"✔ Applied fine-tuned weights from: {sam_path}")
                elif isinstance(ckpt, dict) and "mask_decoder_state_dict" in ckpt:
                    sam_model.model.mask_decoder.load_state_dict(ckpt["mask_decoder_state_dict"])
                    print(f"✔ Applied fine-tuned Mask Decoder from: {sam_path}")
        else:
            print(f"⚠️ Base SAM model not found at {base_path}, falling back to pure YOLO.")

    print("=" * 80)
    print("🦋 INSECT ANATOMY PRE-ANNOTATION PIPELINE")
    print("=" * 80)
    print(f"• Weights:   {args.weights}")
    print(f"• Refinement:{'YOLO + MobileSAM' if sam_model is not None else 'Pure YOLO'}")
    print(f"• Target:    {args.target}")
    print(f"• Track:     {args.track}")
    print(f"• Species:   {args.species or 'ALL'}")
    print(f"• Limit:     {args.limit_per_species or 'No limit'}")
    print(f"• Confidence:{args.conf}")
    print("=" * 80)

    # Load model
    model = YOLO(args.weights)

    # Collect images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    search_dirs = []
    if args.track in ["all", "field"]:
        search_dirs.append(os.path.join(args.target, "field"))
    if args.track in ["all", "inat"]:
        search_dirs.append(os.path.join(args.target, "inat"))

    by_species = defaultdict(list)
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for sp_folder in os.listdir(sdir):
            if args.species and args.species.lower() not in sp_folder.lower():
                continue
            full_sp_path = os.path.join(sdir, sp_folder)
            if not os.path.isdir(full_sp_path):
                continue
            
            imgs = []
            for ext in extensions:
                imgs.extend(glob.glob(os.path.join(full_sp_path, "**", ext), recursive=True))
            imgs = sorted(set(imgs))
            
            if args.limit_per_species:
                imgs = imgs[:args.limit_per_species]
            
            by_species[sp_folder].extend(imgs)

    all_selected_imgs = []
    for sp, flist in by_species.items():
        all_selected_imgs.extend(flist)

    print(f"📁 Selected {len(all_selected_imgs)} candidate images across {len(by_species)} species folders.\n")

    # Run inference & annotation
    annotated, skipped_man, skipped_ai, stats, species_report = annotate_images(
        model, all_selected_imgs, conf=args.conf, iou=args.iou, overwrite_ai=args.overwrite_ai, weights_path=args.weights, sam_model=sam_model
    )

    print("\n" + "=" * 80)
    print("📊 PRE-ANNOTATION SUMMARY")
    print("=" * 80)
    print(f"✔ Newly AI-annotated images:        {annotated}")
    print(f"🛡 Manual annotations preserved:     {skipped_man}")
    print(f"⏭ Existing AI annotations skipped:  {skipped_ai}")
    print("-" * 80)
    print("PARTS DETECTED BY TRACK:")
    for trk, parts in stats.items():
        print(f"  [{trk.upper()}] -> " + ", ".join(f"{k}: {v}" for k, v in sorted(parts.items())))
    print("=" * 80)

    # Show a few sample results
    if species_report:
        print("\n🔍 SAMPLE OF PROCESSED IMAGES:")
        for rep in species_report[:12]:
            parts_str = ", ".join(f"{k}: {v}" for k, v in rep["parts"].items()) or "No parts above conf"
            print(f"  • [{rep['track'].upper():<5}] {rep['species']:<25} | {rep['image']} -> {parts_str}")
        if len(species_report) > 12:
            print(f"  ... and {len(species_report) - 12} more images.")

    print("\n💡 ANYLABELING INSTRUCTIONS:")
    print("1. Open AnyLabeling.")
    print(f"2. Open your target folder: {args.target}")
    print("3. AI-generated masks have '🤖 AI (part score%)' in their description.")
    print("4. Quick inspection: adjust or delete vertices if needed, then hit Ctrl+S!")


if __name__ == "__main__":
    main()
