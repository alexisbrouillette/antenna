#!/usr/bin/env python3
"""
Reconstructs the YOLO segmentation dataset from scratch:
- Combines all field camera trap annotations (including colleague merges).
- Adds all verified iNaturalist annotations.
- Automatically downscales iNat images (max dimension 384px) to match the low-resolution field domain and keep the package lightweight.
- Splits into balanced train (85%) and val (15%) sets.
"""

import cv2
import glob
import json
import os
import random
import shutil
from collections import Counter
from PIL import Image
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

FIELD_DIR = os.path.join(PROJECT_ROOT, "data", "antenna_field_images")
INAT_DIR = os.path.join(PROJECT_ROOT, "data", "inat_high_quality_samples")

CLASS_MAP = {
    "full-body": 0,
    "wing": 1,
    "leg": 2,
    "antenna": 3,
    "head": 4,
    "body": 5,
    "abdomen": 5,
    "torso": 5,
}
CLASS_NAMES = ["full-body", "wing", "leg", "antenna", "head", "body"]

MAX_INAT_DIMENSION = 384  # Matches upper-quartile of field camera trap crops


def clean_and_prepare_dirs():
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if os.path.exists(lbl_dir):
            shutil.rmtree(lbl_dir)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)


def collect_verified_annotations():
    """
    Collects all unique verified ground truth annotations.
    Returns: list of dicts: {'source': 'field'|'inat', 'json_path': str, 'img_path': str, 'stem': str}
    """
    samples = []
    seen_stems = set()

    # 1. Field camera trap annotations (ind_*.json)
    field_jsons = sorted(glob.glob(os.path.join(FIELD_DIR, "ind_*.json")))
    for jf in field_jsons:
        stem = os.path.splitext(os.path.basename(jf))[0]
        img = os.path.join(FIELD_DIR, stem + ".jpg")
        if os.path.exists(img) and stem not in seen_stems:
            samples.append({
                "source": "field",
                "json_path": jf,
                "img_path": img,
                "stem": stem
            })
            seen_stems.add(stem)

    # 2. iNaturalist verified annotations
    inat_jsons = sorted(glob.glob(os.path.join(INAT_DIR, "**", "*.json"), recursive=True))
    for jf in inat_jsons:
        stem = os.path.splitext(os.path.basename(jf))[0]
        img = os.path.join(os.path.dirname(jf), stem + ".jpg")
        if os.path.exists(img) and stem not in seen_stems:
            samples.append({
                "source": "inat",
                "json_path": jf,
                "img_path": img,
                "stem": stem
            })
            seen_stems.add(stem)

    return samples


def apply_field_degradation(img):
    """
    Applies synthetic camera-trap domain degradation to iNat macro images:
    - Downscaled to max 384px (matching typical camera trap crop resolution)
    - Defocus or motion blur (simulating imperfect trap focus / insect wing flutter)
    - Sensor ISO noise & grain (simulating low-light / night camera sensor noise)
    - Flash & contrast shifts (simulating uneven trap LED illumination)
    - JPEG compression artifacts (Q=65-75)
    
    CRUCIAL: These are purely pixel-intensity transformations;
    normalized polygon coordinates (nx, ny in [0, 1]) remain 100% aligned.
    """
    h, w = img.shape[:2]

    # 1. Downscale to field resolution
    if max(w, h) > MAX_INAT_DIMENSION:
        scale = MAX_INAT_DIMENSION / max(w, h)
        new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Defocus or subtle motion blur (50% chance)
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 3. Sensor ISO grain / low-light noise (60% chance)
    if random.random() < 0.6:
        sigma = random.uniform(5.0, 13.0)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Exposure & contrast shift (camera trap LED / flash lighting)
    gamma = random.uniform(0.8, 1.25)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 5. Mild contrast variation
    alpha = random.uniform(0.85, 1.15)
    img = np.clip(alpha * img.astype(np.float32) + random.uniform(-8, 8), 0, 255).astype(np.uint8)

    return img


def process_sample(sample, split):
    """
    Converts LabelMe JSON polygons to YOLO normalized coordinates.
    Downscales & degrades iNat images to field resolution (max dimension 384px).
    """
    json_path = sample["json_path"]
    img_path = sample["img_path"]
    stem = sample["stem"]
    is_inat = sample["source"] == "inat"

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not width or not height:
        with Image.open(img_path) as im:
            width, height = im.size

    yolo_lines = []
    class_counts = Counter()

    for shape in data.get("shapes", []):
        lbl = shape.get("label")
        if lbl not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[lbl]
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue

        norm_pts = []
        for x, y in pts:
            nx = max(0.0, min(1.0, x / width))
            ny = max(0.0, min(1.0, y / height))
            norm_pts.append(f"{nx:.6f} {ny:.6f}")

        yolo_lines.append(f"{cls_id} " + " ".join(norm_pts))
        class_counts[CLASS_NAMES[cls_id]] += 1

    if not yolo_lines:
        return False, class_counts

    dst_img = os.path.join(DATASET_DIR, "images", split, stem + ".jpg")
    dst_lbl = os.path.join(DATASET_DIR, "labels", split, stem + ".txt")

    # Image handling: degrade iNat to field mimic, keep field as-is
    if is_inat:
        im = cv2.imread(img_path)
        if im is not None:
            im_degraded = apply_field_degradation(im)
            quality = random.randint(68, 78)
            cv2.imwrite(dst_img, im_degraded, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            shutil.copy2(img_path, dst_img)
    else:
        shutil.copy2(img_path, dst_img)

    with open(dst_lbl, "w", encoding="utf-8") as fp:
        fp.write("\n".join(yolo_lines) + "\n")

    return True, class_counts


def write_dataset_yaml():
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    content = """# YOLOv8 / YOLO11 Segmentation Dataset Configuration
# Relative paths so this works anywhere you copy the folder
path: ./dataset
train: images/train
val: images/val

names:
  0: full-body
  1: wing
  2: leg
  3: antenna
  4: head
  5: body
"""
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(content)


def main():
    print("=" * 75)
    print("🏗️ RECONSTRUCTING INSECT ANATOMY YOLO-SEG DATASET")
    print("=" * 75)

    clean_and_prepare_dirs()
    samples = collect_verified_annotations()

    field_samples = [s for s in samples if s["source"] == "field"]
    inat_samples = [s for s in samples if s["source"] == "inat"]

    print(f"📦 Total verified annotations found: {len(samples)}")
    print(f"   • Field camera trap samples: {len(field_samples)} (includes colleague merge)")
    print(f"   • iNaturalist macro samples:  {len(inat_samples)} (downscaled to <= {MAX_INAT_DIMENSION}px)")

    # Field-prioritized split:
    # 20% of field images in val (~53 images) so validation is heavily field-focused
    # 10% of iNat images in val (~10 images) so val remains mixed
    # Remaining 80% field + 90% iNat in train split
    random.seed(42)
    random.shuffle(field_samples)
    random.shuffle(inat_samples)

    val_field_cnt = max(1, int(len(field_samples) * 0.20))
    val_inat_cnt = max(1, int(len(inat_samples) * 0.10))

    val_set = set(s["stem"] for s in (field_samples[:val_field_cnt] + inat_samples[:val_inat_cnt]))

    split_counts = {"train": 0, "val": 0}
    split_classes = {"train": Counter(), "val": Counter()}
    split_sources = {"train": Counter(), "val": Counter()}

    for s in samples:
        split = "val" if s["stem"] in val_set else "train"
        ok, c_counts = process_sample(s, split)
        if ok:
            split_counts[split] += 1
            split_sources[split][s["source"]] += 1
            split_classes[split].update(c_counts)

    write_dataset_yaml()

    # Calculate dataset directory size
    total_bytes = 0
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    size_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 75)
    print("📊 DATASET RECONSTRUCTION SUMMARY")
    print("=" * 75)
    print(f"• Total Images:     {split_counts['train'] + split_counts['val']}")
    print(f"  - Train split:    {split_counts['train']} ({split_sources['train']['field']} field, {split_sources['train']['inat']} iNat)")
    print(f"  - Val split:      {split_counts['val']} ({split_sources['val']['field']} field, {split_sources['val']['inat']} iNat)")
    print(f"• Total Disk Size:  {size_mb:.2f} MB (ultra-lightweight & cluster-ready!)")
    print("-" * 75)
    print("ANATOMICAL INSTANCE COUNTS:")
    all_classes = split_classes["train"] + split_classes["val"]
    for cname in CLASS_NAMES:
        t_cnt = split_classes["train"][cname]
        v_cnt = split_classes["val"][cname]
        print(f"  • {cname:<12}: {all_classes[cname]:>4} total (Train: {t_cnt:>4}, Val: {v_cnt:>3})")
    print("=" * 75)
    print(f"✔ Ready to train! Config saved to: {os.path.join(DATASET_DIR, 'dataset.yaml')}\n")


if __name__ == "__main__":
    main()

