#!/usr/bin/env python3
"""
Syncs all master annotations and verified active annotations (including iNat!)
into the training dataset so the model learns from both camera trap and iNat photos.
"""

import cv2
import glob
import json
import os
import random
import shutil
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
MASTER_FIELD_DIR = os.path.join(SCRIPT_DIR, "..", "data", "antenna_field_images")
ACTIVE_SET_DIR = os.path.join(SCRIPT_DIR, "..", "data", "active_annotation_set")

CLASS_MAP = {
    "full-body": 0,
    "wing": 1,
    "leg": 2,
    "antenna": 3,
    "head": 4,
    "body": 5,
    "abdomen": 5,
    "torso": 5
}

def degrade_image_to_field_quality(img_cv):
    """
    Synthetically degrades a high-resolution iNat macro photo to mimic
    the lower resolution, sensor blur, and compression of camera trap crops.
    """
    h, w = img_cv.shape[:2]
    # Downscale to typical field crop dimension (~180-300px)
    target_dim = random.randint(180, 300)
    scale = target_dim / max(h, w)
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    down = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Optional slight blur
    if random.random() > 0.5:
        down = cv2.GaussianBlur(down, (3, 3), 0.5)

    # Upscale back to preserve original coordinate dimensions
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    # Simulate camera trap JPEG compression
    quality = random.randint(45, 65)
    _, enc = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def add_annotation_to_dataset(json_path, img_path, split="train", generate_field_mimic=False):
    base = os.path.splitext(os.path.basename(json_path))[0]
    
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not width or not height:
        with Image.open(img_path) as im:
            width, height = im.size

    yolo_lines = []
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

    if not yolo_lines:
        return False

    # 1. Standard copy
    dst_img = os.path.join(DATASET_DIR, "images", split, base + ".jpg")
    shutil.copy2(img_path, dst_img)

    dst_lbl = os.path.join(DATASET_DIR, "labels", split, base + ".txt")
    with open(dst_lbl, "w", encoding="utf-8") as fp:
        fp.write("\n".join(yolo_lines) + "\n")

    # 2. If iNat image: generate synthetic "field mimic" (downscaled + compressed)
    if generate_field_mimic:
        mimic_base = base + "_field_mimic"
        img_cv = cv2.imread(img_path)
        if img_cv is not None:
            degraded_cv = degrade_image_to_field_quality(img_cv)
            dst_mimic_img = os.path.join(DATASET_DIR, "images", split, mimic_base + ".jpg")
            cv2.imwrite(dst_mimic_img, degraded_cv)

            dst_mimic_lbl = os.path.join(DATASET_DIR, "labels", split, mimic_base + ".txt")
            with open(dst_mimic_lbl, "w", encoding="utf-8") as fp:
                fp.write("\n".join(yolo_lines) + "\n")

    return True


def sync_all_to_training():
    print("🔄 Updating training dataset with verified annotations & field degradation...")
    
    # Check existing dataset files
    existing_train_imgs = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(DATASET_DIR, "images", "train", "*"))}
    existing_val_imgs = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(DATASET_DIR, "images", "val", "*"))}

    new_added = 0
    mimic_added = 0

    # Check active verified annotations (including iNat annotations marked as checked=True)
    if os.path.exists(ACTIVE_SET_DIR):
        active_jsons = glob.glob(os.path.join(ACTIVE_SET_DIR, "**", "*.json"), recursive=True)
        for jf in active_jsons:
            stem = os.path.splitext(os.path.basename(jf))[0]
            is_inat = "inat" in jf
            mimic_stem = stem + "_field_mimic"

            # Check if needs adding or field mimic needs adding
            needs_base = stem not in existing_train_imgs and stem not in existing_val_imgs
            needs_mimic = is_inat and (mimic_stem not in existing_train_imgs)

            if not needs_base and not needs_mimic:
                continue

            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                if d.get("checked") is True or d.get("flags", {}).get("status") == "verified":
                    img_path = os.path.splitext(jf)[0] + ".jpg"
                    if os.path.exists(img_path):
                        if add_annotation_to_dataset(jf, img_path, split="train", generate_field_mimic=is_inat):
                            if needs_base:
                                new_added += 1
                            if needs_mimic:
                                mimic_added += 1
            except Exception as e:
                pass

    total_train = len(glob.glob(os.path.join(DATASET_DIR, "images", "train", "*")))
    total_val = len(glob.glob(os.path.join(DATASET_DIR, "images", "val", "*")))
    print(f"\n✔ Dataset sync complete!")
    print(f"• Added {new_added} new base samples and {mimic_added} synthetic field-degraded samples.")
    print(f"• Total Train images: {total_train}")
    print(f"• Total Val images:   {total_val}")


if __name__ == "__main__":
    sync_all_to_training()

