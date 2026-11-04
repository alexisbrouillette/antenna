#!/usr/bin/env python3
"""
Converts verified insect polygon annotations into YOLO-Pose keypoint format.

Keypoint Topology (11 landmarks):
  0: head
  1: ant_l_base
  2: ant_l_tip
  3: ant_r_base
  4: ant_r_tip
  5: thorax
  6: abdomen_tip
  7: wing_l_root
  8: wing_l_apex
  9: wing_r_root
 10: wing_r_apex

Canonical Insect Bounding Box:
  Calculated as the clean enclosing envelope of all parts (Head + Wings + Body + Legs + Antennae),
  completely resolving inconsistent human/legacy 'full-body' annotations!
"""

import os
import glob
import shutil
import cv2
import numpy as np

CLASS_NAMES = ["full-body", "wing", "leg", "antenna", "head", "body"]

def parse_yolo_polygons(txt_path, img_w, img_h):
    parts = []
    if not os.path.exists(txt_path):
        return parts

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            cls_id = int(tokens[0])
            coords = [float(x) for x in tokens[1:]]
            pts = []
            for i in range(0, len(coords) - 1, 2):
                pts.append([coords[i] * img_w, coords[i + 1] * img_h])
            parts.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "polygon": np.array(pts, dtype=np.float32)
            })
    return parts

def get_centroid(poly):
    M = cv2.moments(poly.astype(np.int32))
    if M["m00"] != 0:
        return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    return np.mean(poly, axis=0)

def extract_insect_pose(parts, img_w, img_h):
    """
    Extracts canonical insect bounding box and 11 landmarks.
    Returns: (bbox_norm, keypoints_list) or None
    """
    if not parts:
        return None

    # 1. Clean Bounding Box: Enclosing all anatomical parts
    all_pts = []
    for p in parts:
        all_pts.extend(p["polygon"])
    all_pts = np.array(all_pts, dtype=np.float32)

    min_x, min_y = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
    max_x, max_y = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

    # Slight padding (2%) for natural bbox
    pad_x = (max_x - min_x) * 0.02
    pad_y = (max_y - min_y) * 0.02
    bx1 = max(0.0, min_x - pad_x)
    by1 = max(0.0, min_y - pad_y)
    bx2 = min(float(img_w), max_x + pad_x)
    by2 = min(float(img_h), max_y + pad_y)

    bw = (bx2 - bx1) / img_w
    bh = (by2 - by1) / img_h
    bcx = (bx1 + bx2) / (2.0 * img_w)
    bcy = (by1 + by2) / (2.0 * img_h)

    # 2. Keypoints initialization: 11 points [x, y, v]
    # v: 0=not labeled/absent, 2=visible
    kpts = [[0.0, 0.0, 0] for _ in range(11)]

    heads = [p for p in parts if p["class_name"] == "head"]
    bodies = [p for p in parts if p["class_name"] == "body"]
    antennae = [p for p in parts if p["class_name"] == "antenna"]
    wings = [p for p in parts if p["class_name"] == "wing"]
    full_bodies = [p for p in parts if p["class_name"] == "full-body"]

    # Locate Head & Body
    head_c = None
    if heads:
        head_c = get_centroid(heads[0]["polygon"])
        kpts[0] = [head_c[0] / img_w, head_c[1] / img_h, 2]

    body_c = None
    if bodies:
        b_poly = bodies[0]["polygon"]
        body_c = get_centroid(b_poly)
        if head_c is not None:
            dists = np.linalg.norm(b_poly - head_c, axis=1)
            thorax_pt = b_poly[np.argmin(dists)]
            abdom_pt = b_poly[np.argmax(dists)]
        else:
            thorax_pt = b_poly[np.argmin(b_poly[:, 1])]
            abdom_pt = b_poly[np.argmax(b_poly[:, 1])]
            head_c = thorax_pt
            kpts[0] = [head_c[0] / img_w, head_c[1] / img_h, 2]

        kpts[5] = [thorax_pt[0] / img_w, thorax_pt[1] / img_h, 2]
        kpts[6] = [abdom_pt[0] / img_w, abdom_pt[1] / img_h, 2]
    elif full_bodies:
        # Fallback to full-body centroid and extrema
        fb_poly = full_bodies[0]["polygon"]
        body_c = get_centroid(fb_poly)
        if head_c is not None:
            dists = np.linalg.norm(fb_poly - head_c, axis=1)
            abdom_pt = fb_poly[np.argmax(dists)]
            thorax_pt = (head_c + body_c) / 2.0
        else:
            thorax_pt = fb_poly[np.argmin(fb_poly[:, 1])]
            abdom_pt = fb_poly[np.argmax(fb_poly[:, 1])]
            head_c = thorax_pt
            kpts[0] = [head_c[0] / img_w, head_c[1] / img_h, 2]

        kpts[5] = [thorax_pt[0] / img_w, thorax_pt[1] / img_h, 2]
        kpts[6] = [abdom_pt[0] / img_w, abdom_pt[1] / img_h, 2]
    elif head_c is not None:
        body_c = head_c

    if head_c is None:
        return None

    # Calculate insect longitudinal axis (Head -> Abdomen)
    tail_ref = np.array([kpts[6][0] * img_w, kpts[6][1] * img_h]) if kpts[6][2] > 0 else (body_c if body_c is not None else head_c)
    axis_vec = head_c - tail_ref
    norm = np.linalg.norm(axis_vec)
    if norm > 1e-3:
        u_axis = axis_vec / norm
    else:
        u_axis = np.array([0.0, -1.0])
    n_axis = np.array([-u_axis[1], u_axis[0]])  # left-pointing normal

    # Assign Antennae (0: head, 1: ant_l_base, 2: ant_l_tip, 3: ant_r_base, 4: ant_r_tip)
    for ant in antennae:
        poly = ant["polygon"]
        c = get_centroid(poly)
        side = np.dot(c - head_c, n_axis)
        idx_base, idx_tip = (1, 2) if side > 0 else (3, 4)

        dists = np.linalg.norm(poly - head_c, axis=1)
        base = poly[np.argmin(dists)]
        tip = poly[np.argmax(dists)]

        kpts[idx_base] = [base[0] / img_w, base[1] / img_h, 2]
        kpts[idx_tip] = [tip[0] / img_w, tip[1] / img_h, 2]

    # Assign Wings (5: thorax, 6: abdomen, 7: wing_l_root, 8: wing_l_apex, 9: wing_r_root, 10: wing_r_apex)
    anchor = np.array([kpts[5][0] * img_w, kpts[5][1] * img_h]) if kpts[5][2] > 0 else head_c
    for wing in wings:
        poly = wing["polygon"]
        c = get_centroid(poly)
        side = np.dot(c - anchor, n_axis)
        idx_root, idx_apex = (7, 8) if side > 0 else (9, 10)

        dists = np.linalg.norm(poly - anchor, axis=1)
        root = poly[np.argmin(dists)]
        apex = poly[np.argmax(dists)]

        current_dist = np.linalg.norm(np.array([kpts[idx_apex][0]*img_w, kpts[idx_apex][1]*img_h]) - anchor) if kpts[idx_apex][2] > 0 else 0
        cand_dist = np.linalg.norm(apex - anchor)

        if kpts[idx_apex][2] == 0 or cand_dist > current_dist:
            kpts[idx_root] = [root[0] / img_w, root[1] / img_h, 2]
            kpts[idx_apex] = [apex[0] / img_w, apex[1] / img_h, 2]

    bbox = [bcx, bcy, bw, bh]
    return bbox, kpts


def convert_dataset(source_dataset_dir, output_pose_dir):
    os.makedirs(output_pose_dir, exist_ok=True)
    stats = {"train": 0, "val": 0, "skipped": 0}

    for split in ["train", "val"]:
        src_img_dir = os.path.join(source_dataset_dir, "images", split)
        src_lbl_dir = os.path.join(source_dataset_dir, "labels", split)

        dst_img_dir = os.path.join(output_pose_dir, "images", split)
        dst_lbl_dir = os.path.join(output_pose_dir, "labels", split)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        txt_files = sorted(glob.glob(os.path.join(src_lbl_dir, "*.txt")))
        print(f"Converting {len(txt_files)} {split} samples...")

        for txt_path in txt_files:
            stem = os.path.splitext(os.path.basename(txt_path))[0]
            img_path = os.path.join(src_img_dir, f"{stem}.jpg")
            if not os.path.exists(img_path):
                stats["skipped"] += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                stats["skipped"] += 1
                continue
            h, w = img.shape[:2]

            parts = parse_yolo_polygons(txt_path, w, h)
            res = extract_insect_pose(parts, w, h)
            if res is None:
                stats["skipped"] += 1
                continue

            bbox, kpts = res

            # Format: class_id bbox_cx bbox_cy bbox_w bbox_h (x y v)*11
            line_tokens = ["0", f"{bbox[0]:.6f}", f"{bbox[1]:.6f}", f"{bbox[2]:.6f}", f"{bbox[3]:.6f}"]
            for kp in kpts:
                line_tokens.extend([f"{kp[0]:.6f}", f"{kp[1]:.6f}", f"{int(kp[2])}"])

            dst_txt = os.path.join(dst_lbl_dir, f"{stem}.txt")
            with open(dst_txt, "w", encoding="utf-8") as fp:
                fp.write(" ".join(line_tokens) + "\n")

            # Symlink or copy image
            dst_img = os.path.join(dst_img_dir, f"{stem}.jpg")
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            stats[split] += 1

    # Write dataset_pose.yaml
    yaml_path = os.path.join(output_pose_dir, "dataset_pose.yaml")
    abs_pose_dir = os.path.abspath(output_pose_dir)
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"""path: {abs_pose_dir}
train: images/train
val: images/val

# 11 anatomical keypoints: [x, y, visibility]
kpt_shape: [11, 3]

# Flip indices for horizontal data augmentation:
# 0:head <-> 0:head
# 1:ant_l_base <-> 3:ant_r_base, 2:ant_l_tip <-> 4:ant_r_tip
# 5:thorax <-> 5:thorax, 6:abdomen <-> 6:abdomen
# 7:wing_l_root <-> 9:wing_r_root, 8:wing_l_apex <-> 10:wing_r_apex
flip_idx: [0, 3, 4, 1, 2, 5, 6, 9, 10, 7, 8]

names:
  0: insect
""")

    print("\n" + "=" * 70)
    print("🎉 POSE DATASET CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"• Training samples:   {stats['train']}")
    print(f"• Validation samples: {stats['val']}")
    print(f"• Skipped:            {stats['skipped']}")
    print(f"• YAML config:        {yaml_path}")
    print("=" * 70)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, "dataset")
    dst_dir = os.path.join(script_dir, "dataset_pose")
    convert_dataset(src_dir, dst_dir)

