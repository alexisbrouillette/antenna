#!/usr/bin/env python3
"""
Renders visual skeleton overlays from dataset_pose to verify landmark alignment.
"""

import os
import glob
import cv2

BONES = [
    (0, 5, (0, 255, 255)),   # Head -> Thorax
    (5, 6, (0, 255, 255)),   # Thorax -> Abdomen
    (0, 1, (255, 255, 0)),   # Head -> Left Antenna Base
    (1, 2, (255, 255, 0)),   # Left Antenna Base -> Tip
    (0, 3, (255, 255, 0)),   # Head -> Right Antenna Base
    (3, 4, (255, 255, 0)),   # Right Antenna Base -> Tip
    (5, 7, (0, 165, 255)),   # Thorax -> Left Wing Root
    (7, 8, (0, 165, 255)),   # Left Wing Root -> Apex
    (5, 9, (0, 165, 255)),   # Thorax -> Right Wing Root
    (9, 10, (0, 165, 255)),  # Right Wing Root -> Apex
]

KPT_COLORS = [
    (0, 0, 255),     # 0: Head (Red)
    (255, 200, 0),   # 1: Ant L Base (Cyan)
    (255, 255, 0),   # 2: Ant L Tip (Cyan)
    (0, 255, 255),   # 3: Ant R Base (Yellow)
    (0, 255, 255),   # 4: Ant R Tip (Yellow)
    (0, 255, 0),     # 5: Thorax (Green)
    (255, 0, 255),   # 6: Abdomen (Magenta)
    (0, 140, 255),   # 7: Wing L Root (Orange)
    (0, 100, 255),   # 8: Wing L Apex (Deep Orange)
    (0, 140, 255),   # 9: Wing R Root (Orange)
    (0, 100, 255),   # 10: Wing R Apex (Deep Orange)
]

def main():
    val_txts = sorted(glob.glob("cluster_yolo_training/dataset_pose/labels/val/*.txt"))[:16]
    out_dir = "scratch/pose_verification"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Visualizing {len(val_txts)} validation pose samples...")

    for i, txt in enumerate(val_txts):
        stem = os.path.splitext(os.path.basename(txt))[0]
        img_path = os.path.join("cluster_yolo_training/dataset_pose/images/val", f"{stem}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(txt, "r") as f:
            tokens = f.readline().strip().split()
        if len(tokens) < 5 + 11 * 3:
            continue

        # Bbox
        cx, cy, bw, bh = [float(x) for x in tokens[1:5]]
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Keypoints
        kpts = []
        raw_k = tokens[5:]
        for k in range(11):
            kx = float(raw_k[k * 3]) * w
            ky = float(raw_k[k * 3 + 1]) * h
            kv = int(raw_k[k * 3 + 2])
            kpts.append((kx, ky, kv))

        # Bones
        for p1, p2, col in BONES:
            if kpts[p1][2] > 0 and kpts[p2][2] > 0:
                pt1 = (int(round(kpts[p1][0])), int(round(kpts[p1][1])))
                pt2 = (int(round(kpts[p2][0])), int(round(kpts[p2][1])))
                cv2.line(img, pt1, pt2, col, 3, cv2.LINE_AA)

        # Landmarks
        for k, (kx, ky, kv) in enumerate(kpts):
            if kv > 0:
                pt = (int(round(kx)), int(round(ky)))
                cv2.circle(img, pt, 6, KPT_COLORS[k], -1, cv2.LINE_AA)
                cv2.circle(img, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"sample_{i:02d}_{stem}.jpg")
        cv2.imwrite(out_path, img)

    print(f"✔ Done! Saved 16 sample skeleton overlays to: {out_dir}/")

if __name__ == "__main__":
    main()
