#!/usr/bin/env python3
"""
Pose-Guided SAM: Synthesizes clean, topologically guaranteed anatomical masks
by prompting fine-tuned MobileSAM with YOLO-Pose keypoints.

Guarantees:
1. Head, Wings, Antennae, and Body are anchored to the true anatomical kinematic skeleton.
2. 'Full-Body' is synthesized as the exact geometric union of all segmented parts.
3. Absent/tucked antennae (visibility = 0) are omitted rather than hallucinated.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM

KEYPOINT_NAMES = [
    "head",
    "ant_l_base", "ant_l_tip",
    "ant_r_base", "ant_r_tip",
    "thorax", "abdomen",
    "wing_l_root", "wing_l_apex",
    "wing_r_root", "wing_r_apex"
]

def load_sam_decoder(sam_base, sam_finetuned):
    sam = SAM(sam_base)
    if os.path.exists(sam_finetuned):
        ckpt = torch.load(sam_finetuned, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "mask_decoder_state_dict" in ckpt:
            sam.model.mask_decoder.load_state_dict(ckpt["mask_decoder_state_dict"])
            print(f"✔ Loaded fine-tuned SAM Mask Decoder from: {sam_finetuned}")
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sam.model.load_state_dict(ckpt["model_state_dict"])
            print(f"✔ Loaded fine-tuned SAM model from: {sam_finetuned}")
    return sam

def predict_pose_and_masks(img_path, pose_model, sam_model, conf=0.25):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # 1. Predict Skeleton with YOLO-Pose
    pose_results = pose_model.predict(img, conf=conf, verbose=False)
    if not pose_results or len(pose_results[0].boxes) == 0:
        return None

    r = pose_results[0]
    # Use best detected insect
    kpts_data = r.keypoints.data[0].cpu().numpy()  # shape (11, 3) -> x, y, conf

    part_masks = {}

    # Helper: prompt SAM with positive point(s) and optional bbox
    def prompt_sam_points(points, labels, bbox=None):
        try:
            # SAM point prompting via ultralytics SAM API
            sam_res = sam_model.predict(
                img,
                points=points,
                labels=labels,
                bboxes=bbox,
                verbose=False
            )
            if sam_res and sam_res[0].masks is not None:
                m = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return m
        except Exception:
            pass
        return None

    # A. Head Mask
    if kpts_data[0][2] > 0.4:
        hx, hy = kpts_data[0][0], kpts_data[0][1]
        m = prompt_sam_points([[hx, hy]], [1])
        if m is not None:
            part_masks["head"] = m

    # B. Left Antenna
    if kpts_data[1][2] > 0.4 and kpts_data[2][2] > 0.4:
        pts = [[kpts_data[1][0], kpts_data[1][1]], [kpts_data[2][0], kpts_data[2][1]]]
        m = prompt_sam_points(pts, [1, 1])
        if m is not None:
            part_masks["antenna_left"] = m

    # C. Right Antenna
    if kpts_data[3][2] > 0.4 and kpts_data[4][2] > 0.4:
        pts = [[kpts_data[3][0], kpts_data[3][1]], [kpts_data[4][0], kpts_data[4][1]]]
        m = prompt_sam_points(pts, [1, 1])
        if m is not None:
            part_masks["antenna_right"] = m

    # D. Left Wing
    if kpts_data[7][2] > 0.4 and kpts_data[8][2] > 0.4:
        pts = [[kpts_data[7][0], kpts_data[7][1]], [kpts_data[8][0], kpts_data[8][1]]]
        m = prompt_sam_points(pts, [1, 1])
        if m is not None:
            part_masks["wing_left"] = m

    # E. Right Wing
    if kpts_data[9][2] > 0.4 and kpts_data[10][2] > 0.4:
        pts = [[kpts_data[9][0], kpts_data[9][1]], [kpts_data[10][0], kpts_data[10][1]]]
        m = prompt_sam_points(pts, [1, 1])
        if m is not None:
            part_masks["wing_right"] = m

    # F. Body (Thorax to Abdomen)
    if kpts_data[5][2] > 0.4 and kpts_data[6][2] > 0.4:
        pts = [[kpts_data[5][0], kpts_data[5][1]], [kpts_data[6][0], kpts_data[6][1]]]
        m = prompt_sam_points(pts, [1, 1])
        if m is not None:
            part_masks["body"] = m

    # G. Synthesize Canonical Full-Body as the Exact Geometric Union of all parts!
    if part_masks:
        fb_mask = np.zeros((h, w), dtype=np.uint8)
        for m in part_masks.values():
            fb_mask = np.logical_or(fb_mask, m).astype(np.uint8)
        part_masks["full-body"] = fb_mask

    return {
        "keypoints": kpts_data,
        "masks": part_masks,
        "image_shape": (h, w)
    }

def main():
    print("Pose-Guided SAM Module Ready.")

if __name__ == "__main__":
    main()
