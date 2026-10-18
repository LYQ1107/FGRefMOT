#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import json
import argparse
import numpy as np


def load_expression_target_ids(expr_json, frame_id):
    with open(expr_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentence = data.get("sentence", "")
    label = data.get("label", {})

    key = str(frame_id)
    if key not in label:
        raise RuntimeError(f"frame {frame_id} not found in {expr_json}")

    target_ids = [int(x) for x in label[key]]
    return sentence, target_ids


def yolo_norm_cxcywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2


def read_all_labels(label_path, img_w, img_h):
    objects = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            cls_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            cx = float(parts[2])
            cy = float(parts[3])
            bw = float(parts[4])
            bh = float(parts[5])

            xyxy = yolo_norm_cxcywh_to_xyxy(cx, cy, bw, bh, img_w, img_h)

            objects.append({
                "cls": cls_id,
                "id": obj_id,
                "xyxy": xyxy,
                "cxcywh_norm": (cx, cy, bw, bh)
            })

    return objects


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - 8)

    cv2.rectangle(img, (x, y0), (x + tw + 8, y), color, -1)
    cv2.putText(
        img,
        text,
        (x + 4, y - 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default="/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/image_02/v_1LwtoLPw2TU_c014/000064.jpg"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02/v_1LwtoLPw2TU_c014/000064.txt"
    )
    parser.add_argument(
        "--expr_json",
        type=str,
        default="/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression/v_1LwtoLPw2TU_c014/volleyball-players-who-are-wearing-black-number-6-sportswear.json"
    )
    parser.add_argument("--frame_id", type=int, default=64)
    parser.add_argument(
        "--out",
        type=str,
        default="/data1/liuyeqiang/TempMam/debug_bbox_frame64.png"
    )

    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"failed to read image: {args.image}")

    H, W = img.shape[:2]
    print(f"[Image shape] W={W}, H={H}")

    sentence, target_ids = load_expression_target_ids(args.expr_json, args.frame_id)
    print(f"[Expression] {sentence}")
    print(f"[Target IDs] {target_ids}")

    objects = read_all_labels(args.label, W, H)
    print(f"[All IDs] {[o['id'] for o in objects]}")

    vis = img.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["xyxy"]
        obj_id = obj["id"]

        if obj_id in target_ids:
            color = (0, 0, 255)
            thickness = 3
            label = f"GT ID {obj_id}"
        else:
            color = (170, 170, 170)
            thickness = 1
            label = str(obj_id)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        draw_label(vis, label, x1, y1, color)

        if obj_id in target_ids:
            print(f"[Target box] ID={obj_id}, xyxy=({x1},{y1},{x2},{y2})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()