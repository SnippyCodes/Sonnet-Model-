"""
====================================================================================================
  SIH 2026: SAHI (SLICING AIDED HYPER INFERENCE) SONAR WATERFALL ENGINE
  Solves Small-Object Acoustic Hazard Detection in Massive Continuous Sonar Waterfalls
  Supports:
  1. High-Resolution Waterfall Slicing (640x640 with configurable overlap)
  2. Batched YOLO11s Multi-Slice Inference
  3. Global Coordinate Mapping & NMS Merging
  4. Test-Time Augmentation (TTA) Toggle (+1.5-3% boost on faint targets)
====================================================================================================
"""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def apply_global_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Performs Non-Maximum Suppression across all sliced window detections."""
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d["conf_val"], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        filtered = []
        for d in detections:
            # Only compare if same class
            if d["class"] == best["class"]:
                iou = compute_iou(np.array(best["bbox"]), np.array(d["bbox"]))
                if iou < iou_threshold:
                    filtered.append(d)
            else:
                filtered.append(d)
        detections = filtered

    return kept


class SAHISonarEngine:
    """Slicing Aided Hyper Inference Engine for Side-Scan Sonar Waterfall Scans."""

    def __init__(self, model_or_path, slice_size: int = 640, overlap_ratio: float = 0.20):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(slice_size * (1.0 - overlap_ratio))

        if isinstance(model_or_path, (str, Path)):
            self.model = YOLO(str(model_or_path))
        else:
            self.model = model_or_path

    def get_slice_coordinates(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Calculates sliding window coordinates (x1, y1, x2, y2) to cover the entire waterfall."""
        slices = []
        y = 0
        while y < img_h:
            y_end = min(y + self.slice_size, img_h)
            y_start = max(0, y_end - self.slice_size)
            x = 0
            while x < img_w:
                x_end = min(x + self.slice_size, img_w)
                x_start = max(0, x_end - self.slice_size)
                slices.append((x_start, y_start, x_end, y_end))
                if x_end >= img_w:
                    break
                x += self.stride
            if y_end >= img_h:
                break
            y += self.stride
        return slices

    def predict_waterfall(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_tta: bool = False
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs sliced inference across high-resolution sonar waterfall scan.
        Returns:
        - list of merged hazard detections with global coordinates
        - elapsed latency in milliseconds
        """
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]

        # If image is already smaller or equal to slice size, run single pass
        if h <= self.slice_size and w <= self.slice_size:
            results = self.model.predict(
                source=img_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=self.slice_size,
                augment=use_tta,
                verbose=False
            )
            raw_detections = []
            res = results[0]
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
            labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

            for b, s, l in zip(boxes, scores, labels):
                cid = int(l)
                cname = self.model.names.get(cid, f"class_{cid}")
                raw_detections.append({
                    "class": cname,
                    "confidence": f"{s*100:.1f}%",
                    "conf_val": float(s),
                    "bbox": [int(v) for v in b]
                })
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return raw_detections, elapsed_ms

        slice_coords = self.get_slice_coordinates(w, h)
        slice_images = []
        for (x1, y1, x2, y2) in slice_coords:
            crop = img_bgr[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch != self.slice_size or cw != self.slice_size:
                # Pad boundary crops to 640x640 with zero-padding to prevent aspect-ratio stretching
                padded = np.zeros((self.slice_size, self.slice_size, 3), dtype=np.uint8)
                padded[:ch, :cw] = crop
                slice_images.append(padded)
            else:
                slice_images.append(crop)

        # Batched prediction on all slices
        batch_results = self.model.predict(
            source=slice_images,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.slice_size,
            augment=use_tta,
            verbose=False
        )

        raw_detections = []
        for idx, (res, (ox, oy, ex, ey)) in enumerate(zip(batch_results, slice_coords)):
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
            labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

            for b, s, l in zip(boxes, scores, labels):
                cid = int(l)
                cname = self.model.names.get(cid, f"class_{cid}")
                # Translate back to global waterfall coordinates and clamp to image boundary
                gx1 = max(0, min(w, int(b[0] + ox)))
                gy1 = max(0, min(h, int(b[1] + oy)))
                gx2 = max(0, min(w, int(b[2] + ox)))
                gy2 = max(0, min(h, int(b[3] + oy)))

                raw_detections.append({
                    "class": cname,
                    "confidence": f"{s*100:.1f}%",
                    "conf_val": float(s),
                    "bbox": [gx1, gy1, gx2, gy2]
                })
                    "bbox": [gx1, gy1, gx2, gy2]
                })

        # Merge overlapping detections via Global NMS
        merged_detections = apply_global_nms(raw_detections, iou_threshold=iou_threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return merged_detections, elapsed_ms
