"""
====================================================================================================
  SIH 2026: SSS ACOUSTIC HAZARD DETECTOR (Gradio + ZeroGPU Space)
  Space: SnippyCodes/Training-Sonnett
  Problem Statement: PS 26057 (MoES / NIOT)
  Compliance: Agent.md & PS.md Pure Acoustic Sonar Architecture
====================================================================================================
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import gradio as gr
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

def gpu_decorator(fn):
    if HAS_ZEROGPU:
        return spaces.GPU(fn)
    return fn

try:
    from sonar_enhancer import SonarImageEnhancer
    HAS_ENHANCER = True
except ImportError:
    HAS_ENHANCER = False

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_A = BASE_DIR / "weights" / "model_a" / "best.pth"
WEIGHTS_B = BASE_DIR / "weights" / "model_b" / "best.pt"

WEIGHTS_HAZARD = BASE_DIR / "weights" / "model_hazard_5class" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_a = None
_model_b = None
_model_hazard = None
CLASSES_A = {0: "__background__", 1: "shipwreck_wreckage", 2: "pipe_cylinder"}
CLASSES_B = {0: "shipwreck_wreckage", 1: "pipe_cylinder"}

def load_model_a():
    global _model_a
    if _model_a is None and WEIGHTS_A.exists():
        ckpt = torch.load(WEIGHTS_A, map_location=DEVICE)
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        _model_a = model
    return _model_a

def load_model_b():
    global _model_b
    if _model_b is None and WEIGHTS_B.exists():
        from ultralytics import YOLO
        _model_b = YOLO(str(WEIGHTS_B))
    return _model_b

def load_model_hazard():
    global _model_hazard
    target = WEIGHTS_HAZARD if WEIGHTS_HAZARD.exists() else WEIGHTS_B
    if _model_hazard is None and target.exists():
        from ultralytics import YOLO
        _model_hazard = YOLO(str(target))
    return _model_hazard

@gpu_decorator
def run_sonar_detection(input_image, model_choice, conf_threshold, enhance_acoustic=False):
    if input_image is None:
        return None, "Please upload or select a sonar scan.", "{}"

    img_rgb = input_image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Acoustic Image Enhancement (CLAHE + Despeckle + TVG) if requested
    if enhance_acoustic and HAS_ENHANCER:
        enhancer = SonarImageEnhancer()
        img_bgr, _ = enhancer.enhance(img_bgr)

    h_orig, w_orig = img_bgr.shape[:2]
    vis_img = img_bgr.copy()
    detections = []

    color_map = {
        "shipwreck": (0, 183, 255),
        "shipwreck_wreckage": (0, 183, 255),
        "pipeline": (237, 160, 57),
        "pipe_cylinder": (237, 160, 57),
        "ghost_net": (255, 220, 0),        # Cyan / bright ALDFG net highlight
        "mine_cylinder": (50, 50, 240),     # Crimson warning
        "crab_pot": (220, 90, 150)          # Violet trap hazard
    }

    t0 = time.perf_counter()

    if "Model C" in model_choice or "Ghost Net" in model_choice:
        model = load_model_hazard()
        if model is None:
            return None, "Multi-hazard weights not found. Please train with train_multiclass_hazard_yolo.py.", "{}"
        results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
        labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            cid = int(l)
            cname = model.names.get(cid, f"class_{cid}") if hasattr(model, "names") else f"hazard_{cid}"
            x1, y1, x2, y2 = [int(v) for v in b]
            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{cname}: {s*100:.1f}%", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            length_m = round((y2 - y1) * 0.05, 2)
            width_m = round((x2 - x1) * 0.05, 2)
            detections.append({
                "id": f"HAZARD-{i+1:03d}",
                "class": cname,
                "confidence": f"{s*100:.1f}%",
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })
    elif "Model B" in model_choice:
        model = load_model_b()
        if model is None:
            return None, "Model B weights not found. Please ensure weights/model_b/best.pt exists.", "{}"
        results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
        labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            cid = int(l)
            cname = CLASSES_B.get(cid, f"class_{cid}")
            x1, y1, x2, y2 = [int(v) for v in b]
            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{cname}: {s*100:.1f}%", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            length_m = round((y2 - y1) * 0.05, 2)
            width_m = round((x2 - x1) * 0.05, 2)
            detections.append({
                "id": f"HAZARD-{i+1:03d}",
                "class": cname,
                "confidence": f"{s*100:.1f}%",
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })
    else:
        model = load_model_a()
        if model is None:
            return None, "Model A weights not found. Please ensure weights/model_a/best.pth exists.", "{}"
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.to(DEVICE)
        with torch.no_grad():
            preds = model([img_t])[0]
        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            if l == 0: continue
            cname = CLASSES_A.get(int(l), f"class_{l}")
            x1, y1, x2, y2 = [int(v) for v in b]
            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{cname}: {s*100:.1f}%", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            length_m = round((y2 - y1) * 0.05, 2)
            width_m = round((x2 - x1) * 0.05, 2)
            detections.append({
                "id": f"HAZARD-{i+1:03d}",
                "class": cname,
                "confidence": f"{s*100:.1f}%",
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0

    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    summary_text = f"**Status:** {len(detections)} acoustic hazards detected | **Latency:** {elapsed_ms:.1f} ms ({fps:.1f} FPS) | **Resolution:** {w_orig}x{h_orig} px"
    telemetry_json = json.dumps({"timestamp": datetime.datetime.now().isoformat(), "model": model_choice, "detections": detections}, indent=2)

    return vis_rgb, summary_text, telemetry_json

with gr.Blocks(title="SIH 2026: SSS Hazard Detector") as demo:
    gr.Markdown("# 🌊 SIH 2026: Side-Scan Sonar Acoustic Hazard Detector")
    gr.Markdown("**MoES / NIOT | Autonomous Underwater Vehicle (AUV) Computer Vision Engine**")

    with gr.Row():
        with gr.Column(scale=1):
            model_radio = gr.Radio(
                choices=[
                    "Model A: Faster R-CNN (MobileNetV3 Edge - 72.9% Recall)",
                    "Model B: YOLO11s Pure Sonar (87.9% Shipwreck mAP50)",
                    "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)"
                ],
                value="Model B: YOLO11s Pure Sonar (87.9% Shipwreck mAP50)",
                label="Select Active Detector"
            )
            conf_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.35, step=0.05, label="Confidence Threshold")
            enhance_check = gr.Checkbox(label="✨ Acoustic Image Enhancement Mode (CLAHE + Despeckle)", value=False)
            img_input = gr.Image(type="numpy", label="Upload Sonar Waterfall Scan")
            btn_run = gr.Button("🔍 Run Sonar Inspection", variant="primary")

        with gr.Column(scale=2):
            img_output = gr.Image(label="Annotated Acoustic Scan")
            status_output = gr.Markdown()
            telemetry_output = gr.Code(label="Mission Telemetry (JSON Export)", language="json")

    btn_run.click(
        fn=run_sonar_detection,
        inputs=[img_input, model_radio, conf_slider, enhance_check],
        outputs=[img_output, status_output, telemetry_output]
    )

if __name__ == "__main__":
    demo.launch()
