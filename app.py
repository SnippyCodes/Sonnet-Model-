"""
====================================================================================================
  SIH 2026: SSS ACOUSTIC HAZARD DETECTOR (Gradio + ZeroGPU Space)
  Space: SnippyCodes/Training-Sonnett
  Problem Statement: PS 26057 (MoES / NIOT)
  Compliance: Pure Acoustic Sonar Architecture (YOLO11s Multi-Hazard Engine)
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

try:
    from sahi_sonar import SAHISonarEngine
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_B = BASE_DIR / "weights" / "model_b" / "best.pt"
WEIGHTS_HAZARD = BASE_DIR / "weights" / "model_hazard_5class" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_b = None
_model_hazard = None
CLASSES_B = {0: "shipwreck_wreckage", 1: "pipe_cylinder"}

# Finding 9: Cached telemetry string for clear seabed scans (avoids redundant json.dumps)
EMPTY_TELEMETRY_JSON = json.dumps({"status": "CLEAR", "detections": []}, indent=2)


def load_model_b():
    global _model_b
    if _model_b is None and WEIGHTS_B.exists():
        from ultralytics import YOLO
        _model_b = YOLO(str(WEIGHTS_B))
    return _model_b


def load_model_hazard():
    """
    Loads calibrated 5-class multi-hazard YOLO11s checkpoint.
    Finding 1: Silent fallback to Model B removed to prevent misclassifying ALDFG/nets as shipwrecks.
    """
    global _model_hazard
    if _model_hazard is None:
        if not WEIGHTS_HAZARD.exists():
            print(f"[ERROR] Multi-hazard weights not found at: {WEIGHTS_HAZARD}")
            return None
        from ultralytics import YOLO
        _model_hazard = YOLO(str(WEIGHTS_HAZARD))
    return _model_hazard


@gpu_decorator
def run_sonar_detection(input_image, model_choice, conf_threshold, enhance_acoustic=False, enable_sahi=False, enable_tta=False):
    if input_image is None:
        return None, "Please upload or select a sonar scan.", EMPTY_TELEMETRY_JSON

    img_rgb = input_image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Finding 5: Acoustic Image Enhancement applied only if user explicitly enables it (avoids covariate shift)
    if enhance_acoustic and HAS_ENHANCER:
        enhancer = SonarImageEnhancer()
        img_bgr, _ = enhancer.enhance(img_bgr, apply_despeckle=True, apply_clahe=True, apply_tvg=True, use_lee_filter=True)

    h_orig, w_orig = img_bgr.shape[:2]
    vis_img = img_bgr.copy()
    detections = []

    color_map = {
        "shipwreck": (0, 140, 255),          # Amber / Orange
        "shipwreck_wreckage": (0, 140, 255),
        "pipeline": (255, 180, 0),          # Cyan / Blue
        "pipe_cylinder": (255, 180, 0),
        "ghost_net": (0, 230, 255),          # Vibrant Gold / Yellow ALDFG highlight
        "mine_cylinder": (40, 40, 240),      # Crimson hazard warning
        "crab_pot": (50, 230, 50)            # Vibrant Emerald Lime trap hazard
    }

    t0 = time.perf_counter()
    is_legacy_model = "Model B" in model_choice

    # Finding 1 & 4: SAHI Sliced Waterfall Processing
    if enable_sahi and HAS_SAHI:
        model = load_model_b() if is_legacy_model else load_model_hazard()
        if model is None:
            err_msg = (
                "⚠️ **Error: Model weights not found.**\n\n"
                f"Missing checkpoint: `{WEIGHTS_HAZARD if not is_legacy_model else WEIGHTS_B}`\n"
                "Silent fallback has been disabled to prevent false shipwreck classifications."
            )
            return None, err_msg, EMPTY_TELEMETRY_JSON

        engine = SAHISonarEngine(model, slice_size=640, overlap_ratio=0.20)
        raw_dets, _ = engine.predict_waterfall(img_bgr, conf_threshold=conf_threshold, iou_threshold=0.45, use_tta=enable_tta)

        img_area = w_orig * h_orig
        det_idx = 0
        for d in raw_dets:
            cname = d["class"]
            conf_str = d["confidence"]
            x1, y1, x2, y2 = d["bbox"]
            bw, bh = x2 - x1, y2 - y1
            box_area = bw * bh
            if (box_area / img_area) > 0.92 or (bw > 0.96 * w_orig and bh > 0.96 * h_orig):
                continue
            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label_str = f"{cname.upper()}: {conf_str}"
            (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(th + 6, y1 - 6)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), (20, 20, 20), -1)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), color, 1)
            cv2.putText(vis_img, label_str, (x1 + 3, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            length_m = round(bh * 0.05, 2)
            width_m = round(bw * 0.05, 2)
            det_idx += 1
            detections.append({
                "id": f"HAZARD-{det_idx:03d}",
                "class": cname,
                "confidence": conf_str,
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })

    elif is_legacy_model:
        # Finding 2: Legacy Model B branch with clear notification
        model = load_model_b()
        if model is None:
            return None, "⚠️ Model B weights not found at weights/model_b/best.pt.", EMPTY_TELEMETRY_JSON
        results = model.predict(source=img_bgr, conf=conf_threshold, iou=0.45, imgsz=640, augment=enable_tta, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
        labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

        img_area = w_orig * h_orig
        det_idx = 0
        for b, s, l in zip(boxes, scores, labels):
            cid = int(l)
            cname = CLASSES_B.get(cid, f"class_{cid}")
            x1, y1, x2, y2 = [int(v) for v in b]
            bw, bh = x2 - x1, y2 - y1
            box_area = bw * bh

            if (box_area / img_area) > 0.92 or (bw > 0.96 * w_orig and bh > 0.96 * h_orig):
                continue

            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label_str = f"{cname.upper()}: {s*100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(th + 6, y1 - 6)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), (20, 20, 20), -1)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), color, 1)
            cv2.putText(vis_img, label_str, (x1 + 3, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            length_m = round(bh * 0.05, 2)
            width_m = round(bw * 0.05, 2)
            det_idx += 1
            detections.append({
                "id": f"HAZARD-{det_idx:03d}",
                "class": cname,
                "confidence": f"{s*100:.1f}%",
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })

    else:
        # Finding 1: Calibrated 5-Class Multi-Hazard Detector (Model C)
        model = load_model_hazard()
        if model is None:
            err_msg = (
                "⚠️ **Error: Multi-hazard Model C weights not found.**\n\n"
                f"Missing checkpoint: `{WEIGHTS_HAZARD}`\n"
                "Silent fallback to Model B has been disabled to prevent false shipwreck classifications. "
                "Please train or deploy Model C weights."
            )
            return None, err_msg, EMPTY_TELEMETRY_JSON

        results = model.predict(source=img_bgr, conf=conf_threshold, iou=0.45, imgsz=640, augment=enable_tta, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []
        labels = res.boxes.cls.cpu().numpy() if len(res.boxes) > 0 else []

        img_area = w_orig * h_orig
        det_idx = 0
        for b, s, l in zip(boxes, scores, labels):
            cid = int(l)
            cname = model.names.get(cid, f"hazard_{cid}") if hasattr(model, "names") else f"hazard_{cid}"
            x1, y1, x2, y2 = [int(v) for v in b]
            bw, bh = x2 - x1, y2 - y1
            box_area = bw * bh

            if (box_area / img_area) > 0.92 or (bw > 0.96 * w_orig and bh > 0.96 * h_orig):
                continue

            color = color_map.get(cname, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label_str = f"{cname.upper()}: {s*100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(th + 6, y1 - 6)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), (20, 20, 20), -1)
            cv2.rectangle(vis_img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + baseline), color, 1)
            cv2.putText(vis_img, label_str, (x1 + 3, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            length_m = round(bh * 0.05, 2)
            width_m = round(bw * 0.05, 2)
            det_idx += 1
            detections.append({
                "id": f"HAZARD-{det_idx:03d}",
                "class": cname,
                "confidence": f"{s*100:.1f}%",
                "dimensions_m": f"{length_m}m x {width_m}m",
                "bbox": [x1, y1, x2, y2]
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0

    mode_tags = []
    if enable_sahi: mode_tags.append("SAHI Sliced")
    if enable_tta: mode_tags.append("TTA")
    if enhance_acoustic: mode_tags.append("Enhanced")
    tag_str = f" [{', '.join(mode_tags)}]" if mode_tags else ""

    legacy_note = ""
    if is_legacy_model:
        legacy_note = (
            "\n\n> ⚠️ **Legacy Baseline Notice:** Model B is a 2-class historical baseline (shipwreck / pipe only). "
            "It has NO taxonomy for ALDFG marine debris (ghost nets, crab pots) or seabed mines. "
            "Select **Model C** for full 5-class detection."
        )

    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    summary_text = (
        f"**Status:** {len(detections)} acoustic hazards detected{tag_str} | "
        f"**Latency:** {elapsed_ms:.1f} ms ({fps:.1f} FPS) | "
        f"**Resolution:** {w_orig}x{h_orig} px{legacy_note}"
    )

    if not detections:
        telemetry_json = EMPTY_TELEMETRY_JSON
    else:
        telemetry_json = json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_choice,
            "detections": detections
        }, indent=2)

    return vis_rgb, summary_text, telemetry_json


with gr.Blocks(title="SIH 2026: SSS Hazard Detector") as demo:
    gr.Markdown("# 🌊 SIH 2026: Side-Scan Sonar Acoustic Hazard Detector")
    gr.Markdown("**MoES / NIOT | Autonomous Underwater Vehicle (AUV) Computer Vision Engine**")

    with gr.Row():
        with gr.Column(scale=1):
            model_radio = gr.Radio(
                choices=[
                    "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)",
                    "[Legacy Baseline] Model B: YOLO11s (Shipwreck / Pipe Only - 2 Class)"
                ],
                value="Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)",
                label="Select Active Detector"
            )
            conf_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.25, step=0.05, label="Confidence Threshold")
            enhance_check = gr.Checkbox(label="✨ Acoustic Enhancement (TVG + Lee Filter + CLAHE)", value=False)
            with gr.Row():
                sahi_check = gr.Checkbox(label="🧩 SAHI Sliced Waterfall Mode", value=False)
                tta_check = gr.Checkbox(label="🔄 Test-Time Augmentation (TTA)", value=False)
            img_input = gr.Image(type="numpy", label="Upload Sonar Waterfall Scan")
            btn_run = gr.Button("🔍 Run Sonar Inspection", variant="primary")

        with gr.Column(scale=2):
            img_output = gr.Image(label="Annotated Acoustic Scan")
            status_output = gr.Markdown()
            telemetry_output = gr.Code(label="Mission Telemetry (JSON Export)", language="json")

    sample_candidates = [
        [str(BASE_DIR / "samples" / "sample_crab_pot.jpg"), "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)", 0.25, False, False, False],
        [str(BASE_DIR / "samples" / "sample_ghost_net.jpg"), "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)", 0.25, False, False, False],
        [str(BASE_DIR / "samples" / "sample_pipeline_cylinder.jpg"), "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)", 0.25, False, False, False],
        [str(BASE_DIR / "samples" / "sample_wreckage.jpg"), "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)", 0.25, False, False, False],
        [str(BASE_DIR / "samples" / "sample_clean_seabed.jpg"), "Model C: YOLO11s Multi-Hazard & Ghost Net Detector (5-Class ALDFG)", 0.25, False, False, False],
    ]
    existing_samples = [c for c in sample_candidates if os.path.exists(c[0])]
    if existing_samples:
        gr.Examples(
            examples=existing_samples,
            inputs=[img_input, model_radio, conf_slider, enhance_check, sahi_check, tta_check],
            outputs=[img_output, status_output, telemetry_output],
            fn=run_sonar_detection,
            cache_examples=False,
            label="Demonstration Sonar Waterfall Scans (Click to Test)"
        )

    btn_run.click(
        fn=run_sonar_detection,
        inputs=[img_input, model_radio, conf_slider, enhance_check, sahi_check, tta_check],
        outputs=[img_output, status_output, telemetry_output]
    )

if __name__ == "__main__":
    demo.launch()
