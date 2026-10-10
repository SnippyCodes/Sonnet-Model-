"""
Quickstart Inference Script for Marine Hazard & Ghost Net YOLO Models
Runs inference on sample sonar scans and displays predictions.
Usage:
    py test_inference.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_C = BASE_DIR / "weights" / "model_hazard_5class" / "best.pt"
WEIGHTS_B = BASE_DIR / "weights" / "model_b" / "best.pt"
SAMPLES_DIR = BASE_DIR / "samples"

def main():
    print("=" * 80)
    print("SIH 2026: ACOUSTIC MARINE HAZARD & GHOST NET YOLO INFERENCE QUICKSTART")
    print("=" * 80)

    # 1. Test Model C (Multi-Hazard & Ghost Net YOLO11s)
    if WEIGHTS_C.exists():
        print(f"\n[TESTING MODEL C: 5-Class Multi-Hazard YOLO11s]")
        print(f"Weights: {WEIGHTS_C}")
        model_c = YOLO(str(WEIGHTS_C))

        for sample_img in SAMPLES_DIR.glob("*.*"):
            print(f"\nScanning: {sample_img.name} ...")
            results = model_c.predict(source=str(sample_img), conf=0.25, iou=0.45, imgsz=640, verbose=False)
            boxes = results[0].boxes
            if len(boxes) == 0:
                print("  -> No hazards above threshold (Clean Seafloor).")
            else:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    xyxy = [int(v) for v in b.xyxy[0]]
                    cname = model_c.names.get(cls_id, f"Class {cls_id}")
                    print(f"  [DETECTED] {cname.upper()} ({conf*100:.1f}%) at BBox: {xyxy}")
    else:
        print(f"[WARNING] Model C weights not found at {WEIGHTS_C}")

    # 2. Test Model B (Pure Sonar Shipwreck YOLO11s)
    if WEIGHTS_B.exists():
        print(f"\n[TESTING MODEL B: Pure Sonar YOLO11s (87.9% Shipwreck mAP50)]")
        print(f"Weights: {WEIGHTS_B}")
        model_b = YOLO(str(WEIGHTS_B))
        wreck_img = SAMPLES_DIR / "sample_wreckage.jpg"
        if wreck_img.exists():
            print(f"Scanning: {wreck_img.name} ...")
            res_b = model_b.predict(source=str(wreck_img), conf=0.25, iou=0.45, imgsz=640, verbose=False)
            for b in res_b[0].boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = [int(v) for v in b.xyxy[0]]
                cname = model_b.names.get(cls_id, f"Class {cls_id}")
                print(f"  [DETECTED] {cname.upper()} ({conf*100:.1f}%) at BBox: {xyxy}")

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("To launch the full interactive web UI, run: py app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
