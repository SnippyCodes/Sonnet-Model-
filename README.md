# 🌊 Team Synora: Side-Scan Sonar (SSS) Marine Hazard & Ghost Net Detector

**Smart India Hackathon (SIH 2026) | Problem Statement: PS 26057**  
**Organization:** Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT)  
**Theme:** Disaster Management & Marine Biodiversity Protection  

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11s-00FFFF.svg)](https://docs.ultralytics.com/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%205050-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Framework](https://img.shields.io/badge/UI-Gradio%20Dashboard-orange.svg)](https://gradio.app/)

---

## 📌 Executive Summary

This repository hosts edge-optimized, real-time deep learning models engineered to detect underwater anthropogenic marine debris, navigation hazards, and abandoned fishing gear from acoustic **Side-Scan Sonar (SSS)** imagery. 

Unlike optical RGB cameras that fail in murky or deep oceanic environments, these models are trained directly on acoustic backscatter and shadow geometry to support Autonomous Underwater Vehicles (AUVs) and ship-towed sonar platforms.

---

## 🚀 Available YOLO Models

Both model weights are bundled directly within this repository in the `weights/` directory:

| Model ID | Weight Path | Classes | Primary Focus | Validated Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Model B** | `weights/model_b/best.pt` | 2 (`shipwreck_wreckage`, `pipe_cylinder`) | High-precision sunken vessel & hull detection | **87.9% mAP50** (92.2% Recall) |
| **Model C** | `weights/model_hazard_5class/best.pt` | 5 (`pipeline`, `shipwreck`, `mine_cylinder`, `ghost_net`, `crab_pot`) | Full marine debris & ghost fishing net detection | **96.6% Ghost Net mAP50** (64.4% Overall Precision, 3.9ms latency) |

---

## 🧠 Model Architectures & Training Methodology

### 1. Model B: Pure Sonar YOLO11s Detector
* **Architecture:** Ultralytics YOLO11s (9.42M parameters, 21.4 GFLOPs)
* **Dataset:** Curated pure acoustic sonar dataset (786 training images including 88 real seafloor negative scans).
* **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM) via PyTorch 2.14 + CUDA 13.0.
* **Physics-Aware Acoustic Augmentations:**
  - **Grayscale Invariance:** Color hue and saturation jittering disabled (`hsv_h=0.0`, `hsv_s=0.0`) to avoid corrupting single-channel acoustic backscatter data.
  - **Acoustic Gain Variation:** `hsv_v=0.2` to simulate acoustic transmission loss and water column attenuation.
  - **Towfish Heading Inversion:** `flipud=0.5` (simulates reverse survey lines).
  - **Port / Starboard Symmetry:** `fliplr=0.5` (transducer channel invariance).
* **Key Results:**
  - **Shipwreck mAP50:** `87.9%` (0.879)
  - **Shipwreck Recall:** `92.2%`
  - **Overall Precision:** `73.8%`
  - **Inference Latency:** `4.0 ms` (~250 FPS on RTX 5050)

---

### 2. Model C: Multi-Hazard & Ghost Net YOLO11s Detector (30 Epochs)
* **Architecture:** Ultralytics YOLO11s with 5-class detection head (9.41M parameters, 21.4 GFLOPs).
* **Dataset:** `production_balanced_v1` extracted from `archive.zip` (14,007 acoustic sonar images: 9,356 train, 2,219 validation, 2,432 test).
* **Hardware:** Local NVIDIA GeForce RTX 5050 Laptop GPU (1.85 hours total training time).
* **Target Classes (Mandated by PS 26057):**
  1. `pipeline`: Subsea pipelines and cylindrical conduits.
  2. `shipwreck`: Sunken hulls, wreckage clusters, and aircraft fuselages.
  3. `mine_cylinder`: Cylindrical ordnance (UXO) and acoustic metallic anomalies.
  4. `ghost_net`: Abandoned, Lost, or Discarded Fishing Gear (ALDFG) presenting critical thruster entanglement risks.
  5. `crab_pot`: Benthic cage traps and tether snag hazards.
* **Final Validated Metrics (2,219 Validation Images, 2,364 Ground-Truth Instances):**
  - **`ghost_net`:** **96.8% Precision** | **92.5% Recall** | **96.6% mAP50** | **79.5% mAP50-95**
  - **`shipwreck`:** **43.5% Precision** | **54.5% Recall** | **55.0% mAP50**
  - **`mine_cylinder`:** **42.1% Precision** | **55.4% Recall** | **45.3% mAP50**
  - **`pipeline`:** **100% Precision** | **4.5% Recall** | **4.5% mAP50**
  - **`crab_pot`:** **39.7% Precision** | **18.0% Recall** | **23.4% mAP50**
  - **Overall Model Summary:** **64.4% Precision** | **45.0% Recall** | **45.0% mAP50** | **28.3% mAP50-95**
  - **Edge Latency:** **3.9 ms inference (~256 FPS)** on RTX 5050 (0.2ms pre, 0.6ms post)
* **Training Methodology:**
  - **`close_mosaic=10`:** Disables mosaic augmentation during the final 10 epochs. This preserves fine acoustic shadow boundaries that would otherwise be chopped across mosaic seams.
  - **Hard Negative Mining:** 2,383 background-only seabed scans included to penalize false alarms on sand ripples and natural rock formations.
  - **Automatic Mixed Precision (`amp=True`):** Accelerated FP16 training with numerical stability.
  - **Memory Paging Guard:** Fixed with `workers=0` for Windows paging file stability.

---

## ✨ Acoustic Image Enhancement Engine (`sonar_enhancer.py`)

Raw side-scan sonar often suffers from acoustic speckle, lateral attenuation, and low acoustic contrast. This repository includes an integrated enhancement engine:
- **Acoustic CLAHE:** Contrast Limited Adaptive Histogram Equalization boosts faint acoustic shadows without amplifying seafloor noise.
- **Bilateral Despeckling:** Edge-preserving smoothing filter eliminates acoustic speckle while keeping target boundaries sharp.
- **Lateral TVG (Time-Varying Gain):** Compensates for signal decay across the sonar swath (far-range boost).

Enhancement can be toggled on or off directly inside the UI.

---

## 🛠️ Getting Started (For Team Members)

### 1. Clone the Repository
```bash
git clone https://github.com/SnippyCodes/Sonnet-Model-.git
cd Sonnet-Model-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Dashboard
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:7860`.

---

## 🎨 Dashboard Features

1. **Model Switcher:** Toggle between **Model B** (pure high-accuracy shipwreck detector) and **Model C** (5-class multi-hazard & ghost net detector).
2. **Acoustic Enhancement Checkbox:** Enable one-click CLAHE + Bilateral filtering.
3. **Telemetry & Actionable Reporting:** Generates structured JSON reports containing:
   - Hazard Class & Confidence percentage
   - Estimated bounding dimensions in meters (assuming 0.05 m/px standard scale)
   - Autonomous AUV recommendation (`IMMEDIATE_AVOIDANCE`, `REMOVAL_COORDINATES`, `TOW_CABLE_STANDOFF`)
4. **Color-Coded Visual Overlays:**
   - 🟦 **Ghost Net:** Cyan (`#00E5FF`)
   - 🟨 **Shipwreck:** Amber (`#FFB703`)
   - 🟧 **Pipeline:** Sky Blue (`#39A0ED`)
   - 🟥 **Mine / UXO:** Crimson (`#E63946`)
   - 🟪 **Crab Pot:** Violet (`#9B5DE5`)

---

## 👥 Contributors & Acknowledgements
- **Team Synora** — Smart India Hackathon (SIH 2026)
- **Ministry of Earth Sciences (MoES) & National Institute of Ocean Technology (NIOT)** for Problem Statement PS 26057.
