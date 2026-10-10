"""
====================================================================================================
  SIH 2026: SSS ACOUSTIC IMAGE ENHANCEMENT & PREPROCESSING ENGINE
  Problem Statement: PS 26057 (MoES / NIOT) | Compliance: Agent.md Component 2
  Features:
  1. Acoustic CLAHE (Contrast-Limited Adaptive Histogram Equalization) for localized acoustic shadow boost
  2. Edge-Preserving Despeckling Filter (Bilateral smoothing of coherent acoustic Rayleigh noise)
  3. Swath Dynamic Range Normalization (TVG / time-varied gain compensation)
  4. Marine Hydrographic Acoustic Colormap rendering (High-contrast Amber / Copper SSS palette)
====================================================================================================
"""

import cv2
import numpy as np
from pathlib import Path


class SonarImageEnhancer:
    """Acoustic-physics image enhancement engine for Side-Scan Sonar (SSS) waterfall scans."""

    def __init__(self, clahe_clip_limit=2.5, clahe_grid_size=(8, 8), despeckle_diameter=7, despeckle_sigma_color=50, despeckle_sigma_space=50):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.despeckle_d = despeckle_diameter
        self.sigma_color = despeckle_sigma_color
        self.sigma_space = despeckle_sigma_space

    @staticmethod
    def lee_filter(img, win_size=7):
        """
        Vectorized Adaptive Lee Filter for multiplicative acoustic speckle noise.
        Smooths homogeneous seabed reverberation while preserving sharp shadow/highlight boundaries.
        """
        img_f = img.astype(np.float32)
        k = win_size
        mean = cv2.boxFilter(img_f, -1, (k, k))
        sqr_mean = cv2.boxFilter(img_f ** 2, -1, (k, k))
        var = np.maximum(sqr_mean - mean ** 2, 0.0)
        overall_var = np.mean(var) + 1e-5
        w = var / (var + overall_var)
        w = np.clip(w, 0.0, 1.0)
        filtered = mean + w * (img_f - mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def enhance(self, img_input, apply_despeckle=True, apply_clahe=True, apply_tvg=True, use_lee_filter=True):
        """
        Enhances raw sonar waterfall imagery:
        - img_input: numpy array (BGR or Grayscale)
        Returns:
        - enhanced_bgr: 3-channel enhanced image ready for YOLO/Faster R-CNN inference
        - enhanced_gray: 1-channel high-contrast acoustic scan
        """
        # Convert to single-channel acoustic intensity
        if len(img_input.shape) == 3:
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_input.copy()

        enhanced = gray.astype(np.float32)

        # 1. TVG / Lateral Range Attenuation Compensation
        if apply_tvg:
            col_mean = np.mean(enhanced, axis=0) + 1e-4
            target_mean = np.median(col_mean)
            tvg_gain = np.clip(target_mean / col_mean, 0.7, 1.4)
            enhanced = enhanced * tvg_gain
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            enhanced = enhanced.astype(np.uint8)

        # 2. Adaptive Speckle Denoising (Lee filter or Bilateral filter)
        if apply_despeckle:
            if use_lee_filter:
                enhanced = self.lee_filter(enhanced, win_size=self.despeckle_d)
            else:
                enhanced = cv2.bilateralFilter(enhanced, d=self.despeckle_d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)

        # 3. Acoustic CLAHE (Local adaptive contrast to pop acoustic highlights and acoustic shadows)
        if apply_clahe:
            enhanced = self.clahe.apply(enhanced)

        # Return both 3-channel for PyTorch/YOLO detectors and 1-channel
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr, enhanced

    def apply_hydrographic_colormap(self, gray_enhanced):
        """Applies marine hydrographic sonar thermal/amber colormap for visual bathymetric inspection."""
        return cv2.applyColorMap(gray_enhanced, cv2.COLORMAP_HOT)


def enhance_and_save(image_path: Path, output_path: Path):
    """Utility function to enhance an image file and save before/after comparison."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    enhancer = SonarImageEnhancer()
    enhanced_bgr, enhanced_gray = enhancer.enhance(img)
    sonar_palette = enhancer.apply_hydrographic_colormap(enhanced_gray)

    # Create composite: [Raw Input | Enhanced Acoustic Gray | Hydrographic Palette]
    raw_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = raw_bgr.shape[:2]
    composite = np.hstack([raw_bgr, enhanced_bgr, sonar_palette])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    return composite


if __name__ == "__main__":
    import sys
    import glob

    test_imgs = glob.glob("pure_sonar_dataset/test/images/*.*")
    if test_imgs:
        sample_img = Path(test_imgs[0])
        out_dest = Path("reports/figures/enhanced_comparison_sample.png")
        print(f"Testing Sonar Image Enhancer on: {sample_img.name}")
        enhance_and_save(sample_img, out_dest)
        print(f"[OK] Saved enhancement comparison to: {out_dest}")
