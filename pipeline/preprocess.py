"""
Input & Canvas Preprocessing Module for Sketch2Shape AI
TechZephyr 4.0 Hackathon (IIT Bhubaneswar)

Handles:
- Ingestion of Gradio Sketchpad (composite, layers, background) and uploaded images (.png, .jpg).
- Detection and inversion of dark-mode strokes to standard black strokes on a clean white background.
- Aspect-ratio preserving normalization, padding, and center-resizing to 512x512 with OpenCV/PIL.
- Empty canvas detection to guard against invalid pipeline execution.
"""

from typing import Any, Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import cv2


def extract_raw_image(input_data: Any) -> Image.Image:
    """
    Extract a PIL.Image from various Gradio input types:
    - Gradio 4+ Sketchpad dict: {'background': ..., 'layers': [...], 'composite': ...}
    - Gradio 3 Image/Mask dict: {'image': ..., 'mask': ...}
    - PIL.Image.Image
    - NumPy ndarray
    - File path string
    """
    if input_data is None:
        raise ValueError("No input data provided to preprocessor.")

    # 1. Gradio 4+ Sketchpad dictionary format
    if isinstance(input_data, dict):
        # Prefer composite if available and not empty
        if "composite" in input_data and input_data["composite"] is not None:
            comp = input_data["composite"]
            if isinstance(comp, Image.Image):
                return comp
            elif isinstance(comp, np.ndarray):
                return Image.fromarray(comp)
            elif isinstance(comp, str):
                return Image.open(comp)

        # Fallback to layers composite
        if "layers" in input_data and input_data["layers"]:
            base = None
            if "background" in input_data and input_data["background"] is not None:
                bg = input_data["background"]
                base = bg if isinstance(bg, Image.Image) else Image.fromarray(bg)
                base = base.convert("RGBA")

            for layer in input_data["layers"]:
                if layer is None:
                    continue
                layer_img = layer if isinstance(layer, Image.Image) else Image.fromarray(layer)
                layer_img = layer_img.convert("RGBA")
                if base is None:
                    base = layer_img
                else:
                    base = Image.alpha_composite(base, layer_img)

            if base is not None:
                return base

        # Older dict format with 'image' key
        if "image" in input_data and input_data["image"] is not None:
            img = input_data["image"]
            return img if isinstance(img, Image.Image) else Image.fromarray(img)

        raise ValueError(f"Unrecognized dictionary format from Sketchpad: {list(input_data.keys())}")

    # 2. PIL Image directly
    if isinstance(input_data, Image.Image):
        return input_data

    # 3. NumPy ndarray
    if isinstance(input_data, np.ndarray):
        return Image.fromarray(input_data)

    # 4. Filepath string
    if isinstance(input_data, str):
        return Image.open(input_data)

    raise TypeError(f"Unsupported input type for canvas preprocessing: {type(input_data)}")


def normalize_sketch(
    input_data: Any,
    target_size: Tuple[int, int] = (512, 512),
    stroke_threshold: float = 0.001,
) -> Image.Image:
    """
    Preprocess and normalize hand-drawn sketch or uploaded image into a standard
    512x512 black-stroke-on-white-background image.

    Steps:
    1. Extract raw image from Gradio component.
    2. Handle alpha channels and composite transparent drawings onto pure white.
    3. Invert dark-mode strokes (light strokes on dark background) to black strokes on white.
    4. Contrast-enhance strokes using thresholding/adaptive filtering.
    5. Pad and resize to exact 512x512 maintaining aspect ratio without stretching.
    """
    raw_img = extract_raw_image(input_data)

    # Convert to RGBA for alpha evaluation
    rgba = raw_img.convert("RGBA")
    r, g, b, a = rgba.split()
    np_a = np.array(a)
    np_rgb = np.array(rgba.convert("RGB"))

    # Determine if drawing was on transparent canvas
    has_transparency = np.any(np_a < 250)

    if has_transparency:
        # Check color of painted strokes (where alpha > 30)
        mask = np_a > 30
        if np.any(mask):
            stroke_pixels = np_rgb[mask]
            stroke_brightness = np.mean(stroke_pixels)
            # If strokes are light/white, invert them to dark strokes
            if stroke_brightness > 128:
                inverted_rgb = 255 - np_rgb
                # Create clean black strokes on transparent
                new_rgba = np.dstack((inverted_rgb, np_a))
                rgba = Image.fromarray(new_rgba, "RGBA")

        # Composite over a clean white background
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, rgba).convert("RGB")
    else:
        composite = raw_img.convert("RGB")

    # Evaluate overall brightness to catch dark mode canvas (dark background)
    np_comp = np.array(composite)
    gray = cv2.cvtColor(np_comp, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    # If background is dark (e.g., dark theme canvas or inverted sketch), invert to white bg
    if mean_brightness < 127:
        gray = 255 - gray

    # Enhance sketch lines: apply slight denoising and normalize line contrast
    # High-pass or bilateral filter to preserve sharp sketch boundaries
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    # Ensure background is crisp pure white (>= 240 mapped to 255)
    _, binary_mask = cv2.threshold(smoothed, 220, 255, cv2.THRESH_BINARY)
    # Blend smoothed gray with clean binary background
    clean_sketch = np.where(binary_mask == 255, 255, smoothed)

    # Convert back to PIL RGB
    clean_pil = Image.fromarray(clean_sketch).convert("RGB")

    # Resize and center-pad to target_size (512, 512) preserving aspect ratio
    target_w, target_h = target_size
    orig_w, orig_h = clean_pil.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized_stroke = clean_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create 512x512 pure white canvas
    final_canvas = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    final_canvas.paste(resized_stroke, (paste_x, paste_y))

    return final_canvas


def is_canvas_empty(
    input_data: Any,
    stroke_threshold: float = 0.001,
) -> bool:
    """
    Validate whether the input canvas contains meaningful strokes or is blank.
    Returns True if empty, False if valid sketch content exists.

    stroke_threshold: Minimum fraction of non-white pixels required (default: 0.1%).
    """
    if input_data is None:
        return True

    try:
        norm_img = normalize_sketch(input_data, target_size=(256, 256))
        gray = np.array(norm_img.convert("L"))
        # Non-white pixels (pixels with brightness < 240)
        stroke_pixels = np.count_nonzero(gray < 240)
        total_pixels = gray.size
        stroke_ratio = stroke_pixels / total_pixels

        return stroke_ratio < stroke_threshold
    except Exception:
        return True
