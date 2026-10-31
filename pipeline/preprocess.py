"""
Input & Canvas Preprocessing Module for Sketch2Shape AI
TechZephyr 4.0 Hackathon (IIT Bhubaneswar)

Handles:
- Ingestion of uploaded 2D images / photos (.png, .jpg, .jpeg, .webp) and Gradio Sketchpad strokes.
- High-fidelity RGB photo preprocessing preserving original colors and textures.
- Sketch normalization: dark-mode inversion, contrast enhancement, aspect-ratio padding to 512x512.
- Robust blank/empty canvas detection guarding against invalid runs.
- Automatic input classification (photo/colored asset vs hand sketch).
"""

from typing import Any, Tuple, Union
import numpy as np
from PIL import Image
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

    # 1. Gradio Sketchpad dictionary format
    if isinstance(input_data, dict):
        if "composite" in input_data and input_data["composite"] is not None:
            comp = input_data["composite"]
            if isinstance(comp, Image.Image):
                return comp
            elif isinstance(comp, np.ndarray):
                return Image.fromarray(comp)
            elif isinstance(comp, str):
                return Image.open(comp)

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


def is_canvas_empty(
    input_data: Any,
    stroke_threshold: float = 0.001,
) -> bool:
    """
    Validate whether the input canvas or uploaded image contains actual content.
    Returns True if empty/blank, False if valid content exists.
    """
    if input_data is None:
        return True

    try:
        raw = extract_raw_image(input_data)
        if raw.size[0] == 0 or raw.size[1] == 0:
            return True

        # Check for empty transparent canvas
        if raw.mode == "RGBA":
            alpha = np.array(raw.split()[-1])
            if np.count_nonzero(alpha > 15) == 0:
                return True

        # Check for completely uniform/solid color canvas (zero variance)
        np_gray = np.array(raw.convert("L"))
        std_dev = float(np.std(np_gray))
        if std_dev < 1.0:
            return True

        return False
    except Exception:
        return True


def detect_input_type(input_data: Any) -> str:
    """
    Detect whether the input image is a photographic/colored asset ('photo')
    or a sparse hand-drawn line sketch ('sketch').
    """
    try:
        img = extract_raw_image(input_data).convert("RGB")
        np_rgb = np.array(img)
        hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))

        # Calculate brightness distribution
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        white_ratio = float(np.count_nonzero(gray > 225)) / gray.size
        dark_ratio = float(np.count_nonzero(gray < 30)) / gray.size

        # If image has vivid colors or textured gradients, it's a photo
        if mean_saturation > 25.0:
            return "photo"
        # If image is almost entirely white background with sparse lines -> sketch
        if white_ratio > 0.70:
            return "sketch"
        # If image is dark chalkboard with white strokes -> sketch
        if dark_ratio > 0.70:
            return "sketch"

        return "photo"
    except Exception:
        return "photo"


def preprocess_photo(
    input_data: Any,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Preprocess an uploaded 2D image or photo preserving 100% of its RGB colors,
    saturation, textures, and details, padding cleanly into target_size.
    """
    raw_img = extract_raw_image(input_data)
    rgba = raw_img.convert("RGBA")

    target_w, target_h = target_size
    orig_w, orig_h = rgba.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Composite onto clean white background
    final_canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    final_canvas.paste(resized, (paste_x, paste_y), mask=resized)

    return final_canvas.convert("RGB")


def normalize_sketch(
    input_data: Any,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Preprocess and normalize a hand-drawn sketch:
    - Inverts dark-mode strokes to standard black lines on clean white background.
    - Enhances stroke line contrast.
    - Centers and pads to target_size (512, 512) preserving aspect ratio.
    """
    raw_img = extract_raw_image(input_data)
    rgba = raw_img.convert("RGBA")
    r, g, b, a = rgba.split()
    np_a = np.array(a)
    np_rgb = np.array(rgba.convert("RGB"))

    has_transparency = np.any(np_a < 250)

    if has_transparency:
        mask = np_a > 30
        if np.any(mask):
            stroke_pixels = np_rgb[mask]
            stroke_brightness = np.mean(stroke_pixels)
            if stroke_brightness > 128:
                inverted_rgb = 255 - np_rgb
                new_rgba = np.dstack((inverted_rgb, np_a))
                rgba = Image.fromarray(new_rgba, "RGBA")

        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, rgba).convert("RGB")
    else:
        composite = raw_img.convert("RGB")

    np_comp = np.array(composite)
    gray = cv2.cvtColor(np_comp, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    # Invert dark background canvas to white background
    if mean_brightness < 127:
        gray = 255 - gray

    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary_mask = cv2.threshold(smoothed, 220, 255, cv2.THRESH_BINARY)
    clean_sketch = np.where(binary_mask == 255, 255, smoothed)

    clean_pil = Image.fromarray(clean_sketch).convert("RGB")

    target_w, target_h = target_size
    orig_w, orig_h = clean_pil.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized_stroke = clean_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    final_canvas = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    final_canvas.paste(resized_stroke, (paste_x, paste_y))

    return final_canvas
