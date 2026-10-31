"""
Foreground Isolation & Background Stripping Module for Sketch2Shape AI
TechZephyr 4.0 Hackathon (IIT Bhubaneswar)

Pipelines:
- rembg (U2-Net) background removal
- Clean alpha mask extraction
- Re-compositing onto a pure white background (255, 255, 255) for TripoSR 3D reconstruction
- Robust OpenCV/GrabCut fallback if rembg model is unavailable
"""

import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger("sketch2shape.segmentation")


class Segmenter:
    """
    Strips image backgrounds using rembg (U2-Net) and re-composites foreground
    objects onto pure white backgrounds as required by 3D reconstruction models.
    """

    def __init__(self, session=None):
        self.session = session
        self.rembg_available = False
        try:
            import rembg
            self.rembg_available = True
            logger.info("rembg (U2-Net) background removal engine initialized.")
        except ImportError:
            logger.warning("rembg is not installed. OpenCV GrabCut fallback will be used.")

    def isolate_foreground(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Removes background from image.

        Returns:
            rgba_image: PIL Image with alpha transparency.
            white_bg_image: PIL RGB Image cleanly composited on pure white background.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.rembg_available:
            try:
                import rembg
                # Strip background with rembg
                rgba_result = rembg.remove(image, session=self.session)
                if rgba_result.mode != "RGBA":
                    rgba_result = rgba_result.convert("RGBA")

                # Composite over pure white
                white_bg = Image.new("RGBA", rgba_result.size, (*bg_color, 255))
                composite_white = Image.alpha_composite(white_bg, rgba_result).convert("RGB")
                return rgba_result, composite_white
            except Exception as e:
                logger.warning("rembg execution failed (%s), switching to OpenCV fallback.", str(e))

        # Fallback segmentation using OpenCV thresholding & alpha mask
        return self._opencv_fallback_isolate(image, bg_color)

    def _opencv_fallback_isolate(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[Image.Image, Image.Image]:
        """
        OpenCV-based fallback segmentation using color distance and edge masking.
        """
        np_img = np.array(image)
        h, w, _ = np_img.shape

        # Measure distance from pure white corners
        corners = [
            np_img[0:15, 0:15],
            np_img[0:15, w-15:w],
            np_img[h-15:h, 0:15],
            np_img[h-15:h, w-15:w],
        ]
        corner_colors = np.mean(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)

        # Distance from background color
        diff = np.linalg.norm(np_img.astype(np.float32) - corner_colors, axis=2)
        # Threshold: foreground pixels differ significantly from corners
        mask = (diff > 25).astype(np.uint8) * 255

        # Morphological close & open to clean up holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Build RGBA
        r, g, b = cv2.split(np_img)
        rgba_np = cv2.merge([r, g, b, mask])
        rgba_image = Image.fromarray(rgba_np, "RGBA")

        # Composite over clean white
        white_bg = Image.new("RGBA", image.size, (*bg_color, 255))
        composite_white = Image.alpha_composite(white_bg, rgba_image).convert("RGB")

        return rgba_image, composite_white
