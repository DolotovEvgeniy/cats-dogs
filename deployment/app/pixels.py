"""Preprocessing used for PyTorch, calibration and HTTP inference."""
import numpy as np
from PIL import Image, ImageOps


def pixels(image, size):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((array - 0.5) / 0.5).transpose(2, 0, 1))
