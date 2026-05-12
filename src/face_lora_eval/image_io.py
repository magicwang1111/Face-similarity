from __future__ import annotations

from pathlib import Path


def load_rgb_image(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def load_bgr_array(path: Path):
    import numpy as np

    rgb = load_rgb_image(path)
    array = np.asarray(rgb)
    return array[:, :, ::-1].copy()

