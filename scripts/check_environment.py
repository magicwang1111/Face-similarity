from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = [
    "insightface",
    "onnxruntime",
    "cv2",
    "PIL",
    "numpy",
    "matplotlib",
]

MODEL_FILES = [
    "models/insightface/models/buffalo_l/det_10g.onnx",
    "models/insightface/models/buffalo_l/w600k_r50.onnx",
    "models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
    "models/insightface/models/antelopev2/glintr100.onnx",
]

LVFACE_MODEL_FILES = [
    "models/lvface/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    missing_models = [path for path in MODEL_FILES if not (root / path).exists()]
    missing_lvface_models = [path for path in LVFACE_MODEL_FILES if not (root / path).exists()]

    print(f"Project root: {root}")
    print("Python modules:")
    for name in REQUIRED_MODULES:
        print(f"  {name}: {'MISSING' if name in missing_modules else 'OK'}")

    print("Model files:")
    for path in MODEL_FILES:
        print(f"  {path}: {'MISSING' if path in missing_models else 'OK'}")
    print("LVFace model files:")
    for path in LVFACE_MODEL_FILES:
        print(f"  {path}: {'MISSING' if path in missing_lvface_models else 'OK'}")

    if missing_modules or missing_models:
        print("Environment is not ready.")
        return 1

    if missing_lvface_models:
        print("Core environment looks ready. LVFace model is missing; run `python scripts/download_lvface.py` for LVFace evaluation.")
        return 0

    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
