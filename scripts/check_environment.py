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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    missing_models = [path for path in MODEL_FILES if not (root / path).exists()]

    print(f"Project root: {root}")
    print("Python modules:")
    for name in REQUIRED_MODULES:
        print(f"  {name}: {'MISSING' if name in missing_modules else 'OK'}")

    print("Model files:")
    for path in MODEL_FILES:
        print(f"  {path}: {'MISSING' if path in missing_models else 'OK'}")

    if missing_modules or missing_models:
        print("Environment is not ready.")
        return 1

    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

