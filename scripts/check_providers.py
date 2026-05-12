from __future__ import annotations

import onnxruntime as ort


def main() -> int:
    print(f"onnxruntime: {ort.__version__}")
    print("available providers:")
    for provider in ort.get_available_providers():
        print(f"  - {provider}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch cuda device count: {torch.cuda.device_count()}")
            print(f"torch cuda device 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch check failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
