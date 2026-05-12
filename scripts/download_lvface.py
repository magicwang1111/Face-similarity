from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ID = "bytedance-research/LVFace"
FILENAME = "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"
DEFAULT_TARGET = Path("models/lvface/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the official LVFace-B_Glint360K ONNX model.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing target file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target
    if target.exists() and not args.force:
        print(f"LVFace model already exists: {target}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install with "
            "`python -m pip install huggingface_hub` or `python -m pip install -r requirements.txt`."
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = target.parent / ".hf_cache"
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            cache_dir=str(cache_dir),
        )
    )
    shutil.copy2(downloaded, target)
    print(f"Downloaded {args.repo_id}/{args.filename}")
    print(f"Saved to {target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
