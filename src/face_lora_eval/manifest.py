from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    role: str
    person_id: str
    file_path: Path
    lora_id: str = ""
    scene: str = ""
    width: int | None = None
    height: int | None = None
    status: str = "pending"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.size
    except Exception:
        return None, None


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_image(path))


def build_manifest(
    reference_root: Path,
    candidate_root: Path,
    person_id: str,
    limit_reference: int | None = None,
    limit_candidates_per_lora: int | None = None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    reference_paths = iter_images(reference_root)
    if limit_reference is not None:
        reference_paths = reference_paths[:limit_reference]

    for path in reference_paths:
        width, height = safe_image_size(path)
        records.append(
            ImageRecord(
                role="reference",
                person_id=person_id,
                file_path=path.resolve(),
                width=width,
                height=height,
            )
        )

    lora_dirs = sorted(path for path in candidate_root.iterdir() if path.is_dir())
    for lora_dir in lora_dirs:
        image_paths = iter_images(lora_dir)
        if limit_candidates_per_lora is not None:
            image_paths = image_paths[:limit_candidates_per_lora]
        for path in image_paths:
            relative = path.relative_to(lora_dir)
            scene = "" if len(relative.parts) == 1 else str(Path(*relative.parts[:-1]))
            width, height = safe_image_size(path)
            records.append(
                ImageRecord(
                    role="candidate",
                    person_id=person_id,
                    lora_id=lora_dir.name,
                    scene=scene,
                    file_path=path.resolve(),
                    width=width,
                    height=height,
                )
            )

    return records

