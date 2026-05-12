from __future__ import annotations

from pathlib import Path

from .image_io import load_rgb_image
from .path_utils import normalize_platform_path


def save_score_distribution(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("similarity") not in ("", None):
            grouped.setdefault(row["lora_id"], []).append(float(row["similarity"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not grouped:
        return

    labels = list(grouped)
    data = [grouped[label] for label in labels]
    fig_width = max(10, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.boxplot(data, labels=[label[-7:] for label in labels], showmeans=True)
    ax.set_title("LoRA Face Similarity Score Distribution")
    ax.set_xlabel("LoRA checkpoint")
    ax.set_ylabel("cosine similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _score_key(row: dict) -> float:
    try:
        return float(row.get("similarity", "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _load_contact_sheet_font(size: int):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttf",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        path = normalize_platform_path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def save_best_worst_contact_sheet(rows: list[dict], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    valid = [row for row in rows if row.get("status") == "ok" and row.get("similarity") not in ("", None)]
    if not valid:
        return

    selected: list[tuple[str, dict]] = []
    by_lora: dict[str, list[dict]] = {}
    for row in valid:
        by_lora.setdefault(row["lora_id"], []).append(row)

    for lora_id in sorted(by_lora):
        group = sorted(by_lora[lora_id], key=_score_key)
        selected.append(("worst", group[0]))
        selected.append(("best", group[-1]))

    thumb_w, thumb_h = 220, 300
    label_h = 58
    cols = 5
    rows_count = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _load_contact_sheet_font(size=13)

    for index, (kind, row) in enumerate(selected):
        col = index % cols
        row_index = index // cols
        x = col * thumb_w
        y = row_index * (thumb_h + label_h)

        try:
            image = load_rgb_image(normalize_platform_path(row["file_path"]))
            image.thumbnail((thumb_w, thumb_h))
            paste_x = x + (thumb_w - image.width) // 2
            paste_y = y + (thumb_h - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
        except Exception:
            draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline="red")

        file_name = Path(str(row["file_path"]).replace("\\", "/")).name
        scene = str(row.get("scene") or "")
        label = (
            f"{kind.upper()} {row['lora_id'][-7:]}  score={float(row['similarity']):.4f}\n"
            f"{_fit_text(scene, 18)}\n"
            f"{_fit_text(file_name, 20)}"
        )
        draw.text((x + 6, y + thumb_h + 4), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
