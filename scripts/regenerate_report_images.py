from __future__ import annotations

import csv
from pathlib import Path

from face_lora_eval.report import save_best_worst_contact_sheet, save_score_distribution


def main() -> int:
    report_dir = Path("reports/xiaohan_20260507")
    csv_path = report_dir / "per_image_scores.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    save_score_distribution(rows, report_dir / "score_distribution.png")
    save_best_worst_contact_sheet(rows, report_dir / "best_worst_contact_sheet.png")
    print(f"Regenerated report images from {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

