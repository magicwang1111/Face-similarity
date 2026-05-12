from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: Iterable[Any], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for row in rows:
        if is_dataclass(row):
            data = asdict(row)
        else:
            data = dict(row)
        materialized.append({key: _stringify(value) for key, value in data.items()})

    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else []

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)

