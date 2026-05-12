from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .path_utils import normalize_platform_path


@dataclass(frozen=True)
class EvalConfig:
    person_id: str
    reference_root: Path
    candidate_root: Path
    output_dir: Path
    insightface: dict[str, Any]
    reference_quality: dict[str, float]
    ranking: dict[str, float]
    limit_reference: int | None = None
    limit_candidates_per_lora: int | None = None
    skip_plots: bool = False

    def with_updates(self, **updates: Any) -> "EvalConfig":
        return replace(self, **updates)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_with_simple_yaml(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("-") and current_section and current_list_key:
            value = stripped[1:].strip()
            raw[current_section][current_list_key].append(_parse_scalar(value))
            continue

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0:
            current_section = key if value == "" else None
            current_list_key = None
            raw[key] = {} if value == "" else _parse_scalar(value)
            continue

        if indent >= 2 and current_section:
            if value == "":
                raw[current_section][key] = []
                current_list_key = key
            else:
                raw[current_section][key] = _parse_scalar(value)
                current_list_key = None

    return raw


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _load_with_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: Path) -> EvalConfig:
    raw = _load_yaml(path)

    insightface = raw.get("insightface") or {}
    reference_quality = raw.get("reference_quality") or {}
    ranking = raw.get("ranking") or {}

    return EvalConfig(
        person_id=str(raw.get("person_id", "person")),
        reference_root=normalize_platform_path(raw["reference_root"]),
        candidate_root=normalize_platform_path(raw["candidate_root"]),
        output_dir=normalize_platform_path(raw["output_dir"]),
        insightface={
            "root": normalize_platform_path(insightface.get("root", "~/.insightface")),
            "model_name": insightface.get("model_name", "buffalo_l"),
            "det_size": int(insightface.get("det_size", 1024)),
            "providers": list(insightface.get("providers") or ["CPUExecutionProvider"]),
        },
        reference_quality={
            "min_det_score": float(reference_quality.get("min_det_score", 0.50)),
            "min_face_area_ratio": float(reference_quality.get("min_face_area_ratio", 0.001)),
        },
        ranking={
            "mean_weight": float(ranking.get("mean_weight", 0.45)),
            "median_weight": float(ranking.get("median_weight", 0.25)),
            "p25_weight": float(ranking.get("p25_weight", 0.20)),
            "fail_rate_weight": float(ranking.get("fail_rate_weight", 0.10)),
        },
    )
