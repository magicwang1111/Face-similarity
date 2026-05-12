from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from statistics import mean, median, pstdev
from typing import Iterable


@dataclass(frozen=True)
class RankingWeights:
    mean_weight: float = 0.45
    median_weight: float = 0.25
    p25_weight: float = 0.20
    fail_rate_weight: float = 0.10


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def safe_float(value: float) -> float | str:
    return "" if isnan(value) else round(value, 6)


def aggregate_lora_scores(
    rows: Iterable[dict],
    weights: RankingWeights,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["lora_id"], []).append(row)

    ranking_rows: list[dict] = []
    for lora_id, group in grouped.items():
        valid_scores = [
            float(row["similarity"])
            for row in group
            if row.get("status") == "ok" and row.get("similarity") not in ("", None)
        ]
        total_count = len(group)
        valid_count = len(valid_scores)
        failed_count = total_count - valid_count
        fail_rate = failed_count / total_count if total_count else 1.0

        if valid_scores:
            mean_score = mean(valid_scores)
            median_score = median(valid_scores)
            p25_score = percentile(valid_scores, 0.25)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
            std_score = pstdev(valid_scores) if len(valid_scores) > 1 else 0.0
            final_score = (
                weights.mean_weight * mean_score
                + weights.median_weight * median_score
                + weights.p25_weight * p25_score
                - weights.fail_rate_weight * fail_rate
            )
        else:
            mean_score = median_score = p25_score = max_score = min_score = std_score = final_score = float("nan")

        ranking_rows.append(
            {
                "lora_id": lora_id,
                "total_count": total_count,
                "valid_count": valid_count,
                "failed_count": failed_count,
                "fail_rate": safe_float(fail_rate),
                "mean": safe_float(mean_score),
                "median": safe_float(median_score),
                "p25": safe_float(p25_score),
                "max": safe_float(max_score),
                "min": safe_float(min_score),
                "std": safe_float(std_score),
                "final_score": safe_float(final_score),
            }
        )

    ranking_rows.sort(key=lambda row: float(row["final_score"] or "-inf"), reverse=True)
    return ranking_rows

