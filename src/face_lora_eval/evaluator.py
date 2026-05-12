from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

from .config import EvalConfig
from .csv_io import write_csv
from .face_model import create_embedder, l2_normalize
from .manifest import ImageRecord, build_manifest
from .scoring import RankingWeights, aggregate_lora_scores


@dataclass(frozen=True)
class EvaluationSummary:
    output_dir: Path
    ranking_csv: Path
    top_rows: list[dict[str, Any]]


def _dot_similarity(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.dot(left, right))


def _centroid(embeddings: list[Any]):
    import numpy as np

    if not embeddings:
        raise ValueError("No valid reference embeddings were extracted.")
    return l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))


def _quality_row(record: ImageRecord, status: str, quality: Any = None, error: str = "") -> dict:
    row = {
        "role": record.role,
        "person_id": record.person_id,
        "lora_id": record.lora_id,
        "scene": record.scene,
        "file_path": str(record.file_path),
        "status": status,
        "error": error,
        "det_score": "",
        "face_area_ratio": "",
        "face_width": "",
        "face_height": "",
        "center_offset": "",
        "selection_score": "",
        "num_faces": "",
    }
    if quality is not None:
        row.update(
            {
                "det_score": quality.det_score,
                "face_area_ratio": quality.face_area_ratio,
                "face_width": quality.face_width,
                "face_height": quality.face_height,
                "center_offset": quality.center_offset,
                "selection_score": quality.selection_score,
                "num_faces": quality.num_faces,
            }
        )
    return row


def _score_row(record: ImageRecord, status: str, similarity: float | str = "", quality: Any = None, error: str = "") -> dict:
    row = _quality_row(record, status=status, quality=quality, error=error)
    row["similarity"] = similarity
    return row


def _is_valid_reference(quality: Any, config: EvalConfig) -> bool:
    min_det_score = config.reference_quality["min_det_score"]
    min_face_area_ratio = config.reference_quality["min_face_area_ratio"]
    return quality.det_score >= min_det_score and quality.face_area_ratio >= min_face_area_ratio


def run_evaluation(config: EvalConfig) -> EvaluationSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        reference_root=config.reference_root,
        candidate_root=config.candidate_root,
        person_id=config.person_id,
        limit_reference=config.limit_reference,
        limit_candidates_per_lora=config.limit_candidates_per_lora,
    )
    write_csv(config.output_dir / "manifest.csv", manifest)

    embedder = create_embedder(config)
    embedder.prepare()

    reference_records = [record for record in manifest if record.role == "reference"]
    candidate_records = [record for record in manifest if record.role == "candidate"]

    reference_rows: list[dict] = []
    reference_embeddings: list[Any] = []
    for record in tqdm(reference_records, desc="Reference embeddings"):
        try:
            result = embedder.extract(record.file_path)
            if result is None:
                reference_rows.append(_quality_row(record, status="no_face"))
                continue
            if not _is_valid_reference(result.quality, config):
                reference_rows.append(_quality_row(record, status="low_quality_reference", quality=result.quality))
                continue
            reference_embeddings.append(result.embedding)
            reference_rows.append(_quality_row(record, status="ok", quality=result.quality))
        except Exception as exc:
            reference_rows.append(_quality_row(record, status="error", error=str(exc)))

    write_csv(config.output_dir / "reference_quality.csv", reference_rows)
    if not reference_embeddings:
        raise RuntimeError("No valid reference embeddings. Check reference images, detection size, or quality thresholds.")

    reference_centroid = _centroid(reference_embeddings)

    score_rows: list[dict] = []
    for record in tqdm(candidate_records, desc="Candidate scores"):
        try:
            result = embedder.extract(record.file_path)
            if result is None:
                score_rows.append(_score_row(record, status="no_face"))
                continue
            similarity = _dot_similarity(result.embedding, reference_centroid)
            score_rows.append(_score_row(record, status="ok", similarity=similarity, quality=result.quality))
        except Exception as exc:
            score_rows.append(_score_row(record, status="error", error=str(exc)))

    per_image_fields = [
        "role",
        "person_id",
        "lora_id",
        "scene",
        "file_path",
        "status",
        "similarity",
        "error",
        "det_score",
        "face_area_ratio",
        "face_width",
        "face_height",
        "center_offset",
        "selection_score",
        "num_faces",
    ]
    write_csv(config.output_dir / "per_image_scores.csv", score_rows, fieldnames=per_image_fields)

    failed_rows = [row for row in reference_rows + score_rows if row["status"] != "ok"]
    write_csv(config.output_dir / "failed_images.csv", failed_rows)

    weights = RankingWeights(
        mean_weight=config.ranking["mean_weight"],
        median_weight=config.ranking["median_weight"],
        p25_weight=config.ranking["p25_weight"],
        fail_rate_weight=config.ranking["fail_rate_weight"],
    )
    ranking_rows = aggregate_lora_scores(score_rows, weights=weights)
    ranking_fields = [
        "lora_id",
        "total_count",
        "valid_count",
        "failed_count",
        "fail_rate",
        "mean",
        "median",
        "p25",
        "max",
        "min",
        "std",
        "final_score",
    ]
    ranking_csv = config.output_dir / "lora_ranking.csv"
    write_csv(ranking_csv, ranking_rows, fieldnames=ranking_fields)

    if not config.skip_plots:
        from .report import save_best_worst_contact_sheet, save_score_distribution

        save_score_distribution(score_rows, config.output_dir / "score_distribution.png")
        save_best_worst_contact_sheet(score_rows, config.output_dir / "best_worst_contact_sheet.png")

    return EvaluationSummary(output_dir=config.output_dir, ranking_csv=ranking_csv, top_rows=ranking_rows)
