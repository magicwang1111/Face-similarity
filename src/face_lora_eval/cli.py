from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import EvalConfig, load_config
from .path_utils import normalize_platform_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_lora_eval",
        description="Evaluate face similarity for batches of LoRA-generated images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run LoRA face similarity evaluation.")
    evaluate.add_argument("--config", type=Path, default=Path("configs/xiaohan_20260507.yaml"))
    evaluate.add_argument("--reference-root", type=Path)
    evaluate.add_argument("--candidate-root", type=Path)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--person-id")
    evaluate.add_argument("--backend", choices=("insightface", "lvface_onnx"))
    evaluate.add_argument("--insightface-root", type=Path)
    evaluate.add_argument("--model-name")
    evaluate.add_argument("--lvface-model-path", type=Path)
    evaluate.add_argument("--det-size", type=int)
    evaluate.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNX Runtime provider. Can be repeated. Defaults to config providers.",
    )
    evaluate.add_argument(
        "--limit-reference",
        type=int,
        help="Limit reference images for a quick smoke test.",
    )
    evaluate.add_argument(
        "--limit-candidates-per-lora",
        type=int,
        help="Limit candidate images per LoRA for a quick smoke test.",
    )
    evaluate.add_argument("--skip-plots", action="store_true", help="Skip PNG plot/contact sheet output.")

    return parser


def apply_overrides(config: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    updates = {}
    for field_name in ("reference_root", "candidate_root", "output_dir"):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = normalize_platform_path(value)
    if args.person_id:
        updates["person_id"] = args.person_id
    if args.backend:
        updates["backend"] = args.backend

    insightface = dict(config.insightface)
    if args.insightface_root is not None:
        insightface["root"] = normalize_platform_path(args.insightface_root)
    if args.model_name:
        insightface["model_name"] = args.model_name
    if args.det_size:
        insightface["det_size"] = args.det_size
    if args.providers:
        insightface["providers"] = args.providers
    updates["insightface"] = insightface

    lvface = dict(config.lvface)
    if args.lvface_model_path is not None:
        lvface["model_path"] = normalize_platform_path(args.lvface_model_path)
    if args.providers:
        lvface["providers"] = args.providers
    updates["lvface"] = lvface

    if args.limit_reference is not None:
        updates["limit_reference"] = args.limit_reference
    if args.limit_candidates_per_lora is not None:
        updates["limit_candidates_per_lora"] = args.limit_candidates_per_lora
    if args.skip_plots:
        updates["skip_plots"] = True

    return config.with_updates(**updates)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        from .evaluator import run_evaluation

        config = apply_overrides(load_config(args.config), args)
        try:
            summary = run_evaluation(config)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Report directory: {summary.output_dir}")
        print(f"Ranking CSV: {summary.ranking_csv}")
        print("Top LoRA checkpoints:")
        for index, row in enumerate(summary.top_rows[:5], start=1):
            print(
                f"{index}. {row['lora_id']} "
                f"final={row['final_score']} mean={row['mean']} "
                f"p25={row['p25']} fail_rate={row['fail_rate']}"
            )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
