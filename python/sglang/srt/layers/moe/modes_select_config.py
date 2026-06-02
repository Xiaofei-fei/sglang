# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Select the best measured MoDES config under an accuracy-drop constraint.

This helper intentionally works from measured benchmark result files. The
SGLang server can auto-calibrate route statistics, but it cannot know task
accuracy without labels and an evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


def _first_number(obj: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        cur: Any = obj
        ok = True
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok and cur is not None:
            return float(cur)
    return None


def _metric(obj: dict[str, Any], name: str) -> float | None:
    aliases = {
        "accuracy": ("accuracy", "summary.accuracy", "result.accuracy"),
        "samples_per_s": (
            "samples_per_s",
            "samples_per_second",
            "summary.samples_per_s",
            "result.samples_per_s",
        ),
        "new_tokens_per_s": (
            "new_tokens_per_s",
            "new_tokens_per_second",
            "summary.new_tokens_per_s",
            "result.new_tokens_per_s",
        ),
        "latency": (
            "avg_latency_s",
            "average_latency_s",
            "summary.avg_latency_s",
            "result.avg_latency_s",
        ),
    }
    return _first_number(obj, aliases.get(name, (name,)))


def _candidate_config(obj: dict[str, Any]) -> dict[str, Any]:
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
    config = obj.get("config") if isinstance(obj.get("config"), dict) else {}
    selected = {
        "alpha": obj.get("alpha")
        or config.get("alpha")
        or metrics.get("calibrated_alpha"),
        "tau_text": obj.get("tau_text")
        or config.get("tau_text")
        or metrics.get("effective_tau_text"),
        "target_skip_rate": obj.get("target_skip_rate")
        or config.get("target_skip_rate")
        or metrics.get("target_skip_rate"),
        "active_skip_rate": obj.get("active_skip_rate")
        or metrics.get("active_skip_rate"),
    }
    return {key: value for key, value in selected.items() if value is not None}


def select_best(
    baseline: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    max_accuracy_drop: float,
    optimize: str,
) -> dict[str, Any]:
    baseline_accuracy = _metric(baseline, "accuracy")
    if baseline_accuracy is None:
        raise ValueError("Baseline result does not contain an accuracy field.")

    feasible = []
    rejected = []
    for path, obj in candidates:
        accuracy = _metric(obj, "accuracy")
        score = _metric(obj, optimize)
        if accuracy is None:
            raise ValueError(f"Candidate lacks accuracy: {path}")
        if score is None:
            raise ValueError(f"Candidate lacks optimize metric '{optimize}': {path}")

        accuracy_drop = baseline_accuracy - accuracy
        record = {
            "path": path,
            "accuracy": accuracy,
            "accuracy_drop": accuracy_drop,
            "score": score,
            "config": _candidate_config(obj),
        }
        if accuracy_drop <= max_accuracy_drop:
            feasible.append(record)
        else:
            rejected.append(record)

    if not feasible:
        return {
            "baseline_accuracy": baseline_accuracy,
            "max_accuracy_drop": max_accuracy_drop,
            "optimize": optimize,
            "best": None,
            "feasible": [],
            "rejected": rejected,
        }

    reverse = optimize != "latency"
    best = sorted(feasible, key=lambda x: x["score"], reverse=reverse)[0]
    selected_config = {
        "version": 1,
        **best["config"],
        "baseline_accuracy": baseline_accuracy,
        "selected_accuracy": best["accuracy"],
        "accuracy_drop": best["accuracy_drop"],
        "optimized_metric": optimize,
        "optimized_score": best["score"],
        "source_result_path": best["path"],
    }
    return {
        "baseline_accuracy": baseline_accuracy,
        "max_accuracy_drop": max_accuracy_drop,
        "optimize": optimize,
        "best": best,
        "selected_config": selected_config,
        "feasible": feasible,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best measured MoDES config under an accuracy constraint."
    )
    parser.add_argument("--baseline", required=True, help="Default baseline result JSON.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Measured MoDES candidate result JSON. Can be passed more than once.",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.05,
        help="Maximum allowed absolute accuracy drop. Default: 0.05.",
    )
    parser.add_argument(
        "--optimize",
        default="samples_per_s",
        choices=("samples_per_s", "new_tokens_per_s", "latency"),
        help="Metric to optimize among feasible candidates.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    baseline = _load_json(args.baseline)
    candidates = [(path, _load_json(path)) for path in args.candidate]
    result = select_best(
        baseline,
        candidates,
        max_accuracy_drop=args.max_accuracy_drop,
        optimize=args.optimize,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["best"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
