"""Build the transparent ObserverArc model configuration from public metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_config(environment_dir: Path) -> dict:
    metadata_files = sorted(environment_dir.glob("*/*/metadata.json"))
    tag_counts: Counter[str] = Counter()
    baseline_lengths: list[int] = []
    baseline_totals: list[int] = []
    for path in metadata_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        tag_counts.update(data.get("tags", []))
        baselines = [int(value) for value in data.get("baseline_actions", [])]
        baseline_lengths.extend(baselines)
        baseline_totals.append(sum(baselines))

    mean_baseline = sum(baseline_lengths) / max(1, len(baseline_lengths))
    sorted_totals = sorted(baseline_totals)
    if sorted_totals:
        p75_total = sorted_totals[int(0.75 * (len(sorted_totals) - 1))]
    else:
        p75_total = 650
    max_actions = max(650, min(1600, int(p75_total * 1.18)))
    click_count = tag_counts.get("click", 0) + tag_counts.get("keyboard_click", 0)
    click_bias = 0.06 if click_count >= tag_counts.get("keyboard", 0) else 0.0

    return {
        "name": "observer_arc_v0",
        "seed": 1729,
        "max_actions": max_actions,
        "click_probe_limit": 32,
        "repeat_window": 12,
        "planner_depth": 48,
        "planner_beam_width": 8,
        "planner_branch_limit": 18,
        "planner_max_nodes": 2200,
        "planner_max_seconds": 0.30,
        "planner_return_best_nonterminal": True,
        "simple_action_prior": {
            "1": 0.58,
            "2": 0.58,
            "3": 0.56,
            "4": 0.56,
            "5": 0.46,
            "7": 0.22,
        },
        "weights": {
            "base": 0.12,
            "novelty": 0.31,
            "value": 0.34,
            "coherence": 0.18,
            "grounding": 0.16 + click_bias,
            "repetition": 0.29,
            "cliche": 0.22,
            "contradiction": 0.45,
        },
        "training": {
            "method": "deterministic metadata-derived priors only",
            "environment_count": len(metadata_files),
            "tag_counts": dict(sorted(tag_counts.items())),
            "mean_public_baseline_actions": round(mean_baseline, 3),
            "p75_public_total_baseline_actions": p75_total,
            "uses_public_action_replays": False,
            "uses_external_models": False,
            "uses_internet_at_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment-dir",
        default="../environment_files",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="observer_arc/model_config.json",
        type=Path,
    )
    args = parser.parse_args()
    config = build_config(args.environment_dir)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
