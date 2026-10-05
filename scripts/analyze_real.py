"""Prewritten G6 analysis: paired sequence bootstrap and generalization gaps.

Inputs are ``outputs/real/<run>/real_report.json`` files from
``evaluate_real.py`` and the matching synthetic ``report.json`` files. For each
declared comparison (candidate vs baseline) it reports the paired,
sequence-level bootstrap of sequence-mean ARE and ATE differences overall and on
the author-defined harsh and low-motion slices. Generalization gaps are reported
with both endpoints (synthetic-validation frame mean and real frame mean) so a
small gap cannot hide poor synthetic competence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from saga_pose.evaluation.bootstrap import paired_sequence_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path, help="YAML with a 'comparisons' list")
    parser.add_argument("--real-dir", type=Path, default=Path("outputs/real"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--condition", default="primary")
    parser.add_argument("--output", type=Path, default=Path("outputs/real/analysis.json"))
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    result = paired_sequence_bootstrap(candidate, baseline)
    better = result.probability_candidate_better
    return {
        "mean_difference": result.observed_mean_difference,
        "ci95": list(result.confidence_interval),
        "p_candidate_better": better,
        # Two-sided bootstrap p-value for "no paired difference".
        "p_two_sided": min(1.0, 2.0 * min(better, 1.0 - better)),
        "sequences": result.sequence_count,
    }


def _holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _is_tss(method: str) -> bool:
    return "-tss" in method


OBJECTS = ("cassini", "satty", "soho")


def _merge(method: str, real_dir: Path, condition: str) -> dict[str, Any]:
    """Pool the three per-object runs of one method into 24-sequence dictionaries."""
    merged: dict[str, Any] = {
        "sequence_mean_are_deg": {},
        "sequence_mean_ate_m": {},
        "slices": {"harsh": {}, "low-motion": {}},
        "frames": 0,
        "are_sum": 0.0,
        "ate_sum": 0.0,
    }
    for slice_name in merged["slices"]:
        merged["slices"][slice_name] = {"sequence_mean_are_deg": {}, "sequence_mean_ate_m": {}}
    for object_name in OBJECTS:
        analysis = _load(real_dir / f"{method}-{object_name}" / "real_report.json")["results"][
            condition
        ]["analysis"]
        for key in ("sequence_mean_are_deg", "sequence_mean_ate_m"):
            merged[key].update(analysis[key])
            for slice_name in merged["slices"]:
                merged["slices"][slice_name][key].update(analysis["slices"][slice_name][key])
        overall = analysis["overall_frames"]
        merged["frames"] += overall["frames"]
        merged["are_sum"] += overall["are_degrees_mean"] * overall["frames"]
        merged["ate_sum"] += overall["ate_m_mean"] * overall["frames"]
    merged["are_degrees_mean"] = merged["are_sum"] / merged["frames"]
    merged["ate_m_mean"] = merged["ate_sum"] / merged["frames"]
    return merged


def _synthetic_natural(method: str, runs_dir: Path) -> tuple[str, float, float]:
    are_sum = ate_sum = 0.0
    count = 0
    protocol = ""
    for object_name in OBJECTS:
        report = _load(runs_dir / f"{method}-{object_name}" / "report.json")
        protocol = report["protocol"]
        pose = report["outcome"]["evaluations"]["validation_natural"]["pose"]
        frames = pose["frame_count"]
        are_sum += pose["all_frames"]["are_degrees_mean"] * frames
        ate_sum += pose["all_frames"]["absolute_translation_error_m_mean"] * frames
        count += frames
    return protocol, are_sum / count, ate_sum / count


def main() -> int:
    args = parse_args()
    spec = yaml.safe_load(args.comparisons.read_text(encoding="utf-8"))
    output: dict[str, Any] = {"condition": args.condition, "gaps": {}, "comparisons": []}
    names = {name for pair in spec["comparisons"] for name in (pair["candidate"], pair["baseline"])}
    for name in sorted(names):
        real = _merge(name, args.real_dir, args.condition)
        protocol, synthetic_are, synthetic_ate = _synthetic_natural(name, args.runs_dir)
        output["gaps"][name] = {
            "training_protocol": protocol,
            "synthetic_are_mean_deg": synthetic_are,
            "real_are_mean_deg": real["are_degrees_mean"],
            "gap_are_deg": real["are_degrees_mean"] - synthetic_are,
            "synthetic_ate_mean_m": synthetic_ate,
            "real_ate_mean_m": real["ate_m_mean"],
            "gap_ate_m": real["ate_m_mean"] - synthetic_ate,
        }
    for pair in spec["comparisons"]:
        candidate = _merge(pair["candidate"], args.real_dir, args.condition)
        baseline = _merge(pair["baseline"], args.real_dir, args.condition)
        entry: dict[str, Any] = {
            "candidate": pair["candidate"],
            "baseline": pair["baseline"],
            "question": pair.get("question", ""),
            "overall": {
                "are_deg": _bootstrap(
                    candidate["sequence_mean_are_deg"], baseline["sequence_mean_are_deg"]
                ),
                "ate_m": _bootstrap(
                    candidate["sequence_mean_ate_m"], baseline["sequence_mean_ate_m"]
                ),
            },
        }
        for slice_name in ("harsh", "low-motion"):
            entry[slice_name] = {
                "are_deg": _bootstrap(
                    candidate["slices"][slice_name]["sequence_mean_are_deg"],
                    baseline["slices"][slice_name]["sequence_mean_are_deg"],
                ),
                "ate_m": _bootstrap(
                    candidate["slices"][slice_name]["sequence_mean_ate_m"],
                    baseline["slices"][slice_name]["sequence_mean_ate_m"],
                ),
            }
        output["comparisons"].append(entry)
    # Primary comparison (pre-registered, co-primary ARE and ATE).
    primary = spec.get("primary")
    if primary:
        candidate = _merge(primary["candidate"], args.real_dir, args.condition)
        baseline = _merge(primary["baseline"], args.real_dir, args.condition)
        output["primary"] = {
            **primary,
            "are_deg": _bootstrap(
                candidate["sequence_mean_are_deg"], baseline["sequence_mean_are_deg"]
            ),
            "ate_m": _bootstrap(candidate["sequence_mean_ate_m"], baseline["sequence_mean_ate_m"]),
        }
    # Holm correction across the TSS secondary family, per metric (overall).
    family = [
        entry
        for entry in output["comparisons"]
        if _is_tss(entry["candidate"])
        and _is_tss(entry["baseline"])
        and not (
            primary
            and entry["candidate"] == primary["candidate"]
            and entry["baseline"] == primary["baseline"]
        )
    ]
    for metric in ("are_deg", "ate_m"):
        adjusted = _holm([entry["overall"][metric]["p_two_sided"] for entry in family])
        for entry, value in zip(family, adjusted, strict=True):
            entry["overall"][metric]["p_holm_tss_family"] = value
    for entry in output["comparisons"]:
        entry["family"] = "tss_holm" if entry in family else "exploratory"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"analysis: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
