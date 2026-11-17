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
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from saga_pose.evaluation.bootstrap import paired_sequence_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path, help="YAML with a 'comparisons' list")
    parser.add_argument("--real-dir", type=Path, default=Path("outputs/real"))
    parser.add_argument(
        "--runs-dir", type=Path, nargs="+", required=True, help="dirs holding synthetic reports"
    )
    parser.add_argument("--condition", default="primary")
    parser.add_argument("--output", type=Path, default=Path("outputs/real/analysis.json"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _synthetic_natural(method: str, runs_dirs: list[Path]) -> tuple[str, float, float] | None:
    """Pooled natural-validation means, or None for methods without synthetic runs."""
    runs_dir = next(
        (d for d in runs_dirs if (d / f"{method}-{OBJECTS[0]}" / "report.json").is_file()), None
    )
    if runs_dir is None:
        return None  # e.g. the learning-free geometric baseline
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


def _seed_summary(real_dir: Path, condition: str) -> dict[str, Any]:
    """Overall sequence-mean ARE/ATE per seed for methods trained with extra seeds."""
    seeded = sorted(
        {
            path.name.rsplit("-", maxsplit=2)[0]
            for path in real_dir.glob("*-s25[0-9]-cassini")
            if (path / "real_report.json").is_file()
        }
    )
    summary: dict[str, Any] = {}
    for base in seeded:
        variants = [base] + [
            f"{base}-s{seed}"
            for seed in (250, 251)
            if (real_dir / f"{base}-s{seed}-cassini").is_dir()
        ]
        rows = []
        for variant in variants:
            try:
                merged = _merge(variant, real_dir, condition)
            except FileNotFoundError:
                print(f"warning: {variant} lacks a real report for every object; omitted")
                continue
            rows.append(
                {
                    "variant": variant,
                    "sequence_mean_are_deg": float(
                        np.mean(list(merged["sequence_mean_are_deg"].values()))
                    ),
                    "sequence_mean_ate_m": float(
                        np.mean(list(merged["sequence_mean_ate_m"].values()))
                    ),
                }
            )
        are = np.array([row["sequence_mean_are_deg"] for row in rows])
        ate = np.array([row["sequence_mean_ate_m"] for row in rows])
        summary[base] = {
            "seeds": rows,
            "are_mean_deg": float(are.mean()),
            "are_sd_deg": float(are.std(ddof=1)) if are.size > 1 else None,
            "ate_mean_m": float(ate.mean()),
            "ate_sd_m": float(ate.std(ddof=1)) if ate.size > 1 else None,
        }
    return summary


def _interventions(real_dir: Path) -> dict[str, Any]:
    """Descriptive deltas (condition minus primary) per method, pooled over objects."""
    methods = sorted({path.name.removesuffix("-cassini") for path in real_dir.glob("*-cassini")})
    result: dict[str, Any] = {}
    for method in methods:
        reports = [real_dir / f"{method}-{o}" / "real_report.json" for o in OBJECTS]
        if not all(path.is_file() for path in reports):
            continue
        conditions = set(_load(reports[0])["results"]) - {"primary", "primary_ransac"}
        if not conditions:
            continue
        base = _merge(method, real_dir, "primary")
        base_are = float(np.mean(list(base["sequence_mean_are_deg"].values())))
        base_ate = float(np.mean(list(base["sequence_mean_ate_m"].values())))
        result[method] = {}
        for condition in sorted(conditions):
            merged = _merge(method, real_dir, condition)
            result[method][condition] = {
                "delta_are_deg": float(np.mean(list(merged["sequence_mean_are_deg"].values())))
                - base_are,
                "delta_ate_m": float(np.mean(list(merged["sequence_mean_ate_m"].values())))
                - base_ate,
            }
    return result


def main() -> int:
    args = parse_args()
    spec = yaml.safe_load(args.comparisons.read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "condition": args.condition,
        "provenance": {
            "freeze_manifest_sha256": (
                _sha256(args.real_dir / "freeze_manifest.json")
                if (args.real_dir / "freeze_manifest.json").is_file()
                else None
            ),
            "comparisons_sha256": _sha256(args.comparisons),
        },
        "gaps_note": (
            "gaps compare real errors under this condition's solver with EPnP synthetic "
            "validation errors"
        ),
        "gaps": {},
        "comparisons": [],
    }
    names = {name for pair in spec["comparisons"] for name in (pair["candidate"], pair["baseline"])}
    for name in sorted(names):
        real = _merge(name, args.real_dir, args.condition)
        synthetic = _synthetic_natural(name, args.runs_dir)
        if synthetic is None:
            continue
        protocol, synthetic_are, synthetic_ate = synthetic
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
    # Holm correction across the explicitly pre-registered family (list in the YAML).
    members = set(spec.get("holm_family", []))
    family = [
        entry
        for entry in output["comparisons"]
        if f"{entry['candidate']}|{entry['baseline']}" in members
    ]
    if len(family) != len(members):
        raise SystemExit("holm_family lists comparisons absent from the comparisons list")
    for metric in ("are_deg", "ate_m"):
        adjusted = _holm([entry["overall"][metric]["p_two_sided"] for entry in family])
        for entry, value in zip(family, adjusted, strict=True):
            entry["overall"][metric]["p_holm_tss_family"] = value
    for entry in output["comparisons"]:
        is_primary = bool(
            primary
            and entry["candidate"] == primary["candidate"]
            and entry["baseline"] == primary["baseline"]
        )
        entry["family"] = (
            "primary" if is_primary else "tss_holm" if entry in family else "exploratory"
        )
    # Pre-specified descriptive interventions: change in overall sequence-mean ARE/ATE
    # under clipping shuffle/zero, gate bypass, and the magnitude event encoding.
    output["interventions"] = _interventions(args.real_dir)
    # Seed variation, reported separately from the sequence bootstrap (AGENTS.md).
    output["seeds"] = _seed_summary(args.real_dir, args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"analysis: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
