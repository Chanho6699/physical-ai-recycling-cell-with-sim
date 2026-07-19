"""V3 End-to-End Training Pipeline -- Dataset Validation gate (see this
task's chat report). Runs automatically after benchmark/build_v3_dataset.py
(or standalone against any v3-style dataset root) and exits non-zero
if the collected dataset doesn't clear the same thresholds
benchmark/collect_v3_recovery_smoke.py's smoke50 validation already
established as "good enough to train on" -- benchmark/train_v3.py calls
this FIRST and refuses to launch training if it fails.

Reuses benchmark/analyze_v2_approach_coverage.py's load_manifest()/
load_frames()/build_episode_records()/basic_stats()/
distance_band_analysis()/trajectory_diversity_analysis() (parameterized
by dataset_root) and benchmark/analyze_v3_recovery_smoke.py's
ee_initial_diversity()/recovery_generation()/_true_stabilization_frame_count()
UNCHANGED -- only the manifest is now a MIX of "normal" and "recovery"
collection_mode rows (see build_v3_dataset.py), so this script filters
by collection_mode before handing each subset to the metric it's
meaningful for (EE-init/recovery metrics only apply to "recovery" rows;
distance/trajectory/basic stats apply to the full merged set).

Run:
  .venv-vla/bin/python -m benchmark.validate_v3_dataset --root datasets/recycling_v3_dataset
"""

import argparse
import json
import sys
from pathlib import Path

from benchmark.analyze_v2_approach_coverage import (
    basic_stats,
    build_episode_records,
    distance_band_analysis,
    load_frames,
    load_manifest,
    trajectory_diversity_analysis,
)
from benchmark.analyze_v3_recovery_smoke import (
    _true_stabilization_frame_count,
    ee_initial_diversity,
    recovery_generation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "results" / "dataset_analysis" / "v3_dataset_validation.json"

# Same gate thresholds smoke50 was judged against (see benchmark/
# collect_v3_recovery_smoke.py's chat report, section 6) -- reused
# unchanged as the "safe to train on" bar for any v3-style dataset,
# not just the 50-episode smoke run.
THRESHOLDS = {
    "min_unique_ee_initial_positions": 15,
    "min_recovery_rate_given_perturbation": 0.80,
    "min_expert_pick_success_rate": 0.90,
    "min_fraction_within_0_05m_vs_v2": 0.0671,  # must EXCEED v2's own baseline, not just be nonzero
    "min_near_frames_before_close_vs_v2": 1.01,
    "max_trajectory_near_duplicate_fraction": 0.9938,  # must be BELOW v2's own baseline
}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate(dataset_root: Path) -> dict:
    manifest = load_manifest(dataset_root)
    frames = load_frames(dataset_root)
    records = build_episode_records(manifest, frames)

    b = basic_stats(manifest, frames, records)
    d = distance_band_analysis(records)
    t = trajectory_diversity_analysis(records)

    recovery_manifest = [e for e in manifest if e.get("collection_mode") == "recovery"]
    normal_manifest = [e for e in manifest if e.get("collection_mode") == "normal"]

    checks = []
    if recovery_manifest:
        eid = ee_initial_diversity(recovery_manifest)
        rg = recovery_generation(recovery_manifest, frames)
        checks.append(("unique_ee_initial_positions", eid["unique_ee_initial_positions_rounded_1e-4m"], ">=", THRESHOLDS["min_unique_ee_initial_positions"]))
        if rg["recovery_rate_given_perturbation"] is not None:
            checks.append(("recovery_rate_given_perturbation", rg["recovery_rate_given_perturbation"], ">=", THRESHOLDS["min_recovery_rate_given_perturbation"]))
    else:
        eid, rg = None, None

    run_summary_path = dataset_root / "run_summary.json"
    expert_pick_success_rate = None
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text())
        total_saved = run_summary.get("total_saved_episodes") or run_summary.get("saved_episodes")
        total_attempts = run_summary.get("total_attempts") or run_summary.get("attempted_episodes")
        if total_saved is not None and total_attempts:
            expert_pick_success_rate = total_saved / total_attempts
            checks.append(("expert_pick_success_rate", expert_pick_success_rate, ">=", THRESHOLDS["min_expert_pick_success_rate"]))

    checks.append(("fraction_within_0_05m", d["fraction_within_0_05m"], ">=", THRESHOLDS["min_fraction_within_0_05m_vs_v2"]))
    checks.append(("near_frames_in_last_5_before_close_mean", d["near_frames_in_last_5_before_close_mean"], ">=", THRESHOLDS["min_near_frames_before_close_vs_v2"]))
    checks.append(("trajectory_near_duplicate_fraction", t["fraction_episodes_with_near_duplicate_trajectory"], "<=", THRESHOLDS["max_trajectory_near_duplicate_fraction"]))

    passed_checks = []
    for name, value, op, threshold in checks:
        ok = (value >= threshold) if op == ">=" else (value <= threshold)
        passed_checks.append({"check": name, "value": value, "op": op, "threshold": threshold, "passed": ok})

    all_passed = all(c["passed"] for c in passed_checks)

    return {
        "dataset_root": str(dataset_root),
        "num_episodes_total": b["num_episodes"],
        "num_normal_episodes": len(normal_manifest),
        "num_recovery_episodes": len(recovery_manifest),
        "basic_stats": b,
        "distance_band_analysis": d,
        "trajectory_diversity": t,
        "ee_initial_diversity": eid,
        "recovery_generation": rg,
        "checks": passed_checks,
        "all_checks_passed": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSON.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    result = validate(resolve(args.root))

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"=== V3 dataset validation: {result['dataset_root']} ===")
    print(f"total episodes: {result['num_episodes_total']} (normal={result['num_normal_episodes']}, recovery={result['num_recovery_episodes']})")
    for c in result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c['value']:.4f} {c['op']} {c['threshold']}")
    print(f"\nResult JSON: {output_path}")

    if result["all_checks_passed"]:
        print("\n=== VALIDATION PASSED -- safe to proceed to training ===")
        sys.exit(0)
    else:
        print("\n=== VALIDATION FAILED -- training must NOT proceed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
