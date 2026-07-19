"""V3 End-to-End Training Pipeline -- Best Checkpoint Selection (see this
task's chat report). Reads benchmark.train_v3's rollout_eval.csv (one row
per evaluated checkpoint step, written by benchmark/evaluate_v3_checkpoint.py)
and recommends the best one.

Selection criterion (this task's chat report, section 7 -- simple,
stated up front, no hidden weighting):
  1. Highest overall_success_rate.
  2. Tie-break: highest pick_rate.
  3. Tie-break: highest approach_5cm_rate.
  4. Tie-break: highest place_rate.
  5. Tie-break: LOWER mean_min_distance (closer average approach wins).
  6. Tie-break: EARLIER step (a tie on every performance metric means no
     benefit from the extra training -- prefer the cheaper, less-
     possibly-overfit checkpoint, not the more-trained one).

Run:
  .venv-vla/bin/python -m benchmark.select_best_v3_checkpoint --csv results/v3_pipeline/rollout_eval.csv
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(csv_path: Path) -> list:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for field in ("step", "num_episodes"):
            if row.get(field):
                row[field] = int(float(row[field]))
        for field in (
            "approach_10cm_rate", "approach_5cm_rate", "close_rate", "close_distance_mean",
            "pick_rate", "place_rate", "overall_success_rate", "mean_min_distance", "mean_rollout_length",
        ):
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
    return rows


def select_best(rows: list) -> dict:
    # Earlier step must win ties, so negate step for a plain max() (all
    # other fields are "bigger is better" already; mean_min_distance is
    # negated too since LOWER is better there).
    return max(
        rows,
        key=lambda r: (
            r["overall_success_rate"], r["pick_rate"], r["approach_5cm_rate"], r["place_rate"],
            -r["mean_min_distance"], -(r["step"] or 0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="results/v3_pipeline/rollout_eval.csv")
    parser.add_argument("--output", type=str, default="results/v3_pipeline/best_checkpoint.json")
    args = parser.parse_args()

    csv_path = resolve(args.csv)
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path} -- has benchmark.train_v3 run any evaluation rounds yet?")

    print(f"=== {len(rows)} evaluated checkpoints in {csv_path} ===")
    for r in sorted(rows, key=lambda r: r["step"] or 0):
        print(
            f"  step={r['step']:>6} label={r['label']:20s} "
            f"approach10cm={r['approach_10cm_rate']:.2%} approach5cm={r['approach_5cm_rate']:.2%} "
            f"close={r['close_rate']:.2%} pick={r['pick_rate']:.2%} place={r['place_rate']:.2%} "
            f"success={r['overall_success_rate']:.2%} mean_min_dist={r['mean_min_distance']:.4f}"
        )

    best = select_best(rows)
    result = {
        "selection_criterion": (
            "max(overall_success_rate, then pick_rate, then approach_5cm_rate, then place_rate, "
            "then LOWER mean_min_distance, then EARLIER step)"
        ),
        "best_checkpoint": best,
        "num_checkpoints_compared": len(rows),
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n=== BEST CHECKPOINT: step={best['step']} (label={best['label']}) ===")
    print(f"    model_id_or_path={best['model_id_or_path']}")
    print(f"    success={best['overall_success_rate']:.2%} pick={best['pick_rate']:.2%} place={best['place_rate']:.2%}")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
