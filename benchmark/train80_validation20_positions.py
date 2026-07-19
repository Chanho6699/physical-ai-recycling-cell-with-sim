"""Coordinate generation for the train80/validation20 LeRobot dataset
expansion (see this task's chat report). Pure position/seed bookkeeping
only -- no simulator, no policy, no dataset writes here, so the full
train/validation coordinate lists and their overlap/min-distance checks
can be printed and verified BEFORE any collection starts (per this
task's explicit requirement).

Design (see report for full rationale):
  - Train: a 4x4 grid of BASE anchors (TRAIN_X x TRAIN_Y), each within
    the expert operating envelope already validated end-to-end in the
    prior task's benchmark/evaluate_expert_policy_benchmark.py run
    (x in [0.27, 0.42], y in [-0.18, 0.18], 36/36 success there).
  - Validation: a 5-point "cross" of positions interpolated at the
    MIDPOINTS between adjacent train grid lines (never coincides with
    any train grid line in either axis) -- not the same coordinates
    with only a different seed.
  - Both splits get independent, disjoint, explicitly recorded seed
    pools, and a smaller jitter radius than
    collect_recycling_dataset.JITTER_RADIUS_M (0.03) -- 0.015 -- so the
    ~0.065m nominal train/validation grid-line spacing survives jitter
    with comfortable margin (see verify_min_distance()'s printed
    result for the actual realized minimum).
"""

import math
import random
from typing import List

OBJECT_Z = 0.05  # matches collect_recycling_dataset.DEFAULT_POSITIONS' z

TRAIN_X = [0.27, 0.32, 0.37, 0.42]
TRAIN_Y = [-0.18, -0.06, 0.06, 0.18]

# Midpoints of TRAIN_X / TRAIN_Y -- by construction never equal to any
# TRAIN_X/TRAIN_Y value.
VALIDATION_X = [(TRAIN_X[i] + TRAIN_X[i + 1]) / 2 for i in range(len(TRAIN_X) - 1)]  # [0.295, 0.345, 0.395]
VALIDATION_Y = [(TRAIN_Y[i] + TRAIN_Y[i + 1]) / 2 for i in range(len(TRAIN_Y) - 1)]  # [-0.12, 0.0, 0.12]

# 5-point cross through the interpolated-grid center -- spans both axes
# (west/east on the x midline, near/far on the y midline) rather than
# clustering validation in one corner.
_VALIDATION_CENTER_X = VALIDATION_X[1]
_VALIDATION_CENTER_Y = VALIDATION_Y[1]
VALIDATION_ANCHORS = {
    "val_center": (_VALIDATION_CENTER_X, _VALIDATION_CENTER_Y),
    "val_west": (VALIDATION_X[0], _VALIDATION_CENTER_Y),
    "val_east": (VALIDATION_X[2], _VALIDATION_CENTER_Y),
    "val_near": (_VALIDATION_CENTER_X, VALIDATION_Y[0]),
    "val_far": (_VALIDATION_CENTER_X, VALIDATION_Y[2]),
}

TRAIN_SEEDS = [0, 1, 2, 3, 4]  # 16 anchors x 5 seeds = 80
VALIDATION_SEEDS = [100, 101, 102, 103]  # 5 anchors x 4 seeds = 20 -- disjoint from TRAIN_SEEDS by construction

TRAIN_JITTER_RADIUS_M = 0.015
VALIDATION_JITTER_RADIUS_M = 0.015

TRAIN_POSITION_SEED_BASE = 70000
VALIDATION_POSITION_SEED_BASE = 80000


def build_train_anchors() -> dict:
    return {
        f"train_x{xi}_y{yi}": (x, y)
        for xi, x in enumerate(TRAIN_X)
        for yi, y in enumerate(TRAIN_Y)
    }


def _jitter_seed_for(base: int, anchor_index: int, seed: int) -> int:
    return base + anchor_index * 1000 + seed


def _jitter_xy(x: float, y: float, z: float, rng: random.Random, radius_m: float) -> list:
    """Independent uniform x/y jitter in [-radius_m, +radius_m] (z
    untouched) -- same convention as
    collect_recycling_dataset.jitter_position(), just parametrized by
    radius so this module's own (smaller) TRAIN/VALIDATION_JITTER_RADIUS_M
    can be used instead of that function's module-level JITTER_RADIUS_M
    constant."""
    return [x + rng.uniform(-radius_m, radius_m), y + rng.uniform(-radius_m, radius_m), z]


def build_positions(anchors: dict, seeds: List[int], jitter_radius_m: float, seed_base: int) -> List[dict]:
    """Returns one dict per (anchor, seed) episode: anchor_name, anchor_xy,
    seed (the actual jitter RNG seed, unique and recorded), position
    (post-jitter [x, y, z])."""
    episodes = []
    for anchor_index, (anchor_name, (x, y)) in enumerate(anchors.items()):
        for seed in seeds:
            jitter_seed = _jitter_seed_for(seed_base, anchor_index, seed)
            rng = random.Random(jitter_seed)
            position = _jitter_xy(x, y, OBJECT_Z, rng, jitter_radius_m)
            episodes.append({
                "anchor_name": anchor_name,
                "anchor_xy": [x, y],
                "seed": jitter_seed,
                "position": position,
            })
    return episodes


def build_train_positions() -> List[dict]:
    return build_positions(build_train_anchors(), TRAIN_SEEDS, TRAIN_JITTER_RADIUS_M, TRAIN_POSITION_SEED_BASE)


def build_validation_positions() -> List[dict]:
    return build_positions(VALIDATION_ANCHORS, VALIDATION_SEEDS, VALIDATION_JITTER_RADIUS_M, VALIDATION_POSITION_SEED_BASE)


def _distance_2d(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def verify_split(train_positions: List[dict], validation_positions: List[dict], round_ndigits: int = 4) -> dict:
    """Verifies (a) zero exact-coordinate duplicates between the two
    splits (rounded to round_ndigits, i.e. sub-0.1mm collisions), (b)
    zero duplicate seeds between the two splits, (c) the realized
    minimum pairwise xy distance between any train position and any
    validation position (post-jitter, the actual numbers that will be
    collected -- not just the nominal anchor grid spacing)."""
    train_xy_rounded = {tuple(round(v, round_ndigits) for v in p["position"][:2]) for p in train_positions}
    validation_xy_rounded = {tuple(round(v, round_ndigits) for v in p["position"][:2]) for p in validation_positions}
    exact_duplicates = train_xy_rounded & validation_xy_rounded

    train_seeds = {p["seed"] for p in train_positions}
    validation_seeds = {p["seed"] for p in validation_positions}
    duplicate_seeds = train_seeds & validation_seeds

    min_distance = min(
        _distance_2d(tp["position"], vp["position"])
        for tp in train_positions
        for vp in validation_positions
    )

    return {
        "num_train": len(train_positions),
        "num_validation": len(validation_positions),
        "exact_coordinate_duplicates": sorted(exact_duplicates),
        "num_exact_coordinate_duplicates": len(exact_duplicates),
        "duplicate_seeds": sorted(duplicate_seeds),
        "num_duplicate_seeds": len(duplicate_seeds),
        "min_train_validation_distance_m": min_distance,
    }


if __name__ == "__main__":
    train_positions = build_train_positions()
    validation_positions = build_validation_positions()

    print(f"=== Train anchors ({len(build_train_anchors())}) ===")
    for name, (x, y) in build_train_anchors().items():
        print(f"  {name}: x={x}, y={y}")
    print(f"\n=== Validation anchors ({len(VALIDATION_ANCHORS)}) ===")
    for name, (x, y) in VALIDATION_ANCHORS.items():
        print(f"  {name}: x={x}, y={y}")

    print(f"\n=== Train positions ({len(train_positions)}) ===")
    for p in train_positions:
        print(f"  {p['anchor_name']:16s} seed={p['seed']:6d} position={[round(v, 4) for v in p['position']]}")
    print(f"\n=== Validation positions ({len(validation_positions)}) ===")
    for p in validation_positions:
        print(f"  {p['anchor_name']:16s} seed={p['seed']:6d} position={[round(v, 4) for v in p['position']]}")

    report = verify_split(train_positions, validation_positions)
    print("\n=== Split verification ===")
    print(json_dump := __import__("json").dumps(report, indent=2))

    assert report["num_exact_coordinate_duplicates"] == 0, "Train/validation coordinates overlap!"
    assert report["num_duplicate_seeds"] == 0, "Train/validation seeds overlap!"
    assert report["num_train"] == 80, f"Expected 80 train positions, got {report['num_train']}"
    assert report["num_validation"] == 20, f"Expected 20 validation positions, got {report['num_validation']}"
    print("\nALL CHECKS PASSED -- 0 coordinate duplicates, 0 seed duplicates, counts correct.")
