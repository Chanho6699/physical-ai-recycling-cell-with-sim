"""Fixed, shared position/seed lists for the 5-way checkpoint rollout
comparison (A zero-shot / B train20-200step / C train80-500step /
D train80-1000step / E train80-2000step -- see this task's chat
report). Every model is evaluated against the EXACT SAME 40 episodes
(20 train-distribution + 20 validation-distribution) so results are
directly comparable.

Train-distribution: a representative 10-anchor SUBSET of
benchmark.train80_validation20_positions.build_train_anchors()'s 16-cell
grid (not all 16, to keep total eval episode count bounded), x 2 NEW
seeds each = 20 episodes. "New" means disjoint from every seed already
consumed by data collection (TRAIN_SEEDS=[0..4] in
train80_validation20_positions.py) or by this task's own pre-collection
dry-run -- so no episode here replays an actual training trajectory
(same anchor, but jitter lands on a different exact xy each time a
different seed is used).

Validation-distribution: the SAME 5 interpolated-midpoint anchors
already used for datasets/recycling_validation20_v1
(val_center/west/east/near/far), x 4 NEW seeds (disjoint from
VALIDATION_SEEDS=[100..103] used to build that dataset).
"""

from typing import List

from benchmark.train80_validation20_positions import (
    OBJECT_Z,
    VALIDATION_ANCHORS,
    _jitter_xy,
    build_train_anchors,
)
import random

TRAIN_JITTER_RADIUS_M = 0.015
VALIDATION_JITTER_RADIUS_M = 0.015

# 10 of the 16 train anchors -- spans all 4 x-levels and all 4 y-levels
# at least once each (not just a corner cluster).
TRAIN_EVAL_ANCHOR_NAMES = [
    "train_x0_y0", "train_x0_y2", "train_x1_y1", "train_x1_y3", "train_x2_y0",
    "train_x2_y2", "train_x3_y1", "train_x3_y3", "train_x1_y0", "train_x2_y3",
]
TRAIN_EVAL_SEEDS = [200, 201]  # 10 anchors x 2 seeds = 20, disjoint from TRAIN_SEEDS=[0..4]
TRAIN_EVAL_SEED_BASE = 200000

VALIDATION_EVAL_SEEDS = [300, 301, 302, 303]  # 5 anchors x 4 seeds = 20, disjoint from VALIDATION_SEEDS=[100..103]
VALIDATION_EVAL_SEED_BASE = 300000


def _build(anchors: dict, seeds: List[int], jitter_radius_m: float, seed_base: int) -> List[dict]:
    episodes = []
    for anchor_index, (anchor_name, (x, y)) in enumerate(anchors.items()):
        for seed in seeds:
            jitter_seed = seed_base + anchor_index * 1000 + seed
            rng = random.Random(jitter_seed)
            position = _jitter_xy(x, y, OBJECT_Z, rng, jitter_radius_m)
            episodes.append({"anchor_name": anchor_name, "anchor_xy": [x, y], "seed": jitter_seed, "position": position})
    return episodes


def build_train_eval_positions() -> List[dict]:
    all_train_anchors = build_train_anchors()
    anchors = {name: all_train_anchors[name] for name in TRAIN_EVAL_ANCHOR_NAMES}
    return _build(anchors, TRAIN_EVAL_SEEDS, TRAIN_JITTER_RADIUS_M, TRAIN_EVAL_SEED_BASE)


def build_validation_eval_positions() -> List[dict]:
    return _build(VALIDATION_ANCHORS, VALIDATION_EVAL_SEEDS, VALIDATION_JITTER_RADIUS_M, VALIDATION_EVAL_SEED_BASE)


if __name__ == "__main__":
    from benchmark.train80_validation20_positions import TRAIN_SEEDS, VALIDATION_SEEDS

    train_eval = build_train_eval_positions()
    validation_eval = build_validation_eval_positions()

    train_eval_seeds = {p["seed"] for p in train_eval}
    validation_eval_seeds = {p["seed"] for p in validation_eval}
    # These are RNG seeds in a different numeric range (TRAIN_EVAL_SEED_BASE=200000 /
    # VALIDATION_EVAL_SEED_BASE=300000) than the collection seeds
    # (TRAIN_POSITION_SEED_BASE=70000 / VALIDATION_POSITION_SEED_BASE=80000), so
    # disjointness is true by construction; asserted here defensively.
    from benchmark.train80_validation20_positions import build_train_positions, build_validation_positions
    collection_seeds = {p["seed"] for p in build_train_positions()} | {p["seed"] for p in build_validation_positions()}

    print(f"train-distribution eval episodes: {len(train_eval)}")
    print(f"validation-distribution eval episodes: {len(validation_eval)}")
    print(f"train_eval seeds disjoint from collection seeds: {train_eval_seeds.isdisjoint(collection_seeds)}")
    print(f"validation_eval seeds disjoint from collection seeds: {validation_eval_seeds.isdisjoint(collection_seeds)}")
    print(f"train_eval / validation_eval seeds mutually disjoint: {train_eval_seeds.isdisjoint(validation_eval_seeds)}")
    for p in train_eval:
        print(" train:", p)
    for p in validation_eval:
        print(" validation:", p)
