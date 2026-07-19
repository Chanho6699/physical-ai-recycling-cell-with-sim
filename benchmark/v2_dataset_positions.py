"""v2 dataset coordinate/split design (see this task's chat report):
object AND bin positions both vary, so a re-collected checkpoint has a
genuine incentive to condition on the object-bin RELATIVE position
instead of memorizing a fixed release timing/trajectory (see the prior
task's bin-position intervention test -- confirmed that shortcut is
exactly what train80's single, never-varied bin position produced).

Object positions: reuse train80_validation20_positions.py's own 16-cell
4x4 grid (build_train_anchors()) unchanged -- this range is already
validated end-to-end (36/36 in benchmark/evaluate_expert_policy_benchmark.py,
80/80 in train80 collection) as within the expert's safe operating
envelope; not re-derived here to avoid a second, possibly-inconsistent
definition of "safe workspace".

Bin positions: center (train80's own single fixed bin) plus the exact
4 shifted positions (+-0.05m in x/y) already used in the prior task's
diagnose_bin_position_intervention.py -- those 4 shifts already ran
real rollouts without any workspace/IK/collision problem surfacing, so
they are re-used here rather than picked fresh.

Combination-pool split (16 objects x 5 bins = 80 total pairs): 20 pairs
are held out entirely as the VALIDATION combination pool (never
episode-collected under the TRAIN seed range), the remaining 60 form
the TRAIN combination pool -- chosen so every one of the 16 object
anchors AND all 5 bin positions appear in BOTH pools (never a novel
coordinate, only a novel PAIRING, per this task's explicit
requirement), via a deterministic round-robin (bin b's held-out objects
= object indices [4b, 4b+1, 4b+2, 4b+3] mod 16).
"""

import random
from typing import Dict, List, Tuple

from benchmark.train80_validation20_positions import OBJECT_Z, _jitter_xy, build_train_anchors

OBJECT_ANCHORS: Dict[str, Tuple[float, float]] = build_train_anchors()  # 16 anchors, unchanged from train80
OBJECT_ANCHOR_NAMES: List[str] = list(OBJECT_ANCHORS.keys())

# Center matches train80's own DEFAULT_BIN_POSITION exactly; the 4
# shifts match diagnose_bin_position_intervention.py's own
# BIN_SHIFT_CONDITIONS exactly (same axis convention: +x=front, -x=back,
# +y=left, -y=right).
BIN_SHIFT_M = 0.05
_ORIGINAL_BIN = [0.3, 0.35, 0.05]
BIN_POSITIONS: Dict[str, List[float]] = {
    "center": list(_ORIGINAL_BIN),
    "front": [_ORIGINAL_BIN[0] + BIN_SHIFT_M, _ORIGINAL_BIN[1], _ORIGINAL_BIN[2]],
    "back": [_ORIGINAL_BIN[0] - BIN_SHIFT_M, _ORIGINAL_BIN[1], _ORIGINAL_BIN[2]],
    "left": [_ORIGINAL_BIN[0], _ORIGINAL_BIN[1] + BIN_SHIFT_M, _ORIGINAL_BIN[2]],
    "right": [_ORIGINAL_BIN[0], _ORIGINAL_BIN[1] - BIN_SHIFT_M, _ORIGINAL_BIN[2]],
}
BIN_POSITION_NAMES: List[str] = list(BIN_POSITIONS.keys())

OBJECT_JITTER_RADIUS_M = 0.015  # matches train80_validation20_positions.py's own TRAIN_JITTER_RADIUS_M
# Bin positions are deliberately NOT jittered -- they are 5 discrete,
# clearly-distinguishable named conditions by design (matching the
# intervention test this reuses), not a continuous grid; jittering them
# would blur exactly the discrete before/after comparison this dataset
# needs to remain diagnosable.

TRAIN_SEED_BASE = 400000  # fresh range, disjoint from every seed base already used
# (train80_validation20_positions: 70000/80000; checkpoint_eval_positions: 200000/300000)
VALIDATION_SEED_BASE = 500000


def _all_combinations() -> List[Tuple[str, str]]:
    return [(obj, bin_name) for obj in OBJECT_ANCHOR_NAMES for bin_name in BIN_POSITION_NAMES]


def split_combination_pools() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Returns (train_combinations, validation_combinations) -- 60/20 of
    the 80 total (object_anchor_name, bin_name) pairs. Deterministic,
    no randomness: bin index b's 4 held-out objects are anchor indices
    [4b, 4b+1, 4b+2, 4b+3] (mod 16), so every bin holds out exactly 4
    distinct objects and (since 5*4=20 > 16) some objects are held out
    for two different bins -- by construction every object and every
    bin still appears in both pools."""
    validation = []
    for bin_index, bin_name in enumerate(BIN_POSITION_NAMES):
        for k in range(4):
            object_index = (bin_index * 4 + k) % len(OBJECT_ANCHOR_NAMES)
            validation.append((OBJECT_ANCHOR_NAMES[object_index], bin_name))
    validation_set = set(validation)
    train = [combo for combo in _all_combinations() if combo not in validation_set]
    return train, validation


def _build_episodes(combinations: List[Tuple[str, str]], episodes_per_combo: List[int], seed_base: int) -> List[dict]:
    """episodes_per_combo[i] repetitions of combinations[i], each with
    its own deterministic jitter seed (seed_base + combo_index*100 +
    repetition_index, a distinct numeric range per combo so no two
    episodes anywhere in this split can collide)."""
    assert len(combinations) == len(episodes_per_combo)
    episodes = []
    for combo_index, ((object_name, bin_name), count) in enumerate(zip(combinations, episodes_per_combo)):
        object_x, object_y = OBJECT_ANCHORS[object_name]
        bin_position = BIN_POSITIONS[bin_name]
        for repetition in range(count):
            seed = seed_base + combo_index * 100 + repetition
            rng = random.Random(seed)
            position = _jitter_xy(object_x, object_y, OBJECT_Z, rng, OBJECT_JITTER_RADIUS_M)
            episodes.append({
                "object_anchor_name": object_name,
                "bin_name": bin_name,
                "bin_position": list(bin_position),
                "seed": seed,
                "position": position,
            })
    return episodes


def build_train_v2_episodes() -> List[dict]:
    """160 episodes over the 60 train combinations: 40 combos get 3
    episodes, 20 combos get 2 episodes (40*3 + 20*2 = 160) -- the extra
    repetition is spread over the FIRST 40 combos in the (deterministic,
    sorted) train-combination list, not concentrated on any one object
    or bin (see verify_v2_split() for the realized per-object/per-bin
    balance this actually produces)."""
    train_combos, _ = split_combination_pools()
    train_combos = sorted(train_combos)
    counts = [3 if i < 40 else 2 for i in range(len(train_combos))]
    assert sum(counts) == 160, sum(counts)
    return _build_episodes(train_combos, counts, TRAIN_SEED_BASE)


def build_validation_v2_episodes() -> List[dict]:
    """40 episodes over the 20 validation combinations, exactly 2 each."""
    _, validation_combos = split_combination_pools()
    validation_combos = sorted(validation_combos)
    counts = [2] * len(validation_combos)
    assert sum(counts) == 40, sum(counts)
    return _build_episodes(validation_combos, counts, VALIDATION_SEED_BASE)


if __name__ == "__main__":
    train_combos, validation_combos = split_combination_pools()
    print(f"train combinations: {len(train_combos)}, validation combinations: {len(validation_combos)}")
    overlap = set(train_combos) & set(validation_combos)
    print(f"combination overlap: {len(overlap)}")

    train_objects = {c[0] for c in train_combos}
    validation_objects = {c[0] for c in validation_combos}
    train_bins = {c[1] for c in train_combos}
    validation_bins = {c[1] for c in validation_combos}
    print(f"objects in train: {len(train_objects)}/16, objects in validation: {len(validation_objects)}/16")
    print(f"bins in train: {sorted(train_bins)}, bins in validation: {sorted(validation_bins)}")

    train_episodes = build_train_v2_episodes()
    validation_episodes = build_validation_v2_episodes()
    print(f"train episodes: {len(train_episodes)}, validation episodes: {len(validation_episodes)}")

    train_seeds = {e["seed"] for e in train_episodes}
    validation_seeds = {e["seed"] for e in validation_episodes}
    print(f"train/validation seed overlap: {len(train_seeds & validation_seeds)}")
    print(f"train seeds unique: {len(train_seeds) == len(train_episodes)}, validation seeds unique: {len(validation_seeds) == len(validation_episodes)}")
