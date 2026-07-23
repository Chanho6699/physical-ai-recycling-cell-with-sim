"""Scripted Expert V1 regression test (see this task's chat report,
"Expert V1 동결 및 회귀 테스트"). Standalone script (plain asserts +
PASS/FAIL summary, matching this project's existing convention -- see
e.g. benchmark/test_smolvla_libero_action_adapter.py -- not pytest).

Verifies V1 (benchmark.so101_scripted_expert.run_pick_and_place_episode,
UNMODIFIED by this task) still behaves identically to itself across two
separate process launches (determinism) AND still reproduces
already-known-good outcomes from this project's own existing datasets
(cube baseline, Stage 1B box yaw=0) -- this is the "frozen baseline" no
Expert V2 work is allowed to disturb.

Run:
  .venv-vla/bin/python -m benchmark.test_so101_expert_v1_regression
"""

import json
from pathlib import Path

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_MODE_ANCHOR_OFFSET_XY, FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY
from benchmark.collect_so101_stage1b_box_dataset import BOX_FOOTPRINT_XY, REGION_DEFS as STAGE1B_REGION_DEFS
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_scripted_expert import PHASE_ID_BY_NAME, So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend

CUBE_SEED = 1  # matches this project's own first validation environment seed, used throughout Stage 1A/1B baselines
BOX_SEED = 15000  # matches datasets/so101_bin_stage1b_box_100's own episode_index=0 (region=center) -- a KNOWN, already-collected-successfully scenario

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def build_backend(object_footprint_xy=None, seed=CUBE_SEED, x_range=(-0.015, 0.015), y_range=(-0.015, 0.015)):
    from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE

    x_range = x_range if x_range is not None else FIXED_BIN_OBJECT_X_RANGE
    y_range = y_range if y_range is not None else FIXED_BIN_OBJECT_Y_RANGE
    sampled_object_position = sample_object_position(seed, x_range, y_range)
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}
    if object_footprint_xy is not None:
        scene_config["object_footprint_xy"] = object_footprint_xy
    backend = So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config,
    )
    return backend, sampled_object_position


def run_v1_episode(object_footprint_xy=None, seed=CUBE_SEED, x_range=(-0.015, 0.015), y_range=(-0.015, 0.015)):
    backend, sampled_position = build_backend(object_footprint_xy, seed, x_range, y_range)
    frames = []

    def on_step(phase, arm_joint_targets, gripper_target_normalized):
        frames.append({"phase": phase, "arm_joint_targets": list(arm_joint_targets), "gripper_target_normalized": gripper_target_normalized})

    try:
        backend.reset()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        try:
            result = run_pick_and_place_episode(backend, transport_delta_xy, on_step=on_step)
            failure_reason = result["failure_reason"]
            place_success = result["place_success"]
        except So101ExpertError as exc:
            failure_reason = exc.failure_reason
            place_success = False
    finally:
        backend.close()

    gripper_close_step = next(
        (f["step"] for f in [dict(fr, step=i) for i, fr in enumerate(frames)] if f["gripper_target_normalized"] <= 0.15), None,
    )
    return {
        "place_success": place_success, "failure_reason": failure_reason, "num_frames": len(frames),
        "frames": frames, "gripper_close_step": gripper_close_step, "sampled_position": sampled_position,
    }


def main() -> None:
    # 1. Cube determinism: run twice, identical action trajectory
    run1 = run_v1_episode(object_footprint_xy=None, seed=CUBE_SEED)
    run2 = run_v1_episode(object_footprint_xy=None, seed=CUBE_SEED)
    check("cube: run1 place_success == True (known baseline outcome)", run1["place_success"] is True)
    check("cube: run1 == run2 num_frames (determinism)", run1["num_frames"] == run2["num_frames"],
          detail=f"{run1['num_frames']} vs {run2['num_frames']}")
    check("cube: run1 == run2 gripper_close_step (determinism)", run1["gripper_close_step"] == run2["gripper_close_step"])
    check("cube: run1 == run2 full action trajectory (determinism)", run1["frames"] == run2["frames"])
    check("cube: run1 == run2 sampled_position (determinism)", run1["sampled_position"] == run2["sampled_position"])

    # 2. Box yaw=0 determinism + reproduces the KNOWN successful dataset episode
    center_region = STAGE1B_REGION_DEFS["center"]
    box_run1 = run_v1_episode(
        object_footprint_xy=BOX_FOOTPRINT_XY, seed=BOX_SEED, x_range=center_region["x_range"], y_range=center_region["y_range"],
    )
    box_run2 = run_v1_episode(
        object_footprint_xy=BOX_FOOTPRINT_XY, seed=BOX_SEED, x_range=center_region["x_range"], y_range=center_region["y_range"],
    )
    check("box yaw=0: run1 place_success == True (matches datasets/so101_bin_stage1b_box_100 episode_index=0)", box_run1["place_success"] is True)
    check("box yaw=0: sampled_position matches recorded dataset manifest (object_x/object_y)",
          abs(box_run1["sampled_position"][0] - 0.39292548873586397) < 1e-9 and abs(box_run1["sampled_position"][1] - 0.002867399132047091) < 1e-9,
          detail=str(box_run1["sampled_position"]))
    check("box yaw=0: run1 == run2 num_frames (determinism)", box_run1["num_frames"] == box_run2["num_frames"])
    check("box yaw=0: run1 == run2 gripper_close_step (determinism)", box_run1["gripper_close_step"] == box_run2["gripper_close_step"])
    check("box yaw=0: run1 == run2 full action trajectory (determinism)", box_run1["frames"] == box_run2["frames"])

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
