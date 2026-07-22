"""Generates a single, fixed SO-101 pick-and-place action trajectory
FILE for vla_server's expert_replay backend (see this task's chat
report, "Desktop용 Expert-Replay VLA Server"). Does NOT run the
scripted expert live per HTTP request -- runs it exactly ONCE here,
records every (phase, arm_joint_targets_rad, gripper_target_normalized)
step via benchmark.so101_scripted_expert.run_pick_and_place_episode()'s
own on_step hook (reused, not reimplemented), and writes the sequence
plus the scene's initial conditions and action-space semantics to a
plain JSON file under vla_server/expert_replay_trajectories/.

Reuses (does NOT reimplement): So101PyBulletBackend construction
pattern, FIXED_BIN_MODE_* constants, and sample_object_position() --
the EXACT same scene-building code
benchmark/so101_smolvla_rollout.py's own build_rollout_backend(seed)
and benchmark/collect_so101_bin_dataset.py's own collect_episode()
already use, so the replay trajectory is generated under a real,
already-validated SO-101 bin scenario, not a synthetic/ad-hoc one.

Run:
  .venv-vla/bin/python -m benchmark.generate_so101_expert_replay_trajectory
"""

import json
from pathlib import Path

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    FIXED_BIN_OBJECT_X_RANGE,
    FIXED_BIN_OBJECT_Y_RANGE,
)
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_scripted_expert import So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import ARM_JOINT_NAMES, DEFAULT_SCENE_CONFIG, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "vla_server" / "expert_replay_trajectories"
DEFAULT_TRAJECTORY_SEED = 1  # matches the first of this project's own 40 validation environment seeds -- an already-well-characterized scenario, not an arbitrary new one
TRAJECTORY_ID = "so101_pick_place_seed1_v1"
GRIPPER_CONVENTION = "0.0 = closed, 1.0 = open"  # matches robot_sim/so101_pybullet_backend.py's own GRASP_GRIPPER_CLOSED_THRESHOLD/GRIPPER_OPEN_RELEASE_THRESHOLD comments


def build_backend(seed: int) -> So101PyBulletBackend:
    """Identical construction to benchmark/so101_smolvla_rollout.py's own
    build_rollout_backend(seed) -- not reimplemented independently, kept
    as a local copy only because importing that module pulls in torch/
    SmolVLA checkpoint-loading code this generator script has no need
    for."""
    sampled_object_position = sample_object_position(seed, FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE)
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    return So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config={"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
    ), sampled_object_position, fixed_bin_center_xy


def generate_trajectory(seed: int = DEFAULT_TRAJECTORY_SEED) -> dict:
    backend, object_position, bin_center_xy = build_backend(seed)
    steps = []

    def on_step(phase, arm_joint_targets, gripper_target_normalized):
        steps.append({
            "step_index": len(steps),
            "phase": phase,
            "arm_joint_targets_rad": [float(v) for v in arm_joint_targets],
            "gripper_target_normalized": float(gripper_target_normalized),
        })

    try:
        backend.reset()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        result = run_pick_and_place_episode(backend, transport_delta_xy, on_step=on_step)
        place_success = result["place_success"]
        failure_reason = result["failure_reason"]
    except So101ExpertError as exc:
        place_success = False
        failure_reason = exc.failure_reason
    finally:
        backend.close()

    trajectory = {
        "trajectory_id": TRAJECTORY_ID,
        "generator": "benchmark.so101_scripted_expert.run_pick_and_place_episode",
        "generation_seed": seed,
        "generation_place_success": place_success,
        "generation_failure_reason": failure_reason,
        "initial_conditions": {
            "object_position": [float(v) for v in object_position],
            "bin_center_xy": [float(v) for v in bin_center_xy],
            "collection_mode": "fixed_bin_object_xy",
        },
        "action_space_metadata": {
            "joint_order": list(ARM_JOINT_NAMES) + ["gripper"],
            "arm_units": "radians_absolute_joint_target",
            "gripper_units": "normalized_0_1",
            "gripper_convention": GRIPPER_CONVENTION,
            "action_dim": len(ARM_JOINT_NAMES) + 1,
            "chunk_size": 1,
        },
        "num_steps": len(steps),
        "steps": steps,
    }
    return trajectory


def main() -> None:
    trajectory = generate_trajectory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{TRAJECTORY_ID}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2)

    print(f"generation_place_success: {trajectory['generation_place_success']}")
    print(f"generation_failure_reason: {trajectory['generation_failure_reason']}")
    print(f"num_steps: {trajectory['num_steps']}")
    print(f"Trajectory JSON: {output_path}")


if __name__ == "__main__":
    main()
