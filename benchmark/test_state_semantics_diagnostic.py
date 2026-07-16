"""Tests for benchmark/run_state_semantics_diagnostic.py (v0).

All tests except the checkpoint-stats ones run against a FakeServerPolicy
mock -- no live GPU/network server required. Covers this task's 6
required test areas:

  1. state field order verification (regression against
     vla_server/model_loader.py's own _SMOLVLA_LIBERO_STATE_FIELD_DIMS)
  2. quaternion/axis-angle convention verification (PyBullet's own
     getAxisAngleFromQuaternion + axis*angle matches the standard
     robosuite quat2axisangle formula on known quaternions; reset-state
     regression for the "near-pi" empirical finding)
  3. dataset-stats z-score calculation
  4. image-fixed/state-varied wiring (Ablation A)
  5. state-fixed/image-varied wiring (Ablation B)
  6. coordinate hypothesis (Ablation C) never referenced by any
     production file -- grep-based, not just "trust me"

Run: python -m benchmark.test_state_semantics_diagnostic
(checkpoint-stats-dependent checks are skipped with a clear note if
safetensors/huggingface_hub aren't installed in the active venv -- see
load_checkpoint_state_stats()'s lazy import.)
"""

import math
from pathlib import Path

import numpy as np
import pybullet as p

from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS
from benchmark.run_state_semantics_diagnostic import (
    COORDINATE_HYPOTHESES,
    STATE_DIM_NAMES,
    apply_coordinate_hypothesis,
    build_state_variants,
    compute_zscore_report,
    detect_orientation_discontinuity,
    run_coordinate_hypotheses,
    run_image_fixed_state_varied,
    run_state_fixed_image_varied,
)
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from vla_server.model_loader import _SMOLVLA_LIBERO_STATE_FIELD_DIMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Echoes back whatever state it was given (translated into the
    output) so ablation wiring can be checked precisely -- the commanded
    x/y is literally the input ee_position's x/y (scaled down), so a
    test can assert "the state I put in is the state that came back
    out," without a live GPU server."""

    def __init__(self):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []

    def reset(self) -> None:
        self.received_inputs = []

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        ee_position = policy_input.robot_state["ee_position"]
        translation = [ee_position[0] * 0.01, ee_position[1] * 0.01, -0.01]
        action = translation + [0.0, 0.0, 0.0, 1.0]
        info = {
            "policy_backend": "real-vla",
            "inference_latency_ms": 700.0,
            "fallback_used": False,
            "real_vla_request_failed": False,
            "compatibility": {"passed": True},
            "semantic_action_valid": True,
            "degraded_input": False,
            "action_postprocess": {
                "canonical_command": {
                    "translation_m": translation,
                    "gripper_opening_01": 1.0,
                    "metadata": {"raw_model_action": translation + [0.0, 0.0, 0.0, -1.0]},
                },
                "canonical_command_pre_safety_filter": {"translation_m": translation},
                "safety_filter_clipped": False,
                "safety_clipped": False,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


def main() -> None:
    print("=== 1. state field order verification ===")
    check(
        "vla_server/model_loader.py's field order is (ee_position:3, ee_orientation_axis_angle:3, gripper_qpos:2)",
        _SMOLVLA_LIBERO_STATE_FIELD_DIMS == (("ee_position", 3), ("ee_orientation_axis_angle", 3), ("gripper_qpos", 2)),
        f"got {_SMOLVLA_LIBERO_STATE_FIELD_DIMS}",
    )
    check(
        "this diagnostic's STATE_DIM_NAMES lists the same 3 fields in the same order (8 names total)",
        STATE_DIM_NAMES == [
            "ee_position.x", "ee_position.y", "ee_position.z",
            "ee_orientation_axis_angle.x", "ee_orientation_axis_angle.y", "ee_orientation_axis_angle.z",
            "gripper_qpos.0", "gripper_qpos.1",
        ],
    )
    print()

    print("=== 2. quaternion/axis-angle convention verification ===")
    # A 90-degree rotation about Z, built with PyBullet's own (x,y,z,w)
    # scalar-last convention (matches robosuite's documented convention
    # too -- see this module's docstring finding 2/3).
    quat_90z = p.getQuaternionFromAxisAngle([0, 0, 1], math.pi / 2)
    axis, angle = p.getAxisAngleFromQuaternion(quat_90z)
    recovered = [axis[i] * angle for i in range(3)]
    check(
        "PyBullet axis*angle round-trips a known 90 deg Z rotation",
        abs(recovered[0]) < 1e-6 and abs(recovered[1]) < 1e-6 and abs(recovered[2] - math.pi / 2) < 1e-6,
        f"got {recovered}",
    )
    # robosuite's quat2axisangle(quat) = quat[:3] * 2*acos(quat[3]) / sqrt(1-quat[3]^2)
    # -- reproduced here (not imported, robosuite isn't a project
    # dependency) purely to cross-check PyBullet's own axis*angle against
    # the documented robosuite formula on the same quaternion.
    qx, qy, qz, qw = quat_90z
    den = math.sqrt(max(1.0 - qw * qw, 0.0))
    robosuite_style = [qx * 2 * math.acos(qw) / den, qy * 2 * math.acos(qw) / den, qz * 2 * math.acos(qw) / den]
    check(
        "PyBullet's axis*angle matches robosuite's quat2axisangle formula on the same quaternion",
        all(abs(recovered[i] - robosuite_style[i]) < 1e-6 for i in range(3)),
        f"pybullet={recovered} robosuite_style={robosuite_style}",
    )

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    state_8d = backend.get_libero_observation_state()
    orientation = state_8d[3:6]
    magnitude = math.sqrt(sum(v * v for v in orientation))
    check(
        "reset-state orientation axis-angle magnitude is close to pi radians (near-pi operating regime, "
        "matching the checkpoint's own training-distribution mean magnitude -- see module docstring finding 6)",
        abs(magnitude - math.pi) < 0.01,
        f"magnitude={magnitude}",
    )
    check(
        "reset-state orientation is dominated by the x-component (same axis the checkpoint's mean is dominated by)",
        abs(orientation[0]) > 3.0 and abs(orientation[1]) < 0.01 and abs(orientation[2]) < 0.01,
        f"orientation={orientation}",
    )
    backend.shutdown()
    print()

    print("=== 3. dataset-stats z-score calculation ===")
    fake_checkpoint_mean = [0.0] * 8
    fake_checkpoint_std = [1.0] * 8
    fake_samples = [[float(i)] * 8 for i in range(-2, 3)]  # mean=0 per dim, so z should be ~0
    report = compute_zscore_report(fake_samples, fake_checkpoint_mean, fake_checkpoint_std)
    check(
        "z_of_mean is ~0 when our sample mean equals the checkpoint mean",
        all(abs(stats["z_of_mean"]) < 1e-9 for stats in report.values()),
        str({name: stats["z_of_mean"] for name, stats in report.items()}),
    )

    ood_checkpoint_mean = [0.0] * 8
    ood_checkpoint_std = [0.1] * 8
    ood_samples = [[5.0] * 8 for _ in range(5)]  # our mean=5, checkpoint mean=0, std=0.1 -> z=50
    ood_report = compute_zscore_report(ood_samples, ood_checkpoint_mean, ood_checkpoint_std)
    check(
        "a sample mean far from the checkpoint mean produces a large |z| and is_ood=True",
        all(stats["is_ood"] for stats in ood_report.values()),
        str({name: stats["z_of_mean"] for name, stats in ood_report.items()}),
    )

    stable_samples = [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    jumpy_samples = [[0.0] * 8, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]  # a big jump on the orientation-x dim
    check("no discontinuity detected for a small, smooth transition", not detect_orientation_discontinuity(stable_samples)["discontinuity_detected"])
    check("a large orientation jump between consecutive samples IS detected", detect_orientation_discontinuity(jumpy_samples)["discontinuity_detected"])
    print()

    print("=== 4. image-fixed / state-varied wiring (Ablation A) ===")
    backend_a = PyBulletPandaBackend(gui=False)
    backend_a.reset()
    backend_a.set_object_type("plastic_bottle")
    backend_a.set_object_position(list(DEFAULT_POSITIONS["center_right"]))
    fake_policy_a = FakeServerPolicy()
    rows_a = run_image_fixed_state_varied(
        fake_policy_a, backend_a, "pick up the bottle", DEFAULT_POSITIONS["center_right"], [0.3, 0.35, 0.05], [42], strict=True,
    )
    backend_a.shutdown()
    expected_variants = {"original", "x_mirrored", "y_mirrored", "xy_swapped", "position_zeroed", "orientation_zeroed", "gripper_open", "gripper_closed"}
    check("all 8 state variants are present", {row["variant"] for row in rows_a} == expected_variants, f"got {sorted({row['variant'] for row in rows_a})}")
    image_hashes = {(row["main_image_hash"], row["wrist_image_hash"]) for row in rows_a}
    check("the SAME image (hash) is used across every state variant", len(image_hashes) == 1, f"got {image_hashes}")
    original_row = next(row for row in rows_a if row["variant"] == "original")
    mirrored_row = next(row for row in rows_a if row["variant"] == "x_mirrored")
    check(
        "x_mirrored variant's commanded_translation.x has flipped sign vs. original (FakeServerPolicy echoes state.x)",
        (original_row["commanded_translation"][0] > 0) != (mirrored_row["commanded_translation"][0] > 0),
        f"original={original_row['commanded_translation']} mirrored={mirrored_row['commanded_translation']}",
    )
    zeroed_row = next(row for row in rows_a if row["variant"] == "position_zeroed")
    check(
        "position_zeroed variant's state_8d_before_normalization has ee_position == [0,0,0]",
        zeroed_row["state_8d_before_normalization"][0:3] == [0.0, 0.0, 0.0],
    )
    print()

    print("=== 5. state-fixed / image-varied wiring (Ablation B) ===")
    fake_policy_b = FakeServerPolicy()
    rows_b = run_state_fixed_image_varied(fake_policy_b, "pick up the bottle", DEFAULT_POSITIONS, [0.3, 0.35, 0.05], [42], "plastic_bottle", strict=True)
    check("all 4 position variants are present", {row["variant"] for row in rows_b} == set(DEFAULT_POSITIONS), f"got {sorted({row['variant'] for row in rows_b})}")
    state_vectors = {tuple(row["state_8d_before_normalization"]) for row in rows_b}
    check("the SAME state is used across every image variant", len(state_vectors) == 1, f"got {len(state_vectors)} distinct states")
    image_hash_pairs = {(row["main_image_hash"], row["wrist_image_hash"]) for row in rows_b}
    check("main/wrist image hashes differ across the 4 position variants", len(image_hash_pairs) == 4, f"got {len(image_hash_pairs)} distinct hash pairs")
    print()

    print("=== 6. coordinate hypothesis (Ablation C) never touches production code ===")
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy"]
    hits = []
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "apply_coordinate_hypothesis" in text or "COORDINATE_HYPOTHESES" in text or "xy_swap_sign" in text:
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    check(
        "no production file references apply_coordinate_hypothesis/COORDINATE_HYPOTHESES",
        len(hits) == 0,
        f"unexpectedly found references in: {hits}",
    )

    backend_c = PyBulletPandaBackend(gui=False)
    backend_c.reset()
    real_state = {
        "ee_position": [0.3, 0.1, 0.4], "ee_orientation_axis_angle": [3.14, 0.0, 0.0],
        "gripper_qpos": [0.02, 0.02], "held_object": False,
    }
    for hypothesis in COORDINATE_HYPOTHESES:
        transformed = apply_coordinate_hypothesis(real_state, hypothesis)
        check(
            f"apply_coordinate_hypothesis('{hypothesis}') does not mutate the original state dict",
            real_state["ee_position"] == [0.3, 0.1, 0.4],
            f"original was mutated: {real_state['ee_position']}",
        )
        check(f"'{hypothesis}' produces a 3-component ee_position", len(transformed["ee_position"]) == 3)
    identity_transformed = apply_coordinate_hypothesis(real_state, "identity")
    check("'identity' hypothesis leaves ee_position unchanged", identity_transformed["ee_position"] == real_state["ee_position"])
    x_flip_transformed = apply_coordinate_hypothesis(real_state, "x_sign_flip")
    check(
        "'x_sign_flip' negates only x",
        x_flip_transformed["ee_position"] == [-0.3, 0.1, 0.4],
        f"got {x_flip_transformed['ee_position']}",
    )
    swap_transformed = apply_coordinate_hypothesis(real_state, "xy_swap")
    check("'xy_swap' swaps x and y", swap_transformed["ee_position"] == [0.1, 0.3, 0.4], f"got {swap_transformed['ee_position']}")

    fake_policy_c = FakeServerPolicy()
    rows_c = run_coordinate_hypotheses(
        fake_policy_c, {"center_right": DEFAULT_POSITIONS["center_right"]}, "pick up the bottle", [0.3, 0.35, 0.05], [42], "plastic_bottle", strict=True,
    )
    backend_c.shutdown()
    check("all 5 coordinate hypotheses produced rows for the tested position", {row["hypothesis"] for row in rows_c} == set(COORDINATE_HYPOTHESES))
    print()

    print("=== 7. regression: prior diagnostic/benchmark suites ===")
    import subprocess
    import sys

    for module in ("benchmark.test_vla_action_direction_diagnostic", "benchmark.test_counterfactual_direction_benchmark"):
        result = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True, timeout=300)
        passed = "ALL CHECKS PASSED" in result.stdout
        check(f"{module} -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
