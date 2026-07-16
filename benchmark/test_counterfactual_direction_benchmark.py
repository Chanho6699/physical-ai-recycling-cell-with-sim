"""Tests for benchmark/run_counterfactual_direction_benchmark.py (v0).

All tests run against a FakeServerPolicy mock (BasePolicy subclass) --
no live GPU/network server required, per this task's explicit
requirement. Covers:

  1. _sign_match() edge cases (negligible vector / negligible commanded)
  2. run_condition() wiring in both "one-step" (no movement) and
     "multi-step" (real PyBulletPandaBackend movement) modes
  3. judge_fixed_axis_bias() on synthetic mirrored-position data
  4. judge_language_issue() on synthetic per-instruction data
  5. judge_domain_gap() on synthetic all-positions/all-instructions data
  6. judge_seed_instability() on synthetic per-seed data
  7. a full run_benchmark() over a small grid (2 positions x 2
     instructions x 2 seeds), confirming the end-to-end pipeline
     (log file + summary structure) works against the mock
  8. regression: run_vla_action_direction_diagnostic.py's own test suite
     still passes (this file reuses several of its building blocks)

Run: python -m benchmark.test_counterfactual_direction_benchmark
"""

import subprocess
import sys

from benchmark.run_counterfactual_direction_benchmark import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_POSITIONS,
    _sign_match,
    judge_domain_gap,
    judge_fixed_axis_bias,
    judge_language_issue,
    judge_seed_instability,
    parse_args,
    run_benchmark,
    run_condition,
)
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Always returns a fixed, non-degraded, non-fallback response whose
    translation direction is controllable per-test -- so run_condition()
    wiring can be checked without a live GPU server."""

    def __init__(self, translation=(0.02, 0.0, -0.02), gripper_raw=-0.5):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []
        self._translation = translation
        self._gripper_raw = gripper_raw

    def reset(self) -> None:
        self.received_inputs = []

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        raw_model_action = [
            self._translation[0] * 5, self._translation[1] * 5, self._translation[2] * 5, 0.0, 0.0, 0.0, self._gripper_raw,
        ]
        gripper_opening_01 = 0.0 if self._gripper_raw >= 0.0 else 1.0
        wire_gripper = 1.0 if gripper_opening_01 <= 0.5 else 0.0
        action = list(self._translation) + [0.0, 0.0, 0.0, wire_gripper]
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
                    "translation_m": list(self._translation),
                    "gripper_opening_01": gripper_opening_01,
                    "metadata": {"raw_model_action": raw_model_action, "gripper_raw": self._gripper_raw},
                },
                "canonical_command_pre_safety_filter": {"translation_m": list(self._translation)},
                "safety_filter_clipped": False,
                "safety_clipped": False,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


def make_args(**overrides):
    argv_backup = sys.argv
    try:
        sys.argv = ["run_counterfactual_direction_benchmark.py"]
        args = parse_args()
    finally:
        sys.argv = argv_backup
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def main() -> None:
    print("=== 1. _sign_match edge cases ===")
    check("matching positive signs -> True", _sign_match(0.02, 0.05) is True)
    check("opposite signs -> False", _sign_match(-0.02, 0.05) is False)
    check("negligible vector component -> None (not meaningful)", _sign_match(0.02, 1e-6) is None)
    check("negligible commanded component vs. real vector -> False (didn't move that way)", _sign_match(1e-6, 0.05) is False)
    print()

    print("=== 2. run_condition() wiring ===")
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    fake_policy = FakeServerPolicy(translation=(0.02, 0.0, -0.01))

    rows_one_step = run_condition(
        fake_policy, "center_right", DEFAULT_POSITIONS["center_right"], "en_full", DEFAULT_INSTRUCTIONS["en_full"],
        seed=42, mode="one-step", steps_per_condition=5, steps_per_action=10, object_type="plastic_bottle",
        bin_position=[0.3, 0.35, 0.05], strict=True, backend=backend,
    )
    check("one-step mode produces exactly 1 row regardless of steps_per_condition", len(rows_one_step) == 1, f"got {len(rows_one_step)}")
    ee_after_one_step = backend.get_state()["end_effector_position"]
    initial_ee = [0.3068690001964569, -9.325392966275103e-06, 0.48526519536972046]
    check(
        "one-step mode does NOT move the robot (apply_command never called)",
        all(abs(ee_after_one_step[i] - initial_ee[i]) < 1e-4 for i in range(3)),
        f"ee_after={ee_after_one_step}",
    )
    row = rows_one_step[0]
    for field in (
        "object_position", "ee_position", "vector_ee_to_object", "instruction", "seed", "raw_model_action",
        "adapted_translation", "commanded_translation", "cosine_commanded_vs_object", "sign_match_x", "sign_match_y",
        "sign_match_z", "gripper_command", "server_latency_ms", "main_image_hash", "wrist_image_hash",
        "compatibility_passed", "degraded_input", "fallback_used",
    ):
        check(f"row has required field '{field}'", field in row, f"keys={sorted(row.keys())}")
    check("row.instruction matches the requested instruction", row["instruction"] == DEFAULT_INSTRUCTIONS["en_full"])
    check("row.seed matches the requested seed", row["seed"] == 42)
    check(
        "commanded_translation matches FakeServerPolicy's fixed translation",
        [round(v, 6) for v in row["commanded_translation"]] == [0.02, 0.0, -0.01],
    )
    backend.shutdown()

    backend2 = PyBulletPandaBackend(gui=False)
    backend2.reset()
    fake_policy2 = FakeServerPolicy(translation=(0.02, 0.0, -0.01))
    rows_multi_step = run_condition(
        fake_policy2, "center_right", DEFAULT_POSITIONS["center_right"], "en_full", DEFAULT_INSTRUCTIONS["en_full"],
        seed=42, mode="multi-step", steps_per_condition=3, steps_per_action=10, object_type="plastic_bottle",
        bin_position=[0.3, 0.35, 0.05], strict=True, backend=backend2,
    )
    check("multi-step mode produces steps_per_condition rows", len(rows_multi_step) == 3, f"got {len(rows_multi_step)}")
    ee_after_multi_step = backend2.get_state()["end_effector_position"]
    check(
        "multi-step mode DOES move the robot",
        not all(abs(ee_after_multi_step[i] - initial_ee[i]) < 1e-4 for i in range(3)),
        f"ee_after={ee_after_multi_step}",
    )
    check(
        "ee_position differs between step 0 and step 1 in multi-step mode (fresh observation each step)",
        rows_multi_step[0]["ee_position"] != rows_multi_step[1]["ee_position"],
    )
    backend2.shutdown()
    print()

    print("=== 3. judge_fixed_axis_bias() ===")
    def make_row(position_name, commanded_x, vector_x):
        return {
            "position_name": position_name, "commanded_translation": [commanded_x, 0.0, 0.0],
            "vector_ee_to_object": [vector_x, 0.0, 0.0], "sign_match_x": (commanded_x > 0) == (vector_x > 0),
        }

    biased_rows = (
        [make_row("center_right", 0.02, 0.07) for _ in range(3)]  # object to the right, commanded +x (correct)
        + [make_row("center_left", 0.02, -0.07) for _ in range(3)]  # object to the left, commanded STILL +x (bug)
    )
    result = judge_fixed_axis_bias(biased_rows, DEFAULT_POSITIONS, axis="x", mirror_pair=("center_right", "center_left"))
    check("fixed x bias correctly detected when commanded sign doesn't flip with the object", result["suspected"] is True, str(result))

    unbiased_rows = (
        [make_row("center_right", 0.02, 0.07) for _ in range(3)]
        + [make_row("center_left", -0.02, -0.07) for _ in range(3)]  # commanded sign DOES flip -- correct behavior
    )
    result = judge_fixed_axis_bias(unbiased_rows, DEFAULT_POSITIONS, axis="x", mirror_pair=("center_right", "center_left"))
    check("no fixed bias reported when commanded sign correctly tracks the object", result["suspected"] is False, str(result))
    print()

    print("=== 4. judge_language_issue() ===")
    result = judge_language_issue({
        "ko_full": {"mean_cosine": 0.02}, "en_full": {"mean_cosine": 0.55},
        "en_short": {"mean_cosine": 0.5}, "en_minimal": {"mean_cosine": 0.48},
    })
    check("large ko-vs-en gap -> language issue suspected", result["suspected"] is True, str(result))
    result = judge_language_issue({
        "ko_full": {"mean_cosine": 0.20}, "en_full": {"mean_cosine": 0.22},
        "en_short": {"mean_cosine": 0.19}, "en_minimal": {"mean_cosine": 0.21},
    })
    check("small gap across instructions -> language issue NOT suspected", result["suspected"] is False, str(result))
    print()

    print("=== 5. judge_domain_gap() ===")
    overall_bad = {"mean_cosine": 0.02}
    by_position_bad = {name: {"mean_cosine": 0.03} for name in DEFAULT_POSITIONS}
    by_instruction_bad = {name: {"mean_cosine": 0.01} for name in DEFAULT_INSTRUCTIONS}
    result = judge_domain_gap(overall_bad, by_position_bad, by_instruction_bad)
    check("uniformly low cosine everywhere -> domain gap suspected", result["suspected"] is True, str(result))

    overall_mixed = {"mean_cosine": 0.15}
    by_position_mixed = {**{name: {"mean_cosine": 0.05} for name in DEFAULT_POSITIONS}, "center_right": {"mean_cosine": 0.7}}
    by_instruction_mixed = {name: {"mean_cosine": 0.15} for name in DEFAULT_INSTRUCTIONS}
    result = judge_domain_gap(overall_mixed, by_position_mixed, by_instruction_mixed)
    check("one strong position -> domain gap NOT suspected", result["suspected"] is False, str(result))
    print()

    print("=== 6. judge_seed_instability() ===")
    result = judge_seed_instability({"a__b": 0.5, "c__d": 0.6, "e__f": 0.55})
    check("high across-seed std -> instability suspected", result["suspected"] is True, str(result))
    result = judge_seed_instability({"a__b": 0.05, "c__d": 0.1, "e__f": 0.08})
    check("low across-seed std -> instability NOT suspected", result["suspected"] is False, str(result))
    result = judge_seed_instability({})
    check("no data -> suspected is None (not a crash, not a false claim)", result["suspected"] is None, str(result))
    print()

    print("=== 7. full run_benchmark() over a small grid, against the mock ===")
    small_positions = {"center_right": DEFAULT_POSITIONS["center_right"], "center_left": DEFAULT_POSITIONS["center_left"]}
    small_instructions = {"ko_full": DEFAULT_INSTRUCTIONS["ko_full"], "en_full": DEFAULT_INSTRUCTIONS["en_full"]}
    args = make_args(seeds=[0, 42], mode="one-step", steps_per_condition=1, strict=True)
    fake_policy3 = FakeServerPolicy(translation=(0.02, 0.0, -0.01))
    result = run_benchmark(args, policy=fake_policy3, positions=small_positions, instructions=small_instructions)
    expected_rows = 2 * 2 * 2  # positions x instructions x seeds, 1 row each (one-step mode)
    check("run_benchmark produced one row per condition", len(result["rows"]) == expected_rows, f"got {len(result['rows'])}")
    check("summary has all 5 judgments", set(result["summary"]["judgments"].keys()) == {
        "fixed_x_bias", "fixed_y_bias", "language_issue", "domain_gap", "seed_instability",
    })
    check("summary.by_position covers both positions", set(result["summary"]["by_position"].keys()) == set(small_positions))
    check("summary.by_instruction covers both instructions", set(result["summary"]["by_instruction"].keys()) == set(small_instructions))
    import os

    check("JSONL log file was actually written", os.path.exists(result["log_path"]))
    check("summary JSON file was actually written", os.path.exists(result["summary_path"]))
    print()

    print("=== 8. regression: run_vla_action_direction_diagnostic.py's own test suite ===")
    proc = subprocess.run(
        [sys.executable, "-m", "benchmark.test_vla_action_direction_diagnostic"], capture_output=True, text=True, timeout=300
    )
    passed = "ALL CHECKS PASSED" in proc.stdout
    check("benchmark.test_vla_action_direction_diagnostic -- ALL CHECKS PASSED", passed, proc.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
