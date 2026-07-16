"""Tests for benchmark/run_gripper_channel_sign_ab_experiment.py (v0).

All tests run against a FakeServerPolicy mock -- no live GPU/network
server required. Covers:

  1. apply_gripper_condition() correctness for all 4 conditions +
     non-mutation of the input dict
  2. one-step grid wiring: conditions differ ONLY in gripper_qpos (same
     images, same ee_position/orientation across all 4 conditions);
     real_q1/real_q2 actually come from the backend's real open/closed
     finger qpos
  3. compare_conditions()'s delta/consistency arithmetic on synthetic rows
  4. judge_consistent_improvement()/judge_improvement_scope() on
     synthetic comparison data (clear-improvement, no-improvement,
     gripper-only, translation-only cases)
  5. count_gripper_switches() on a synthetic step sequence
  6. this module's own diagnostic-only functions are never referenced
     from any production directory (grep-based, not "trust me")

Run: python -m benchmark.test_gripper_channel_sign_ab_experiment
"""

from pathlib import Path

from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS
from benchmark.run_gripper_channel_sign_ab_experiment import (
    ALL_CONDITIONS,
    DERIVED_CONDITIONS,
    FIXED_CONDITIONS,
    apply_gripper_condition,
    compare_conditions,
    count_gripper_switches,
    judge_consistent_improvement,
    judge_improvement_scope,
    run_one_step_grid,
)
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Deliberately makes the response depend on gripper_qpos[1]'s sign
    (close if positive, open if negative) and echoes ee_position.y into
    commanded y -- so the test can verify the actual A/B behavioral
    difference (not just "some value passed through") is wired
    correctly end to end."""

    def __init__(self):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []

    def reset(self) -> None:
        self.received_inputs = []

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        gripper_qpos = policy_input.robot_state["gripper_qpos"]
        ee_position = policy_input.robot_state["ee_position"]
        wire_gripper = 1.0 if gripper_qpos[1] > 0 else 0.0  # 1.0 = close (legacy wire polarity)
        translation = [0.01, ee_position[1] * 0.01, -0.005]
        action = translation + [0.0, 0.0, 0.0, wire_gripper]
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
                    "gripper_opening_01": 0.0 if wire_gripper == 1.0 else 1.0,
                    "metadata": {
                        "raw_model_action": translation + [0.0, 0.0, 0.0, gripper_qpos[1] * 10],
                        "native_action_raw_values": translation + [0.0, 0.0, 0.0, gripper_qpos[1] * 5],
                    },
                },
                "canonical_command_pre_safety_filter": {"translation_m": translation},
                "safety_filter_clipped": False,
                "safety_clipped": False,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


def main() -> None:
    print("=== 1. apply_gripper_condition() correctness ===")
    base_state = {"ee_position": [0.3, 0.1, 0.4], "ee_orientation_axis_angle": [3.14, 0.0, 0.0], "gripper_qpos": [999.0, 999.0]}
    check("ALL_CONDITIONS has exactly 4 entries", len(ALL_CONDITIONS) == 4, f"got {ALL_CONDITIONS}")

    current = apply_gripper_condition(base_state, "current_positive_pair", real_q1=0.03, real_q2=0.03)
    check("current_positive_pair keeps both channels as-is", current["gripper_qpos"] == [0.03, 0.03])

    mirrored = apply_gripper_condition(base_state, "mirrored_signed_pair", real_q1=0.03, real_q2=0.03)
    check("mirrored_signed_pair negates ONLY the second channel", mirrored["gripper_qpos"] == [0.03, -0.03], f"got {mirrored['gripper_qpos']}")

    ckpt = apply_gripper_condition(base_state, "checkpoint_mean_open")
    check("checkpoint_mean_open returns the fixed checkpoint-derived pair", ckpt["gripper_qpos"] == FIXED_CONDITIONS["checkpoint_mean_open"])

    zero = apply_gripper_condition(base_state, "zero_pair")
    check("zero_pair returns [0.0, 0.0]", zero["gripper_qpos"] == [0.0, 0.0])

    check(
        "apply_gripper_condition never mutates the input dict",
        base_state["gripper_qpos"] == [999.0, 999.0],
        f"original was mutated: {base_state['gripper_qpos']}",
    )
    check(
        "ee_position/orientation are untouched by any condition (only gripper_qpos changes)",
        current["ee_position"] == base_state["ee_position"] and mirrored["ee_position"] == base_state["ee_position"],
    )
    print()

    print("=== 2. one-step grid wiring ===")
    small_positions = {"center_right": DEFAULT_POSITIONS["center_right"]}
    small_instructions = {"en_minimal": "Move the gripper toward the bottle."}
    fake_policy = FakeServerPolicy()
    rows = run_one_step_grid(fake_policy, small_positions, small_instructions, [42], "plastic_bottle", [0.3, 0.35, 0.05], strict=True, checkpoint_stats=None)

    check("rows produced for all 4 conditions", {row["condition"] for row in rows} == set(ALL_CONDITIONS), f"got {sorted({row['condition'] for row in rows})}")
    check("both grip_physical_state=open and closed appear for the derived conditions", {row["grip_physical_state"] for row in rows} >= {"open", "closed", "fixed"})

    open_current = next(row for row in rows if row["condition"] == "current_positive_pair" and row["grip_physical_state"] == "open")
    open_mirrored = next(row for row in rows if row["condition"] == "mirrored_signed_pair" and row["grip_physical_state"] == "open")
    check(
        "current_positive_pair and mirrored_signed_pair (same grip state) use the SAME image",
        open_current["main_image_hash"] == open_mirrored["main_image_hash"] and open_current["wrist_image_hash"] == open_mirrored["wrist_image_hash"],
    )
    check(
        "current_positive_pair and mirrored_signed_pair differ ONLY in gripper_qpos (state[6:8]), not ee_position/orientation",
        open_current["exact_input_state_8d"][0:6] == open_mirrored["exact_input_state_8d"][0:6],
        f"current={open_current['exact_input_state_8d'][0:6]} mirrored={open_mirrored['exact_input_state_8d'][0:6]}",
    )
    check(
        "mirrored_signed_pair's state[7] is the exact negation of current_positive_pair's",
        abs(open_current["exact_input_state_8d"][7] + open_mirrored["exact_input_state_8d"][7]) < 1e-9,
        f"current[7]={open_current['exact_input_state_8d'][7]} mirrored[7]={open_mirrored['exact_input_state_8d'][7]}",
    )
    # NOTE: robot_sim/pybullet_panda_backend.py's get_libero_observation_state()
    # now applies this experiment's confirmed fix directly in production
    # (negates its own second finger channel to match
    # HuggingFaceVLA/smolvla_libero's training convention) -- so real_q2
    # read from the backend is ALREADY negative, and
    # "current_positive_pair" (a plain [real_q1, real_q2] passthrough)
    # now reproduces the CORRECT [+, -] pair, while "mirrored_signed_pair"
    # ([real_q1, -real_q2]) now double-negates back to the OLD, buggy
    # [+, +] pair. The condition *names* describe the hypothesis this
    # diagnostic was built to test before the fix landed; their concrete
    # sign outcomes are naturally swapped now that production itself
    # applies the negation. What still matters here is unchanged: the
    # override reaches the policy and measurably changes its output.
    check(
        "real_q1/real_q2 reflect the now-fixed production backend (q1>0, q2<0) -- "
        "current_positive_pair passes them through unchanged",
        open_current["exact_input_state_8d"][6] > 0 and open_current["exact_input_state_8d"][7] < 0,
        f"got {open_current['exact_input_state_8d'][6:8]}",
    )
    check(
        "FakeServerPolicy's gripper-sign-dependent behavior actually differs between current (q2<0->open) "
        "and mirrored (q2>0->close, since mirroring the now-already-negative real_q2 flips it back positive) "
        "-- proves the override reaches the policy and changes its output",
        open_current["executed_gripper_command"] == "open" and open_mirrored["executed_gripper_command"] == "close",
        f"current={open_current['executed_gripper_command']} mirrored={open_mirrored['executed_gripper_command']}",
    )

    closed_current = next(row for row in rows if row["condition"] == "current_positive_pair" and row["grip_physical_state"] == "closed")
    check(
        "closed-gripper-state row's real gripper qpos differs from the open-gripper-state row's",
        closed_current["exact_input_state_8d"][6:8] != open_current["exact_input_state_8d"][6:8],
    )
    print()

    print("=== 3. compare_conditions() delta arithmetic ===")

    def make_row(position, instruction, grip_state, condition, seed, cosine, sign_x, sign_y, far_close):
        return {
            "position_name": position, "instruction_name": instruction, "grip_physical_state": grip_state,
            "condition": condition, "seed": seed, "cosine_commanded_vs_object": cosine,
            "sign_match_x": sign_x, "sign_match_y": sign_y, "far_gripper_close": far_close,
        }

    synthetic_rows = []
    for seed in (0, 42, 123):
        synthetic_rows.append(make_row("center_right", "ko_full", "open", "current_positive_pair", seed, 0.1, False, False, True))
        synthetic_rows.append(make_row("center_right", "ko_full", "open", "mirrored_signed_pair", seed, 0.6, True, True, False))
    comparison = compare_conditions(synthetic_rows)
    check("1 cell compared", comparison["num_cells"] == 1, f"got {comparison['num_cells']}")
    check("cosine_delta_mean is positive (B better than A)", comparison["cosine_delta_mean"] > 0, f"got {comparison['cosine_delta_mean']}")
    check("cosine_delta_mean is close to the expected 0.5", abs(comparison["cosine_delta_mean"] - 0.5) < 1e-9, f"got {comparison['cosine_delta_mean']}")
    check("xy_sign_accuracy_delta_mean is positive", comparison["xy_sign_accuracy_delta_mean"] > 0)
    check("far_gripper_close_rate_delta_mean is negative (fewer far closes in B)", comparison["far_gripper_close_rate_delta_mean"] < 0)
    check("reproducibility_fraction is 1.0 (the only cell improved)", comparison["reproducibility_fraction"] == 1.0)
    print()

    print("=== 4. judge_consistent_improvement()/judge_improvement_scope() ===")
    consistent_comparison = {"num_cells": 8, "cells_with_cosine_improvement": 7, "cosine_delta_mean": 0.3, "xy_sign_accuracy_delta_mean": 0.2, "far_gripper_close_rate_delta_mean": -0.2, "reproducibility_fraction": 7 / 8}
    result = judge_consistent_improvement(consistent_comparison)
    check("7/8 cells improving -> consistent improvement suspected", result["suspected"] is True, str(result))

    fluke_comparison = {"num_cells": 8, "cells_with_cosine_improvement": 1, "cosine_delta_mean": 0.05, "xy_sign_accuracy_delta_mean": 0.02, "far_gripper_close_rate_delta_mean": -0.01, "reproducibility_fraction": 1 / 8}
    result = judge_consistent_improvement(fluke_comparison)
    check("1/8 cells improving -> NOT consistent (possible fluke)", result["suspected"] is False, str(result))

    both_scope = judge_improvement_scope({"cosine_delta_mean": 0.3, "xy_sign_accuracy_delta_mean": 0.3, "far_gripper_close_rate_delta_mean": -0.3})
    check("large positive direction delta + large negative far-close delta -> scope='both'", both_scope["scope"] == "both", str(both_scope))

    gripper_only_scope = judge_improvement_scope({"cosine_delta_mean": 0.01, "xy_sign_accuracy_delta_mean": 0.0, "far_gripper_close_rate_delta_mean": -0.3})
    check("only far-close improves -> scope='gripper_only'", gripper_only_scope["scope"] == "gripper_only", str(gripper_only_scope))

    translation_only_scope = judge_improvement_scope({"cosine_delta_mean": 0.3, "xy_sign_accuracy_delta_mean": 0.3, "far_gripper_close_rate_delta_mean": 0.0})
    check("only direction improves -> scope='translation_only'", translation_only_scope["scope"] == "translation_only", str(translation_only_scope))

    neither_scope = judge_improvement_scope({"cosine_delta_mean": 0.01, "xy_sign_accuracy_delta_mean": 0.0, "far_gripper_close_rate_delta_mean": 0.0})
    check("no meaningful deltas -> scope='neither'", neither_scope["scope"] == "neither", str(neither_scope))
    print()

    print("=== 5. count_gripper_switches() ===")
    no_switch_rows = [{"step": 0, "executed_gripper_command": "open"}, {"step": 1, "executed_gripper_command": "open"}, {"step": 2, "executed_gripper_command": "open"}]
    check("no switches when the command never changes", count_gripper_switches(no_switch_rows) == 0)
    switch_rows = [{"step": 0, "executed_gripper_command": "open"}, {"step": 1, "executed_gripper_command": "close"}, {"step": 2, "executed_gripper_command": "open"}, {"step": 3, "executed_gripper_command": "open"}]
    check("2 switches detected (open->close->open)", count_gripper_switches(switch_rows) == 2, f"got {count_gripper_switches(switch_rows)}")
    print()

    print("=== 6. diagnostic-only functions never IMPORTED/CALLED from production ===")
    # Checks for an actual functional dependency (import statement or a
    # real function call, i.e. "apply_gripper_condition(") -- NOT a bare
    # filename mention. get_libero_observation_state()'s own docstring
    # legitimately references "benchmark/run_gripper_channel_sign_ab_experiment.py"
    # in plain prose for traceability (this is the diagnostic that
    # justified its gripper-sign fix) -- that is documentation, not a
    # code dependency, and must not trip this check.
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy"]
    hits = []
    dependency_patterns = (
        "import run_gripper_channel_sign_ab_experiment",
        "from benchmark.run_gripper_channel_sign_ab_experiment",
        "apply_gripper_condition(",
    )
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(pattern in text for pattern in dependency_patterns):
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    check("no production file imports/calls this diagnostic's functions", len(hits) == 0, f"unexpected references: {hits}")

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    check(
        "get_libero_observation_state() reflects production's own fix at reset (q1>0, q2<0) -- "
        "this diagnostic script itself applies no separate patch on top of it, it only reads what "
        "production already returns",
        backend.get_libero_observation_state()[6] > 0 and backend.get_libero_observation_state()[7] < 0,
    )
    backend.shutdown()
    print()

    print("=== 7. regression: prior state-semantics diagnostic suite ===")
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "benchmark.test_state_semantics_diagnostic"], capture_output=True, text=True, timeout=300)
    passed = "ALL CHECKS PASSED" in result.stdout
    check("benchmark.test_state_semantics_diagnostic -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
