"""Tests for this task's quantitative-behavior-eval features (v0):

  1. distance-to-object/distance-to-bin calculation accuracy
  2. per-step structured action log is actually produced (raw model
     action, postprocessed action, canonical command before/after the
     safety filter, gripper command, action delta/repetition)
  3. the Korean and English eval runs are launched with identical CLI
     args except --instruction/--eval-log-path (same scene, same seed,
     same step budget, same strict-mode enforcement)
  4. --strict-real-vla still allows no fallback (re-verified with the
     new fields threaded through this turn)
  5. --max-policy-steps and --control-loop-timeout-s are both enforced
  6. an EE position outside --workspace-bounds ends the episode
     immediately with task_status="aborted_workspace_exceeded" (not a
     silent continue, not marked as success)
  7. regression: the closed-loop test suite from the previous turn
     still passes unmodified

Run: python -m benchmark.test_language_eval_and_guards
"""

import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark.run_smolvla_language_comparison_eval import build_cmd
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

import benchmark.run_full_recycling_cell_demo as demo
from benchmark.test_full_demo_closed_loop import FakeRecorder, make_args, make_task_goal

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FixedActionPolicy(BasePolicy):
    """Always returns the same canned, "clean" (non-degraded, non-fallback,
    compatibility-passed) response with a controllable translation
    direction -- for the workspace/runaway-guard and per-step-log-content
    tests below, where what matters is that the SAME action is applied
    every step (to deterministically drive the EE somewhere or to trigger
    the runaway-direction guard), not any particular model's behavior."""

    def __init__(self, translation=(0.05, 0.0, 0.0), gripper=-1.0, steps_before_done=None):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []
        self._translation = translation
        self._gripper = gripper
        self._steps_before_done = steps_before_done
        self._step_count = 0

    def reset(self) -> None:
        self.phase = "move_to_object"
        self.received_inputs = []
        self._step_count = 0

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        self._step_count += 1
        action = [self._translation[0], self._translation[1], self._translation[2], 0.0, 0.0, 0.0, self._gripper]
        done = self._steps_before_done is not None and self._step_count >= self._steps_before_done
        info = {
            "policy_backend": "real-vla",
            "model_family": "smolvla",
            "inference_latency_ms": 10.0,
            "fallback_used": False,
            "real_vla_request_failed": False,
            "compatibility": {"passed": True},
            "semantic_action_valid": True,
            "degraded_input": False,
            "action_postprocess": {
                "canonical_command": {
                    "translation_m": list(self._translation),
                    "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                    "gripper_opening_01": 0.0 if self._gripper >= 0.5 else 1.0,
                    "metadata": {
                        "raw_model_action": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper],
                        "native_action_raw_values": [0.85, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper],
                    },
                },
                "canonical_command_pre_safety_filter": {
                    "translation_m": [self._translation[0] * 1.5, self._translation[1], self._translation[2]],
                    "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                    "gripper_opening_01": 0.0 if self._gripper >= 0.5 else 1.0,
                },
                "safety_filter_clipped": True,
                "safety_clipped": True,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=done, info=info)


def run_loop(policy, args_overrides: dict, recorder=None):
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    args = make_args(**args_overrides)
    task_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    original_create_policy_backend = demo.create_policy_backend
    demo.create_policy_backend = lambda _args: policy
    try:
        final_state, steps_run = demo.run_dummy_openvla_policy(
            args,
            backend,
            safety_gate=None,
            recorder=recorder,
            task_frame=task_frame,
            task_goal=make_task_goal(),
            sim_position=[0.5, 0.0, 0.05],
            bin_position=[0.3, 0.3, 0.05],
        )
    finally:
        demo.create_policy_backend = original_create_policy_backend
        backend.shutdown()
    return final_state, steps_run


def main() -> None:
    print("=== 1. distance-to-object/bin calculation accuracy ===")
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    state = backend.get_state()
    ee_position = state["end_effector_position"]
    object_position = state["object_position"]
    bin_position = [0.3, 0.35, 0.05]
    expected_object_distance = math.sqrt(sum((ee_position[i] - object_position[i]) ** 2 for i in range(3)))
    expected_bin_distance = math.sqrt(sum((ee_position[i] - bin_position[i]) ** 2 for i in range(3)))
    computed_object_distance = demo._distance_3d(list(ee_position), list(object_position))
    computed_bin_distance = demo._distance_3d(list(ee_position), bin_position)
    check(
        "distance_to_object matches manual Euclidean calculation",
        abs(computed_object_distance - expected_object_distance) < 1e-9,
        f"expected={expected_object_distance} got={computed_object_distance}",
    )
    check(
        "distance_to_bin matches manual Euclidean calculation",
        abs(computed_bin_distance - expected_bin_distance) < 1e-9,
        f"expected={expected_bin_distance} got={computed_bin_distance}",
    )
    known_a, known_b = [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]
    check("_distance_3d matches a known 3-4-5 triangle", demo._distance_3d(known_a, known_b) == 5.0)
    backend.shutdown()
    print()

    print("=== 2. per-step structured action log content ===")
    policy = FixedActionPolicy(translation=(0.02, 0.0, 0.0), gripper=1.0)
    recorder = FakeRecorder()
    run_loop(policy, {"max_policy_steps": 2}, recorder=recorder)
    logs = [s["extra"]["real_vla_step_log"] for s in recorder.steps if s.get("extra") and "real_vla_step_log" in s["extra"]]
    check("2 step logs recorded", len(logs) == 2, f"got {len(logs)}")
    first_log = logs[0]
    for field in (
        "raw_model_action_7d", "postprocessed_action_7d", "applied_action_7d",
        "canonical_command_before_safety_filter", "canonical_command_after_safety_filter",
        "safety_filter_clipped", "ee_position", "ee_orientation_axis_angle",
        "distance_to_object_m", "distance_to_bin_m", "gripper_command",
        "action_delta_norm", "action_repeated", "post_apply_ee_position", "post_apply_gripper_width",
    ):
        check(f"real_vla_step_log has '{field}'", field in first_log, f"keys={sorted(first_log.keys())}")
    check("gripper_command reflects the close command (gripper=1.0 -> close)", first_log["gripper_command"] == "close")
    check(
        "canonical_command_before_safety_filter differs from after (pre-clip vs. post-clip)",
        logs[0]["canonical_command_before_safety_filter"]["translation_m"]
        != logs[0]["canonical_command_after_safety_filter"]["translation_m"],
    )
    check("first step has action_delta_norm=None (nothing to compare against yet)", logs[0]["action_delta_norm"] is None)
    check(
        "second step's action_delta_norm is ~0 (FixedActionPolicy repeats the same action)",
        logs[1]["action_delta_norm"] is not None and logs[1]["action_delta_norm"] < 1e-6,
    )
    check("second step is marked action_repeated=True", logs[1]["action_repeated"] is True)
    print()

    print("=== 3. Korean/English eval runs are launched with identical args (except instruction/log path) ===")
    args = SimpleNamespace(
        real_vla_config="configs/vla_backend_smolvla_libero_config.json",
        image_path="data/test_images/recyclable_scene.jpg",
        max_policy_steps=20,
        control_loop_timeout_s=300.0,
        steps_per_action=10,
        seed=42,
        allow_fallback=False,
    )
    ko_cmd = build_cmd(args, "플라스틱 병을 플라스틱 수거함에 넣어줘", Path("results/x/ko_steps.jsonl"))
    en_cmd = build_cmd(args, "Pick up the plastic bottle and place it in the plastic bin.", Path("results/x/en_steps.jsonl"))

    def strip_instruction_and_log(cmd):
        stripped = list(cmd)
        for flag in ("--instruction", "--eval-log-path"):
            index = stripped.index(flag)
            stripped[index + 1] = "<REDACTED>"
        return stripped

    check(
        "KO/EN command lines are identical apart from --instruction/--eval-log-path",
        strip_instruction_and_log(ko_cmd) == strip_instruction_and_log(en_cmd),
        f"ko={ko_cmd}\nen={en_cmd}",
    )
    check("--strict-real-vla is present by default (allow_fallback=False)", "--strict-real-vla" in ko_cmd)
    check("--seed 42 present in both", "42" in ko_cmd and "42" in en_cmd)
    print()

    print("=== 4. --strict-real-vla still forces fallback_policy=None (regression, new fields don't break this) ===")
    strict_args = make_args(strict_real_vla=True, real_vla_fallback_backend="local-dummy")
    strict_policy = demo.create_policy_backend(strict_args)
    check("fallback_policy is None under --strict-real-vla", strict_policy.fallback_policy is None)
    print()

    print("=== 5. --max-policy-steps and --control-loop-timeout-s are enforced ===")
    policy_maxsteps = FixedActionPolicy(translation=(0.001, 0.0, 0.0))
    final_state, steps_run = run_loop(policy_maxsteps, {"max_policy_steps": 3})
    check("loop stops at max_policy_steps=3", steps_run == 3, f"got {steps_run}")

    policy_timeout = FixedActionPolicy(translation=(0.001, 0.0, 0.0))
    timeout_raised = False
    try:
        run_loop(policy_timeout, {"max_policy_steps": 10_000, "control_loop_timeout_s": 0.01})
    except RuntimeError as exc:
        timeout_raised = True
        check("timeout RuntimeError mentions control-loop-timeout-s", "timeout" in str(exc).lower(), str(exc)[:200])
    check("--control-loop-timeout-s aborts a run that would otherwise spin", timeout_raised)
    print()

    print("=== 6. EE leaving --workspace-bounds ends the episode immediately ===")
    policy_runaway = FixedActionPolicy(translation=(0.05, 0.0, 0.0))
    final_state, steps_run = run_loop(
        policy_runaway,
        {
            "max_policy_steps": 20,
            # Tiny box guaranteed to be exceeded almost immediately by a
            # real Panda EE (~0.3m from origin at reset already).
            "workspace_bounds": "-0.01,0.01,-0.01,0.01,-0.01,0.01",
            "runaway_window": 0,
        },
    )
    check(
        "task_status is aborted_workspace_exceeded",
        final_state["task_status"] == "aborted_workspace_exceeded",
        f"task_status={final_state['task_status']}",
    )
    check("run ended well before max_policy_steps=20", steps_run < 20, f"steps_run={steps_run}")
    check("aborted run is not disguised as a success", final_state["task_status"] != "success")

    print("--- runaway-same-direction guard ---")
    # +Y, not +X: the sim object sits at y~=0 (same as the EE's starting
    # y), directly in front of the EE only along +X -- a repeated +X
    # command would keep approaching it (as step 6 above shows), which
    # would never trigger this guard and wouldn't actually be a runaway.
    # +Y moves directly away from the object every step, giving a command
    # that is both same-direction AND non-approaching, i.e. the actual
    # failure mode this guard exists to catch.
    policy_same_direction = FixedActionPolicy(translation=(0.0, 0.02, 0.0))
    final_state_runaway, steps_run_runaway = run_loop(
        policy_same_direction,
        {
            "max_policy_steps": 20,
            "workspace_bounds": "-2.0,2.0,-2.0,2.0,-2.0,2.0",  # effectively disabled
            "runaway_window": 3,
        },
    )
    check(
        "task_status is aborted_runaway_same_direction (same-direction, non-approaching command repeated)",
        final_state_runaway["task_status"] == "aborted_runaway_same_direction",
        f"task_status={final_state_runaway['task_status']}",
    )
    check("runaway guard also ends before max_policy_steps=20", steps_run_runaway < 20, f"steps_run={steps_run_runaway}")
    print()

    print("=== 7. regression: prior closed-loop test suite ===")
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.test_full_demo_closed_loop"], capture_output=True, text=True, timeout=300
    )
    passed = "ALL CHECKS PASSED" in result.stdout
    check("benchmark.test_full_demo_closed_loop -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
