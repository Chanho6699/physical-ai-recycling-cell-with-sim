"""Closed-loop wiring tests for run_full_recycling_cell_demo.py (v0).

Covers this task's 9 required scenarios:

  1-4, 6 (state/images real + change step-to-step, fallback_used) --
        exercised by calling the REAL run_dummy_openvla_policy() control
        loop against a REAL PyBulletPandaBackend, with only the network
        boundary (create_policy_backend()'s RealVLAPolicyClient) swapped
        for a FakeServerPolicy stub that mimics an already-verified real
        HuggingFaceVLA/smolvla_libero server response (see
        benchmark/test_libero_real_observation.py, which already proved
        that response is achievable against a real GPU server). This is a
        wiring test -- it proves run_dummy_openvla_policy() builds/sends
        real per-step observations and correctly threads the response
        through, not that any particular model produces a good action.
  5    (degraded_input=false reaches the loop) -- same mechanism.
  7    (strict mode fails immediately on server failure) -- exercised
        against the REAL RealVLAPolicyClient (no stub) pointed at a
        server URL nothing is listening on, proving create_policy_backend
        actually forces fallback_policy=None under --strict-real-vla and
        that the resulting RuntimeError is not swallowed.
  8    (existing mock/local-dummy path regression) -- runs the actual
        script end-to-end as a subprocess with --policy-backend
        local-dummy, confirming this turn's changes didn't break it.
  9    (existing suite regression) -- re-runs the three prior suites.

Run: python -m benchmark.test_full_demo_closed_loop
"""

import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from action_adapter.adapter_v0 import ActionAdapter
from llm_agent.rule_based_parser import TaskGoal
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

import benchmark.run_full_recycling_cell_demo as demo

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Stands in for create_policy_backend()'s RealVLAPolicyClient. Records
    every PolicyInput it receives (so the test can inspect exactly what
    run_dummy_openvla_policy() built and sent) and returns a canned
    response shaped exactly like a real, non-degraded, non-fallback
    HuggingFaceVLA/smolvla_libero server reply -- the same info shape
    SmolVLAActionAdapter._accept() produces (see vla_adapters/
    smolvla_adapter.py), so downstream field-name assumptions are real,
    not invented for this test."""

    def __init__(self, degraded_input: bool = False, fallback_used: bool = False, compatibility_passed: bool = True):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []
        self._degraded_input = degraded_input
        self._fallback_used = fallback_used
        self._compatibility_passed = compatibility_passed

    def reset(self) -> None:
        self.phase = "move_to_object"
        self.received_inputs = []

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # small +X move, gripper open
        info = {
            "policy_backend": "real-vla",
            "model_family": "smolvla",
            "inference_latency_ms": 12.5,
            "fallback_used": self._fallback_used,
            "real_vla_request_failed": False,
            "compatibility": {"passed": self._compatibility_passed},
            "semantic_action_valid": True,
            "degraded_input": self._degraded_input,
            "action_postprocess": {
                "canonical_command": {"frame": "robot_base", "translation": [0.01, 0.0, 0.0]},
                "safety_clipped": False,
                "degraded_input": self._degraded_input,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


class FakeRecorder:
    def __init__(self):
        self.steps = []

    def record_step(self, **kwargs):
        self.steps.append(kwargs)


def make_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        instruction="pick up the bottle",
        policy_backend="real-vla",
        policy_server_url="http://127.0.0.1:8000/predict",
        policy_request_timeout=5.0,
        real_vla_config="configs/vla_backend_smolvla_libero_config.json",
        real_vla_fallback_backend="none",
        real_vla_observation_mode="pybullet",
        strict_real_vla=False,
        control_loop_timeout_s=120.0,
        policy_observation_source="none",
        record_policy_observations=False,
        record_images=False,
        policy_observation_save_interval=5,
        max_policy_steps=3,
        steps_per_action=15,
        max_step_size=0.03,
        position_tolerance=0.03,
        carry_height=0.18,
        grasp_z_offset=0.015,
        safety_mode="off",
        hand_safety_source="none",
        safety_resume_stable_steps=3,
        save_wrist_camera_images=False,
        save_hand_safety_debug_images=False,
        wrist_camera_mode="off",
        wrist_refinement_policy="blend",
        wrist_refinement_alpha=0.7,
        refine_distance_threshold=0.08,
        wrist_min_object_pixels=50,
        wrist_max_refinement_delta=0.08,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_task_goal() -> TaskGoal:
    return TaskGoal(
        task="pick_and_place",
        target_object="plastic_bottle",
        target_bin="plastic_bin",
        vla_instruction="pick up the bottle",
        success_condition="object_in_bin",
        raw_user_command="pick up the bottle",
    )


def run_loop_with_fake_server(fake_policy: FakeServerPolicy, args_overrides: dict, recorder=None):
    backend = PyBulletPandaBackend(gui=False)
    state = backend.reset()
    args = make_args(**args_overrides)
    task_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    original_create_policy_backend = demo.create_policy_backend
    demo.create_policy_backend = lambda _args: fake_policy
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
    print("=== 1-2. real 8D state + real main/wrist images reach PolicyInput ===")
    fake_policy = FakeServerPolicy()
    run_loop_with_fake_server(fake_policy, {"max_policy_steps": 3})
    inputs = fake_policy.received_inputs
    check("3 steps were actually sent to the (fake) server", len(inputs) == 3, f"got {len(inputs)}")

    first = inputs[0]
    check("images_by_role has both 'main' and 'wrist' keys", set(first.images_by_role or {}) == {"main", "wrist"})
    main_img = first.images_by_role["main"]
    wrist_img = first.images_by_role["wrist"]
    check(
        "main/wrist images are real (256,256,3) uint8 arrays",
        main_img.shape == (256, 256, 3) and main_img.dtype == np.uint8
        and wrist_img.shape == (256, 256, 3) and wrist_img.dtype == np.uint8,
        f"main.shape={main_img.shape} wrist.shape={wrist_img.shape}",
    )
    check("main and wrist images are not identical", not np.array_equal(main_img, wrist_img))

    state_keys = {"ee_position", "ee_orientation_axis_angle", "gripper_qpos"}
    check("robot_state carries the 3 LIBERO 8D-state fields", state_keys.issubset(first.robot_state.keys()))
    flat_8d = list(first.robot_state["ee_position"]) + list(first.robot_state["ee_orientation_axis_angle"]) + list(
        first.robot_state["gripper_qpos"]
    )
    check("flattened state is exactly 8 floats", len(flat_8d) == 8, f"len={len(flat_8d)}")
    print()

    print("=== 3. 8D state changes at the step after a command is applied ===")
    state_step0 = list(inputs[0].robot_state["ee_position"])
    state_step1 = list(inputs[1].robot_state["ee_position"])
    check(
        "ee_position differs between step 0 and step 1 (a real move was applied)",
        state_step0 != state_step1,
        f"step0={state_step0} step1={state_step1}",
    )
    print()

    print("=== 4. wrist image also changes as the robot moves ===")
    wrist_step0 = inputs[0].images_by_role["wrist"]
    wrist_step_last = inputs[-1].images_by_role["wrist"]
    check("wrist image at the last step differs from the first step", not np.array_equal(wrist_step0, wrist_step_last))
    print()

    print("=== 5-6. degraded_input=False / fallback_used=False reach the loop's step log ===")
    fake_policy2 = FakeServerPolicy(degraded_input=False, fallback_used=False, compatibility_passed=True)
    recorder = FakeRecorder()
    run_loop_with_fake_server(fake_policy2, {"max_policy_steps": 2, "strict_real_vla": True}, recorder=recorder)
    step_logs = [s["extra"]["real_vla_step_log"] for s in recorder.steps if s.get("extra") and "real_vla_step_log" in s["extra"]]
    check("real_vla_step_log was attached every recorded step", len(step_logs) == 2, f"got {len(step_logs)}")
    check("degraded_input=False in every step log", all(not log["degraded_input"] for log in step_logs))
    check("fallback_used=False in every step log", all(not log["fallback_used"] for log in step_logs))
    check("semantic_action_valid=True in every step log", all(log["semantic_action_valid"] for log in step_logs))
    check("compatibility_passed=True in every step log", all(log["compatibility_passed"] is True for log in step_logs))
    check(
        "observation_repeated=False after step 0 (fresh observation every step)",
        all(not log["observation_repeated"] for log in step_logs),
    )
    check(
        "post_apply_ee_position recorded (proves command was actually applied before logging)",
        all(log.get("post_apply_ee_position") is not None for log in step_logs),
    )
    check(
        "--strict-real-vla does NOT raise when the response is genuinely clean",
        True,  # reaching this line at all means run_loop_with_fake_server didn't raise
    )
    print()

    print("=== 7a. --strict-real-vla raises immediately when the response is degraded ===")
    fake_policy3 = FakeServerPolicy(degraded_input=True)
    raised = False
    try:
        run_loop_with_fake_server(fake_policy3, {"max_policy_steps": 5, "strict_real_vla": True})
    except RuntimeError as exc:
        raised = True
        check("RuntimeError mentions degraded_input", "degraded_input" in str(exc), str(exc)[:200])
    check("--strict-real-vla raised RuntimeError on a degraded response", raised)
    print()

    print("=== 7b. --strict-real-vla forces fallback_policy=None + raises on real server-connection failure ===")
    args = make_args(
        real_vla_config="configs/vla_backend_smolvla_libero_config.json",
        real_vla_fallback_backend="local-dummy",  # deliberately mismatched -- strict must override this
        strict_real_vla=True,
    )
    policy = demo.create_policy_backend(args)
    check("create_policy_backend forces fallback_policy=None under --strict-real-vla", policy.fallback_policy is None)

    unreachable_args = make_args(
        real_vla_config="configs/vla_backend_smolvla_libero_config.json",
        real_vla_fallback_backend="none",
        strict_real_vla=True,
    )
    unreachable_policy = demo.create_policy_backend(unreachable_args)
    unreachable_policy.server_url = "http://127.0.0.1:59999/predict"  # nothing listens here
    task_goal = make_task_goal()
    policy_input = PolicyInput(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        instruction="pick up the bottle",
        robot_state={},
        task_goal=asdict(task_goal),
        target_object_position=[0.5, 0.0, 0.05],
        bin_position=[0.3, 0.3, 0.05],
        step_index=0,
        phase="move_to_object",
    )
    server_failure_raised = False
    try:
        unreachable_policy.predict_action(policy_input)
    except RuntimeError as exc:
        server_failure_raised = True
        check("RuntimeError explains no fallback is configured", "fallback" in str(exc).lower(), str(exc)[:200])
    check(
        "real RealVLAPolicyClient (fallback_policy=None) raises immediately on connection failure",
        server_failure_raised,
    )
    print()

    print("=== 8. regression: existing --policy-backend local-dummy path still runs end-to-end ===")
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmark.run_full_recycling_cell_demo",
            "--policy", "dummy-openvla", "--policy-backend", "local-dummy",
            "--image-path", "data/test_images/recyclable_scene.jpg",
            "--headless", "--max-policy-steps", "5",
        ],
        capture_output=True, text=True, timeout=120,
    )
    check(
        "local-dummy demo run exits 0",
        result.returncode == 0,
        f"returncode={result.returncode} stderr_tail={result.stderr[-800:]}",
    )
    print()

    print("=== 9. regression: prior policy_semantics / rotation / mock-action / real-observation suites ===")
    for module in (
        "benchmark.test_policy_semantics",
        "benchmark.test_smolvla_libero_action_adapter",
        "benchmark.test_panda_rotation_and_capability",
        "benchmark.test_libero_real_observation",
    ):
        result = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True, timeout=300)
        passed = "ALL CHECKS PASSED" in result.stdout
        check(f"{module} -- ALL CHECKS PASSED", passed, result.stdout[-800:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
