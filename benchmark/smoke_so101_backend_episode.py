"""SO-101 backend episode smoke test (see this task's chat report).

Exercises robot_sim/so101_pybullet_backend.So101PyBulletBackend through
one full reset -> observe -> EE-delta approach -> gripper close/open ->
reset-again cycle, entirely standalone: no object/bin scene, no grasp-
success judgment, no SmolVLA/RealVLAPolicyClient involved. This is a
backend plumbing check, not a task-performance evaluation.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_backend_episode
"""

import argparse
import json
import math
from pathlib import Path

from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "backend_episode_smoke.json"

APPROACH_STEP_M = 0.01  # small, safe per-step Cartesian delta
NUM_APPROACH_STEPS = 5
JOINT_LIMIT_EPS = 1e-6


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def check_observation(obs: dict, backend: So101PyBulletBackend, events: list, label: str) -> bool:
    expected_keys = {
        "simulator", "joint_positions", "end_effector_position",
        "end_effector_orientation", "gripper_position_normalized", "gripper_position_radians",
    }
    missing = expected_keys - set(obs.keys())
    ok = True
    if missing:
        events.append({"stage": label, "issue": f"missing observation keys: {missing}"})
        ok = False
    if len(obs["joint_positions"]) != 5:
        events.append({"stage": label, "issue": f"expected 5 arm joint positions, got {len(obs['joint_positions'])}"})
        ok = False
    if not all_finite(obs["joint_positions"]):
        events.append({"stage": label, "issue": "non-finite joint_positions"})
        ok = False
    if not all_finite(obs["end_effector_position"]):
        events.append({"stage": label, "issue": "non-finite end_effector_position"})
        ok = False
    if not all_finite(obs["end_effector_orientation"]):
        events.append({"stage": label, "issue": "non-finite end_effector_orientation"})
        ok = False
    if not math.isfinite(obs["gripper_position_normalized"]) or not (0.0 <= obs["gripper_position_normalized"] <= 1.0):
        events.append({"stage": label, "issue": f"gripper_position_normalized out of [0,1]: {obs['gripper_position_normalized']}"})
        ok = False

    for name, pos in zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], obs["joint_positions"]):
        info = backend.joint_info_by_name[name]
        if pos < info["lower"] - JOINT_LIMIT_EPS or pos > info["upper"] + JOINT_LIMIT_EPS:
            events.append({"stage": label, "issue": f"joint '{name}' out of limit: {pos} not in [{info['lower']}, {info['upper']}]"})
            ok = False

    gripper_info = backend.joint_info_by_name["gripper"]
    if obs["gripper_position_radians"] < gripper_info["lower"] - JOINT_LIMIT_EPS or obs["gripper_position_radians"] > gripper_info["upper"] + JOINT_LIMIT_EPS:
        events.append({"stage": label, "issue": f"gripper out of limit: {obs['gripper_position_radians']}"})
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    events = []
    crashed = False
    crash_reason = None
    stage_results = {}

    backend = So101PyBulletBackend(gui=False)
    try:
        # 1. reset
        obs = backend.reset()
        stage_results["reset_ok"] = check_observation(obs, backend, events, "reset")
        stage_results["initial_gripper_normalized"] = obs["gripper_position_normalized"]
        initial_ee_position = obs["end_effector_position"]

        # 2. small safe EE delta move
        move_obs = backend.command_end_effector_delta([APPROACH_STEP_M, 0.0, 0.0])
        stage_results["single_delta_move_ok"] = check_observation(move_obs, backend, events, "single_delta_move")
        stage_results["single_delta_move_error_m"] = move_obs["ee_delta_position_error"]

        # 3. gripper open (explicit, even though reset() already opens it)
        open_obs = backend.set_gripper(1.0)
        stage_results["gripper_open_ok"] = check_observation(open_obs, backend, events, "gripper_open")
        stage_results["gripper_open_normalized"] = open_obs["gripper_position_normalized"]

        # 4. multi-step "approach" -- several successive small EE deltas
        approach_errors = []
        approach_positions = []
        for i in range(NUM_APPROACH_STEPS):
            delta = [0.0, 0.0, -APPROACH_STEP_M]  # descend toward a notional table
            step_obs = backend.command_end_effector_delta(delta)
            ok = check_observation(step_obs, backend, events, f"approach_step_{i}")
            stage_results.setdefault("approach_steps_ok", []).append(ok)
            approach_errors.append(step_obs["ee_delta_position_error"])
            approach_positions.append(step_obs["end_effector_position"])
        stage_results["approach_all_ok"] = all(stage_results["approach_steps_ok"])
        stage_results["approach_mean_error_m"] = sum(approach_errors) / len(approach_errors)
        stage_results["approach_final_ee_position"] = approach_positions[-1]

        # 5. gripper close
        close_obs = backend.set_gripper(0.0)
        stage_results["gripper_close_ok"] = check_observation(close_obs, backend, events, "gripper_close")
        stage_results["gripper_close_normalized"] = close_obs["gripper_position_normalized"]

        # 6. reset again -- must cleanly return to the same neutral state
        reset2_obs = backend.reset()
        stage_results["reset_again_ok"] = check_observation(reset2_obs, backend, events, "reset_again")
        stage_results["reset_again_ee_position"] = reset2_obs["end_effector_position"]
        stage_results["reset_again_matches_initial"] = math.sqrt(
            sum((reset2_obs["end_effector_position"][i] - initial_ee_position[i]) ** 2 for i in range(3))
        ) < 1e-6

    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    all_stage_checks = [v for k, v in stage_results.items() if k.endswith("_ok") and not isinstance(v, list)]
    all_stage_checks += stage_results.get("approach_steps_ok", [])
    passed = (not crashed) and all(all_stage_checks) and not events

    result = {
        "crashed": crashed, "crash_reason": crash_reason,
        "stage_results": stage_results, "issues": events,
        "all_passed": passed,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 backend episode smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    for k, v in stage_results.items():
        print(f"  {k}: {v}")
    print(f"issues: {events}")
    print(f"\n=== ALL PASSED: {passed} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
