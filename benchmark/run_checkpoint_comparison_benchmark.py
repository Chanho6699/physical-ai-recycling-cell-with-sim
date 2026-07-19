"""5-way checkpoint rollout comparison (A zero-shot LIBERO / B train20
200-step / C train80 500-step / D train80 1000-step / E train80
2000-step -- see this task's chat report). Reuses the SAME real
production path benchmark/run_zero_shot_vs_finetuned_comparison.py
already established (RealVLAPolicyClient -> vla_server ->
SmolVLALiberoActionAdapter -> PyBulletPandaBackend, real cameras/state/
ActionAdapter) -- this script only adds: (a) a fixed, shared
train-distribution + validation-distribution position/seed set from
benchmark/checkpoint_eval_positions.py so every one of the 5 models is
evaluated against byte-identical conditions, and (b) richer per-step
logging (object_position, ee_position, phase, first-close-step/distance,
workspace violation, timeout) needed for this task's trajectory
diagnosis and metric requirements.

Run ONCE PER CHECKPOINT against a server already /load_model'd with
that checkpoint (see this task's chat report for the exact env vars):

  .venv-vla/bin/python -m benchmark.run_checkpoint_comparison_benchmark \\
    --label C_train80_500step --output results/checkpoint_comparison/C_train80_500step.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.checkpoint_eval_positions import build_train_eval_positions, build_validation_eval_positions
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d, parse_workspace_bounds
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]
DEFAULT_MAX_POLICY_STEPS = 40
DEFAULT_STEPS_PER_ACTION = 40
DEFAULT_WORKSPACE_BOUNDS_STR = "-0.1,0.9,-0.7,0.7,0.0,1.0"  # matches run_full_recycling_cell_demo.py's own default


def _classify_failure_reason(final_status: str, ever_held: bool, timed_out: bool) -> str:
    if final_status == "success":
        return "none (success)"
    if final_status == "released":
        return "released_away_from_bin"
    if final_status == "grasped":
        return "grasped_then_timeout_before_release"
    if ever_held:
        return "held_then_lost_track"
    if timed_out:
        return "never_grasped_timeout"
    return "never_grasped_other"


def run_episode(
    policy, split_tag, anchor_name, position, seed, instruction, instruction_name, bin_position,
    max_steps, steps_per_action, object_type, strict, label, workspace_bounds,
) -> dict:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    backend.set_object_position(list(position))
    policy.reset()

    x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds

    step_rows = []
    first_distance_to_object = None
    ever_held = False
    first_grasp_step = None
    first_close_command_step = None
    distance_at_first_close_command = None
    release_step = None
    final_status = "running"
    success = False
    workspace_violations = 0

    for step_index in range(max_steps):
        robot_state, _state_8d, object_position = build_robot_state(backend)
        ee_position = list(robot_state["ee_position"])
        distance_to_object = _distance_3d(ee_position, object_position)
        if first_distance_to_object is None:
            first_distance_to_object = distance_to_object

        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()

        policy_input = PolicyInput(
            image=main_image,
            instruction=instruction,
            robot_state=robot_state,
            task_goal={},
            target_object_position=object_position,
            bin_position=bin_position,
            step_index=step_index,
            phase=policy.phase,
            images_by_role={"main": main_image, "wrist": wrist_image},
            seed=seed,
        )
        policy_output = policy.predict_action(policy_input)
        info = policy_output.info or {}

        compatibility_passed = (info.get("compatibility") or {}).get("passed")
        semantic_action_valid = bool(info.get("semantic_action_valid", True))
        degraded_input = bool(info.get("degraded_input", False))
        fallback_used = bool(info.get("fallback_used", False))
        if strict:
            violations = []
            if compatibility_passed is not True:
                violations.append(f"compatibility.passed={compatibility_passed!r}")
            if not semantic_action_valid:
                violations.append("semantic_action_valid=False")
            if degraded_input:
                violations.append("degraded_input=True")
            if fallback_used:
                violations.append("fallback_used=True")
            if violations:
                raise RuntimeError(
                    f"--strict violated at label={label} anchor={anchor_name} seed={seed} step={step_index}: "
                    f"{'; '.join(violations)}. info={info}"
                )

        action_adapter = ActionAdapter()
        robot_command = action_adapter.convert(policy_output.action)
        commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]
        vector_to_object = [object_position[i] - ee_position[i] for i in range(3)]
        cosine_commanded_vs_object = _cosine_similarity(commanded_translation, vector_to_object)

        if robot_command.gripper_command == "close" and first_close_command_step is None:
            first_close_command_step = step_index
            distance_at_first_close_command = distance_to_object

        robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
        ee_position_after = list(robot_state_after["end_effector_position"])
        final_status = robot_state_after["task_status"]
        held_now = bool(robot_state_after["held_object"])
        if held_now and not ever_held:
            ever_held = True
            first_grasp_step = step_index
        if ever_held and not held_now and release_step is None and step_index != first_grasp_step:
            release_step = step_index

        in_bounds = (
            x_min <= ee_position_after[0] <= x_max
            and y_min <= ee_position_after[1] <= y_max
            and z_min <= ee_position_after[2] <= z_max
        )
        if not in_bounds:
            workspace_violations += 1

        step_rows.append({
            "step": step_index,
            "phase": policy_output.phase,
            "object_position": list(object_position),
            "ee_position": list(ee_position),
            "distance_to_object": distance_to_object,
            "predicted_translation_m": commanded_translation,
            "simulator_command_translation_m": commanded_translation,  # identical by design post scale-fix (see chat report)
            "cosine_commanded_vs_object": cosine_commanded_vs_object,
            "gripper_command": robot_command.gripper_command,
            "held_object": held_now,
            "task_status": final_status,
            "in_workspace_bounds": in_bounds,
            "inference_latency_ms": info.get("inference_latency_ms"),
            "server_inference_ms": info.get("server_inference_ms"),
            "main_image_hash": image_hash(main_image),
        })

        if final_status == "success":
            success = True
            break

    final_robot_state, _, final_object_position = build_robot_state(backend)
    final_distance_to_object = _distance_3d(final_robot_state["ee_position"], final_object_position)
    backend.shutdown()

    cosines = [r["cosine_commanded_vs_object"] for r in step_rows if r["cosine_commanded_vs_object"] is not None]
    latencies = [r["inference_latency_ms"] for r in step_rows if r["inference_latency_ms"] is not None]
    gripper_open = sum(1 for r in step_rows if r["gripper_command"] == "open")
    gripper_close = sum(1 for r in step_rows if r["gripper_command"] == "close")
    timed_out = (not success) and len(step_rows) >= max_steps

    return {
        "label": label,
        "split_tag": split_tag,
        "anchor_name": anchor_name,
        "position": list(position),
        "instruction_name": instruction_name,
        "seed": seed,
        "num_steps": len(step_rows),
        "success": success,
        "final_task_status": final_status,
        "pick_success": ever_held,
        "first_grasp_step": first_grasp_step,
        "release_step": release_step,
        "first_close_command_step": first_close_command_step,
        "distance_at_first_close_command": distance_at_first_close_command,
        "first_distance_to_object": first_distance_to_object,
        "final_distance_to_object": final_distance_to_object,
        "distance_improvement": (first_distance_to_object - final_distance_to_object) if first_distance_to_object is not None else None,
        "mean_cosine_commanded_vs_object": sum(cosines) / len(cosines) if cosines else None,
        "gripper_open_count": gripper_open,
        "gripper_close_count": gripper_close,
        "gripper_close_ratio": gripper_close / len(step_rows) if step_rows else None,
        "workspace_violations": workspace_violations,
        "timed_out": timed_out,
        "retry_count": 0,
        "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "failure_reason": _classify_failure_reason(final_status, ever_held, timed_out),
        "rows": step_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--real-vla-config", type=str, default="configs/real_vla_backend_config.json")
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_MAX_POLICY_STEPS)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]
    workspace_bounds = parse_workspace_bounds(DEFAULT_WORKSPACE_BOUNDS_STR)

    train_eval_positions = build_train_eval_positions()
    validation_eval_positions = build_validation_eval_positions()

    policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)
    health = policy.check_health()
    print(f"=== Checkpoint comparison -- label={args.label!r} ===")
    print(f"server health: {health}")
    if health.get("model_status") != "loaded":
        raise RuntimeError(f"Server model_status={health.get('model_status')!r}, expected 'loaded'.")
    if not (health.get("compatibility") or {}).get("passed"):
        raise RuntimeError(f"Server compatibility.passed is not True: {health.get('compatibility')}")
    model_id_or_path = health.get("model_id_or_path")
    print(f"model_id_or_path: {model_id_or_path}")
    print(f"train_eval episodes: {len(train_eval_positions)}, validation_eval episodes: {len(validation_eval_positions)}")

    episodes = []
    all_positions = [("train_distribution", p) for p in train_eval_positions] + [
        ("validation_distribution", p) for p in validation_eval_positions
    ]
    start_time = time.time()
    for n, (split_tag, p) in enumerate(all_positions, start=1):
        episode = run_episode(
            policy, split_tag, p["anchor_name"], p["position"], p["seed"], instruction, args.instruction_name,
            DEFAULT_BIN_POSITION, args.max_policy_steps, args.steps_per_action, args.object_type, args.strict,
            args.label, workspace_bounds,
        )
        episodes.append(episode)
        print(
            f"[{n:02d}/{len(all_positions)}] split={split_tag:20s} anchor={p['anchor_name']:16s} seed={p['seed']:7d} "
            f"success={episode['success']} status={episode['final_task_status']:<10} steps={episode['num_steps']:3d} "
            f"pick={episode['pick_success']} dist_improve={episode['distance_improvement']:.4f} "
            f"mean_cos={episode['mean_cosine_commanded_vs_object']}"
        )

    elapsed_s = time.time() - start_time
    result = {
        "label": args.label,
        "model_id_or_path": model_id_or_path,
        "server_health_at_start": health,
        "instruction_name": args.instruction_name,
        "instruction": instruction,
        "max_policy_steps": args.max_policy_steps,
        "steps_per_action": args.steps_per_action,
        "object_type": args.object_type,
        "strict": args.strict,
        "num_episodes": len(episodes),
        "wall_clock_s": elapsed_s,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    success_rate = sum(1 for e in episodes if e["success"]) / len(episodes)
    print(f"\n=== Done: {len(episodes)} episodes, success_rate={success_rate:.2%}, wall_clock={elapsed_s:.1f}s ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
