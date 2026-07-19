"""Zero-shot vs Fine-tuned SmolVLA comparison benchmark (v0).

Runs a fixed batch of full pick-and-place episodes against whichever
SmolVLA checkpoint the currently-running vla_server is loaded with (see
vla_server/model_loader.py) -- through the EXACT same production path
benchmark/run_full_recycling_cell_demo.py's --policy-backend real-vla
--real-vla-observation-mode pybullet --strict-real-vla uses
(RealVLAPolicyClient -> PyBulletPandaBackend, real main/wrist cameras,
real 8D state, real ActionAdapter.convert()) -- reusing that path's own
already-verified helper functions rather than reimplementing any of
them, the same reuse pattern
benchmark/run_ee_position_offset_ab_experiment.py already established:

  run_full_recycling_cell_demo._cosine_similarity / ._distance_3d
  run_vla_action_direction_diagnostic.build_robot_state / .image_hash / .resolve
  run_counterfactual_direction_benchmark.DEFAULT_POSITIONS / ._sign_match
  collect_recycling_dataset.jitter_position / .DEFAULT_INSTRUCTIONS

Only NEW code here is run_episode() (a full closed loop to success/
max-steps, unlike the offset experiment's fixed 5-step probe -- modeled
on collect_recycling_dataset.run_one_episode()'s structure) and this
file's own metric bookkeeping (pick/place success, gripper timing,
failure-reason bucketing).

This script does NOT switch checkpoints itself -- the server is loaded
with exactly one checkpoint at a time (see vla_server/model_loader.py's
env-var-driven loading, and this task's chat report for how the local
fine-tuned checkpoint gets loaded). Run this script ONCE PER CHECKPOINT,
against a server already loaded with that checkpoint, with the SAME
--seeds/--positions/--instruction/--max-policy-steps both times so the
two result files are directly comparable. See
benchmark/analyze_zero_shot_vs_finetuned_comparison.py for the actual
comparison table/statistics.

Retry count: RealVLAPolicyClient (policy/real_vla_policy_client.py) has
NO retry logic at all -- any request failure goes straight to
_fallback() (raises under --strict, since this script always runs
strict). So "retry count" is always 0 for every episode here; reported
as a constant, not measured, per that source-level confirmation.

Run (server must already be up and /load_model'd for ONE checkpoint --
see this task's chat report for the exact env vars):

  .venv-vla/bin/python -m benchmark.run_zero_shot_vs_finetuned_comparison \\
    --label zero_shot \\
    --output results/zero_shot_vs_finetuned/zero_shot.json

  .venv-vla/bin/python -m benchmark.run_zero_shot_vs_finetuned_comparison \\
    --label fine_tuned \\
    --output results/zero_shot_vs_finetuned/fine_tuned.json
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, jitter_position
from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS, _sign_match
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/real_vla_backend_config.json"
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MAX_POLICY_STEPS = 40
DEFAULT_STEPS_PER_ACTION = 40  # matches collect_recycling_dataset.DEFAULT_STEPS_PER_ACTION
FAR_GRIPPER_CLOSE_THRESHOLD_M = 0.15  # matches run_ee_position_offset_ab_experiment.py's own threshold

# Which DEFAULT_POSITIONS anchors were actually seen during this project's
# fine-tuning data collection (see collect_recycling_dataset.SPLIT_POSITIONS,
# this task's earlier chat report): train saw center_right/center_left,
# validation saw positive_y, negative_y was held out entirely (reserved for
# Real2Sim). Tagging each episode with this lets the analysis distinguish
# "improved on data it trained on" from "generalized to something it never saw".
POSITION_SPLIT_TAG = {
    "center_right": "train_seen",
    "center_left": "train_seen",
    "positive_y": "validation_seen",
    "negative_y": "never_seen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True, help="e.g. 'zero_shot' or 'fine_tuned' -- tags every row in the output JSON.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--positions", type=str, nargs="+", default=list(DEFAULT_POSITIONS.keys()), choices=list(DEFAULT_POSITIONS.keys()))
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_MAX_POLICY_STEPS)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--position-seed-base", type=int, default=5000, help="Base seed for jitter_position() -- kept fixed across BOTH checkpoint runs so object positions are byte-identical.")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def _classify_failure_reason(final_status: str, ever_held: bool) -> str:
    if final_status == "success":
        return "none (success)"
    if final_status == "released":
        return "released_away_from_bin"
    if final_status == "grasped":
        return "grasped_then_timeout_before_release"
    if ever_held:
        return "held_then_lost_track"  # shouldn't normally happen given task_status semantics, kept for completeness
    return "never_grasped_timeout"


def run_episode(
    policy, position_name, split_tag, position, instruction, instruction_name, bin_position, seed,
    max_steps, steps_per_action, object_type, strict, label,
) -> dict:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    backend.set_object_position(list(position))
    policy.reset()

    step_rows = []
    first_distance_to_object = None
    ever_held = False
    first_grasp_step = None
    release_step = None
    final_status = "running"
    success = False

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
                    f"--strict violated at label={label} pos={position_name} seed={seed} step={step_index}: "
                    f"{'; '.join(violations)}. info={info}"
                )

        action_adapter = ActionAdapter()
        robot_command = action_adapter.convert(policy_output.action)
        commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]
        vector_to_object = [object_position[i] - ee_position[i] for i in range(3)]
        cosine_commanded_vs_object = _cosine_similarity(commanded_translation, vector_to_object)
        sign_match_x = _sign_match(commanded_translation[0], vector_to_object[0])
        far_gripper_close = distance_to_object > FAR_GRIPPER_CLOSE_THRESHOLD_M and robot_command.gripper_command == "close"

        robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
        final_status = robot_state_after["task_status"]
        held_now = bool(robot_state_after["held_object"])
        if held_now and not ever_held:
            ever_held = True
            first_grasp_step = step_index
        if ever_held and not held_now and release_step is None and step_index != first_grasp_step:
            release_step = step_index

        step_rows.append({
            "step": step_index,
            "distance_to_object": distance_to_object,
            "cosine_commanded_vs_object": cosine_commanded_vs_object,
            "sign_match_x": sign_match_x,
            "gripper_command": robot_command.gripper_command,
            "far_gripper_close": far_gripper_close,
            "held_object": held_now,
            "task_status": final_status,
            "inference_latency_ms": info.get("inference_latency_ms"),
            "server_inference_ms": info.get("server_inference_ms"),
            "main_image_hash": image_hash(main_image),
            "wrist_image_hash": image_hash(wrist_image),
        })

        if final_status == "success":
            success = True
            break

    final_robot_state, _, _ = build_robot_state(backend)
    final_distance_to_object = _distance_3d(final_robot_state["ee_position"], object_position)
    backend.shutdown()

    cosines = [r["cosine_commanded_vs_object"] for r in step_rows if r["cosine_commanded_vs_object"] is not None]
    x_matches = [r["sign_match_x"] for r in step_rows if r["sign_match_x"] is not None]
    latencies = [r["inference_latency_ms"] for r in step_rows if r["inference_latency_ms"] is not None]

    return {
        "label": label,
        "position_name": position_name,
        "split_tag": split_tag,
        "position": list(position),
        "instruction_name": instruction_name,
        "seed": seed,
        "num_steps": len(step_rows),
        "success": success,
        "final_task_status": final_status,
        "pick_success": ever_held,
        "place_success": success,  # place implies task completion by definition of task_status=="success"
        "first_grasp_step": first_grasp_step,
        "release_step": release_step,
        "first_distance_to_object": first_distance_to_object,
        "final_distance_to_object": final_distance_to_object,
        "distance_improvement": (first_distance_to_object - final_distance_to_object) if first_distance_to_object is not None else None,
        "mean_cosine_commanded_vs_object": sum(cosines) / len(cosines) if cosines else None,
        "x_sign_accuracy": (sum(1 for m in x_matches if m) / len(x_matches)) if x_matches else None,
        "far_gripper_close_count": sum(1 for r in step_rows if r["far_gripper_close"]),
        "retry_count": 0,  # RealVLAPolicyClient has no retry logic -- see module docstring
        "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "failure_reason": _classify_failure_reason(final_status, ever_held),
        "rows": step_rows,
    }


def main() -> None:
    args = parse_args()
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]

    policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)
    health = policy.check_health()
    print(f"=== Zero-shot vs Fine-tuned comparison -- label={args.label!r} ===")
    print(f"server health: {health}")
    if health.get("model_status") != "loaded":
        raise RuntimeError(
            f"Server model_status={health.get('model_status')!r}, expected 'loaded'. "
            "POST /load_model on the server first."
        )
    if not (health.get("compatibility") or {}).get("passed"):
        raise RuntimeError(f"Server compatibility.passed is not True: {health.get('compatibility')}")
    model_id_or_path = health.get("model_id_or_path")
    print(f"model_id_or_path: {model_id_or_path}")
    print(f"instruction ({args.instruction_name}): {instruction!r}")
    print(f"positions: {args.positions}, seeds: {args.seeds}, max_policy_steps: {args.max_policy_steps}")

    episodes = []
    total = len(args.positions) * len(args.seeds)
    n = 0
    start_time = time.time()
    for position_index, position_name in enumerate(args.positions):
        anchor_position = DEFAULT_POSITIONS[position_name]
        split_tag = POSITION_SPLIT_TAG[position_name]
        for seed in args.seeds:
            n += 1
            # Deterministic across separate process runs (zero-shot and
            # fine-tuned are two different `python -m ...` invocations) --
            # deliberately NOT Python's built-in hash() on a string, which
            # is randomized per-process (PYTHONHASHSEED) and would silently
            # break the "byte-identical object position" guarantee this
            # script exists to provide.
            jitter_seed = args.position_seed_base + position_index * 1000 + seed
            rng = random.Random(jitter_seed)
            position = jitter_position(anchor_position, rng)
            episode = run_episode(
                policy, position_name, split_tag, position, instruction, args.instruction_name,
                DEFAULT_BIN_POSITION, seed, args.max_policy_steps, args.steps_per_action, args.object_type,
                args.strict, args.label,
            )
            episodes.append(episode)
            print(
                f"[{n:02d}/{total}] pos={position_name:<13} split={split_tag:<15} seed={seed} "
                f"success={episode['success']} status={episode['final_task_status']:<10} "
                f"steps={episode['num_steps']:3d} pick={episode['pick_success']} "
                f"dist_improve={episode['distance_improvement']:.4f} "
                f"mean_cos={episode['mean_cosine_commanded_vs_object']}"
            )

    elapsed_s = time.time() - start_time
    result = {
        "label": args.label,
        "model_id_or_path": model_id_or_path,
        "server_health_at_start": health,
        "instruction_name": args.instruction_name,
        "instruction": instruction,
        "positions_requested": args.positions,
        "seeds": args.seeds,
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
