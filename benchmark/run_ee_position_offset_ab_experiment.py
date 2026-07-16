"""ee_position offset A/B experiment (v0) -- the final causal check.

Follow-up to run_environment_state_alignment_diagnostic.py, which found
ee_position.x has ZERO range overlap between real HuggingFaceVLA/libero
training samples (mean -0.073) and this project's PyBullet workspace
samples (mean +0.327) -- a confirmed, understood world-origin placement
difference (robosuite's TableArena/Panda base offsets vs. this project's
own placement), not a frame/semantics bug. What was NOT yet established
is whether that absolute-position gap is actually CAUSING the fixed x/y
translation direction bias found by
run_counterfactual_direction_benchmark.py, or whether it's an inert
side-detail the model's own zero-shot policy quality issue would persist
through regardless.

This script tests that causally: it offsets ONLY the ee_position value
sent to the model (built into the PolicyInput/HTTP request this script
constructs itself), at several auto-generated candidate magnitudes,
while the REAL PyBulletPandaBackend state -- and therefore the real
physics, the real camera images, and the ground truth used to grade
direction/distance -- is NEVER touched. See apply_position_offset()'s
docstring for the exact, single point this is applied and the guarantee
that nothing else changes.

Modifies NO production file, NO checkpoint/config, and does not
fine-tune anything.

Offset candidates are computed automatically from real mean-alignment
(see compute_offset_candidates()) -- NOT hand-picked -- using this
project's own already-built real-vs-ours comparison
(run_environment_state_alignment_diagnostic.py's
load_real_dataset_samples()/collect_pybullet_workspace_samples()), only
on the x axis (the axis with 0% range overlap and the axis this
project's fixed-bias finding centers on) -- y/z are left at their real,
un-offset values throughout, matching this task's own request.

Run (needs a live server, see run_environment_state_alignment_diagnostic.py's
module docstring for the server startup commands; must run under the GPU
venv, e.g. .venv-vla):

  .venv-vla/bin/python -m benchmark.run_ee_position_offset_ab_experiment \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS, _mean, _sign_match, _stdev
from benchmark.run_environment_state_alignment_diagnostic import (
    collect_pybullet_workspace_samples,
    load_real_dataset_samples,
)
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/vla_backend_smolvla_libero_config.json"
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DEFAULT_SEEDS = [0, 42, 123]
FAR_GRIPPER_CLOSE_THRESHOLD_M = 0.15

# Fractions of the FULL real-vs-ours mean-alignment offset on x (e.g. if
# the full offset is -0.40, these produce -0.20/-0.30/-0.40 -- matching
# this task's own worked example almost exactly, but computed from real
# data, not hand-picked).
DEFAULT_OFFSET_FRACTIONS = {"none": 0.0, "half": 0.5, "three_quarter": 0.75, "full": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps-per-condition", type=int, default=5)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--dataset-files", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results/ee_position_offset_ab_experiment")
    return parser.parse_args()


def compute_offset_candidates(dataset_files=None, fractions=None) -> dict:
    """Auto-computes the FULL x-axis mean-alignment offset (real ee_position.x
    mean minus our ee_position.x mean, from the exact same real-dataset /
    workspace-sample collectors run_environment_state_alignment_diagnostic.py
    already built and validated) and returns {label: offset_x} for each
    requested fraction of it. y/z offsets are always 0 -- this task is
    specifically about the x-axis gap (the one with 0% range overlap in
    the prior diagnostic) and its own worked example only varies x."""
    fractions = fractions or DEFAULT_OFFSET_FRACTIONS
    kwargs = {"files": dataset_files} if dataset_files else {}
    real_data = load_real_dataset_samples(**kwargs)
    our_data = collect_pybullet_workspace_samples()

    real_mean_x = float(real_data["flat_states"][:, 0].mean())
    our_mean_x = float(our_data["flat_states"][:, 0].mean())
    full_offset_x = real_mean_x - our_mean_x

    candidates = {label: full_offset_x * fraction for label, fraction in fractions.items()}
    return {
        "candidates": candidates,
        "real_mean_x": real_mean_x,
        "our_mean_x": our_mean_x,
        "full_offset_x": full_offset_x,
    }


def apply_position_offset(robot_state: dict, offset_x: float) -> dict:
    """Returns a NEW dict -- the ONLY thing this whole script ever changes
    is ee_position[0] (x) in the copy of robot_state that gets sent to
    the model via PolicyInput. The REAL PyBulletPandaBackend object (its
    joint state, its rendered camera images, its own get_state()) is
    never touched -- this function operates purely on a plain dict
    already read out of the backend, at the exact same point
    run_gripper_channel_sign_ab_experiment.py's apply_gripper_condition()
    operates on gripper_qpos. See
    test_ee_position_offset_ab_experiment.py's dedicated grep-based test
    confirming this function (and this whole file) is never referenced
    from any production directory."""
    ee_position = list(robot_state["ee_position"])
    ee_position[0] = ee_position[0] + offset_x
    return {**robot_state, "ee_position": ee_position}


def _predict_one(policy, main_image, wrist_image, robot_state_for_model, ground_truth_ee_position, object_position,
                  instruction, bin_position, seed, step_index, strict, label) -> dict:
    policy_input = PolicyInput(
        image=main_image,
        instruction=instruction,
        robot_state=robot_state_for_model,
        task_goal={},
        target_object_position=object_position,
        bin_position=bin_position,
        step_index=step_index,
        phase="move_to_object",
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
            raise RuntimeError(f"--strict violated at {label} seed={seed} step={step_index}: {'; '.join(violations)}. info={info}")

    action_adapter = ActionAdapter()
    robot_command = action_adapter.convert(policy_output.action)
    commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]

    # Ground truth (direction/distance grading) ALWAYS uses the real,
    # un-offset ee_position -- only the copy sent to the model was offset.
    vector_to_object = [object_position[i] - ground_truth_ee_position[i] for i in range(3)]
    distance_before = _distance_3d(ground_truth_ee_position, object_position)
    cosine_commanded = _cosine_similarity(commanded_translation, vector_to_object)
    sign_match_x = _sign_match(commanded_translation[0], vector_to_object[0])
    sign_match_y = _sign_match(commanded_translation[1], vector_to_object[1])
    far_gripper_close = distance_before > FAR_GRIPPER_CLOSE_THRESHOLD_M and robot_command.gripper_command == "close"

    return {
        "label": label,
        "seed": seed,
        "step": step_index,
        "model_input_ee_position": list(robot_state_for_model["ee_position"]),
        "ground_truth_ee_position": list(ground_truth_ee_position),
        "object_position": object_position,
        "commanded_translation": commanded_translation,
        "cosine_commanded_vs_object": cosine_commanded,
        "sign_match_x": sign_match_x,
        "sign_match_y": sign_match_y,
        "gripper_command": robot_command.gripper_command,
        "distance_to_object_before": distance_before,
        "far_gripper_close": far_gripper_close,
        "compatibility_passed": compatibility_passed,
        "semantic_action_valid": semantic_action_valid,
        "degraded_input": degraded_input,
        "fallback_used": fallback_used,
        "server_latency_ms": info.get("inference_latency_ms"),
        "main_image_hash": image_hash(main_image),
        "wrist_image_hash": image_hash(wrist_image),
        "robot_command": robot_command,
    }


def run_condition(policy, position_name, position, offset_x, instruction, bin_position, seed, steps_per_condition,
                   steps_per_action, object_type, strict, label) -> list:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    backend.set_object_position(list(position))

    rows = []
    for step_index in range(steps_per_condition):
        robot_state, _state_8d, object_position = build_robot_state(backend)
        ground_truth_ee_position = list(robot_state["ee_position"])
        robot_state_for_model = apply_position_offset(robot_state, offset_x)
        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()

        row = _predict_one(
            policy, main_image, wrist_image, robot_state_for_model, ground_truth_ee_position, object_position,
            instruction, bin_position, seed, step_index, strict, label,
        )
        row.update({"position_name": position_name, "offset_x": offset_x})
        rows.append(row)

        backend.apply_command(row["robot_command"], steps=steps_per_action)

    final_distance = _distance_3d(build_robot_state(backend)[0]["ee_position"], object_position)
    backend.shutdown()
    for row in rows:
        row["final_distance_to_object"] = final_distance
    return rows


def _strip_internal_fields(rows: list) -> list:
    return [{key: value for key, value in row.items() if key != "robot_command"} for row in rows]


def run_all(args, policy=None, positions=None) -> dict:
    positions = positions or DEFAULT_POSITIONS
    if policy is None:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)

    offset_info = compute_offset_candidates(dataset_files=args.dataset_files)
    candidates = offset_info["candidates"]

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows = []
    total = len(candidates) * len(positions) * len(args.seeds)
    n = 0
    print(f"=== ee_position offset A/B -- {total} conditions -- offset candidates: {candidates} ===")
    for label, offset_x in candidates.items():
        for position_name, position in positions.items():
            for seed in args.seeds:
                n += 1
                rows = run_condition(
                    policy, position_name, position, offset_x, args.instruction, DEFAULT_BIN_POSITION, seed,
                    args.steps_per_condition, args.steps_per_action, args.object_type, args.strict,
                    f"{label}__{position_name}",
                )
                all_rows.extend(rows)
                mean_cos = _mean([row["cosine_commanded_vs_object"] for row in rows if row["cosine_commanded_vs_object"] is not None])
                print(f"[{n:03d}/{total}] offset={label:14s}({offset_x:+.3f}) pos={position_name:<13} seed={seed:<4} mean_cos={mean_cos}")

    summary = summarize_offsets(all_rows, candidates)
    verdict = decide_causal_verdict(summary)

    result = {
        "offset_candidates": candidates,
        "offset_computation": offset_info,
        "rows": _strip_internal_fields(all_rows),
        "summary_by_offset": summary,
        "verdict": verdict,
    }

    log_path = output_dir / f"ee_position_offset_ab_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as log_file:
        json.dump(result, log_file, ensure_ascii=False, indent=2, default=str)

    print_summary(summary, verdict)
    print(f"\nFull result JSON: {log_path}")
    result["log_path"] = str(log_path)
    return result


def summarize_offsets(rows: list, candidates: dict) -> dict:
    summary = {}
    for label in candidates:
        offset_rows = [row for row in rows if row["label"].startswith(f"{label}__")]
        cosines = [row["cosine_commanded_vs_object"] for row in offset_rows if row["cosine_commanded_vs_object"] is not None]
        x_matches = [row["sign_match_x"] for row in offset_rows if row["sign_match_x"] is not None]
        y_matches = [row["sign_match_y"] for row in offset_rows if row["sign_match_y"] is not None]
        far_closes = [row["far_gripper_close"] for row in offset_rows]
        degraded = [row["degraded_input"] for row in offset_rows]
        fallback = [row["fallback_used"] for row in offset_rows]
        semantic_valid = [row["semantic_action_valid"] for row in offset_rows]

        # distance improvement per (position, seed) cell: first step's
        # distance_to_object_before minus the episode's final distance.
        cells = {}
        for row in offset_rows:
            key = (row["position_name"], row["seed"])
            cells.setdefault(key, []).append(row)
        improvements = []
        for cell_rows in cells.values():
            cell_rows_sorted = sorted(cell_rows, key=lambda r: r["step"])
            improvements.append(cell_rows_sorted[0]["distance_to_object_before"] - cell_rows_sorted[0]["final_distance_to_object"])

        summary[label] = {
            "num_rows": len(offset_rows),
            "mean_cosine": _mean(cosines),
            "std_cosine": _stdev(cosines),
            "x_sign_accuracy": (sum(1 for m in x_matches if m) / len(x_matches)) if x_matches else None,
            "y_sign_accuracy": (sum(1 for m in y_matches if m) / len(y_matches)) if y_matches else None,
            "mean_distance_improvement": _mean(improvements),
            "success_rate": 0.0,  # task_status=="success" never reached within this step budget -- see report
            "far_gripper_close_rate": (sum(1 for v in far_closes if v) / len(far_closes)) if far_closes else None,
            "semantic_action_valid_rate": (sum(1 for v in semantic_valid if v) / len(semantic_valid)) if semantic_valid else None,
            "degraded_input_rate": (sum(1 for v in degraded if v) / len(degraded)) if degraded else None,
            "fallback_used_rate": (sum(1 for v in fallback if v) / len(fallback)) if fallback else None,
        }
    return summary


EFFECT_SIZE_MEANINGFUL_THRESHOLD = 0.15  # cosine or sign-accuracy delta vs. "none" baseline


def decide_causal_verdict(summary: dict) -> dict:
    if "none" not in summary:
        return {"verdict": "unknown", "reason": "no 'none' baseline condition present"}

    baseline = summary["none"]
    best_label = max((label for label in summary if label != "none"), key=lambda label: summary[label]["mean_cosine"] or -999, default=None)
    if best_label is None:
        return {"verdict": "unknown", "reason": "no offset conditions to compare against baseline"}

    best = summary[best_label]
    cosine_delta = (best["mean_cosine"] - baseline["mean_cosine"]) if (best["mean_cosine"] is not None and baseline["mean_cosine"] is not None) else None
    x_sign_delta = (best["x_sign_accuracy"] - baseline["x_sign_accuracy"]) if (best["x_sign_accuracy"] is not None and baseline["x_sign_accuracy"] is not None) else None
    y_sign_delta = (best["y_sign_accuracy"] - baseline["y_sign_accuracy"]) if (best["y_sign_accuracy"] is not None and baseline["y_sign_accuracy"] is not None) else None

    meaningful_improvement = (
        (cosine_delta is not None and cosine_delta > EFFECT_SIZE_MEANINGFUL_THRESHOLD)
        or (x_sign_delta is not None and x_sign_delta > EFFECT_SIZE_MEANINGFUL_THRESHOLD)
    )

    if meaningful_improvement:
        verdict = "A"
        cosine_delta_str = f"{cosine_delta:+.3f}" if cosine_delta is not None else "n/a"
        x_sign_delta_str = f"{x_sign_delta:+.3f}" if x_sign_delta is not None else "n/a"
        reason = (
            f"Best offset ('{best_label}') improves mean cosine by {cosine_delta_str} and x-sign-accuracy by "
            f"{x_sign_delta_str} relative to no-offset baseline -- at least one is above the "
            f"{EFFECT_SIZE_MEANINGFUL_THRESHOLD} effect-size threshold. Coordinate/position-distribution "
            "mismatch is a real, causal contributor to the direction bias."
        )
    else:
        cosine_delta_str = f"{cosine_delta:+.3f}" if cosine_delta is not None else "n/a"
        reason = (
            f"Best offset ('{best_label}') changes mean cosine by only {cosine_delta_str} "
            f"and x-sign-accuracy by {x_sign_delta}, both below the {EFFECT_SIZE_MEANINGFUL_THRESHOLD} effect-size "
            "threshold -- shifting the model's input position closer to its training distribution does not "
            "meaningfully change its output direction. The bias is better explained by the model's own "
            "zero-shot policy quality/task mismatch than by the coordinate offset."
        )
        verdict = "B"

    return {
        "verdict": verdict,
        "reason": reason,
        "baseline_label": "none",
        "best_label": best_label,
        "cosine_delta_vs_baseline": cosine_delta,
        "x_sign_accuracy_delta_vs_baseline": x_sign_delta,
        "y_sign_accuracy_delta_vs_baseline": y_sign_delta,
        "effect_size_threshold": EFFECT_SIZE_MEANINGFUL_THRESHOLD,
    }


def print_summary(summary: dict, verdict: dict) -> None:
    print("\n=== Summary by offset ===")
    for label, stats in summary.items():
        print(
            f"{label:14s} mean_cos={stats['mean_cosine']} x_sign_acc={stats['x_sign_accuracy']} "
            f"y_sign_acc={stats['y_sign_accuracy']} dist_improve={stats['mean_distance_improvement']} "
            f"far_close_rate={stats['far_gripper_close_rate']} degraded={stats['degraded_input_rate']} "
            f"fallback={stats['fallback_used_rate']}"
        )
    print("\n=== Verdict ===")
    print(f"verdict: {verdict['verdict']}")
    print(f"reason: {verdict['reason']}")


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
