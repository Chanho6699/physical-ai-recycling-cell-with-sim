"""Gripper channel-2 sign A/B experiment (v0).

Follow-up to run_state_semantics_diagnostic.py, which found (and this
module's docstring reproduces the confirming evidence for) that
gripper_qpos[1] is likely sign-flipped relative to
HuggingFaceVLA/smolvla_libero's training convention. This script does
NOT apply that fix to production -- it runs a controlled A/B/C/D
comparison, diagnostic-only, to separate whether the sign mismatch
actually explains (1) far-gripper premature close/open behavior, (2) the
fixed x/y translation bias found by run_counterfactual_direction_benchmark.py,
both, or neither.

=== CONFIRMED EVIDENCE (this task's explicit priority: real dataset
samples over checkpoint mean alone) ===

1. Real HuggingFaceVLA/libero LeRobot dataset sample (chunk-000/file-000.parquet,
   843 rows, 3 episodes) -- observation.state dims 6/7 read directly:
     dim 6 (gripper_qpos[0]): range [0.0019, 0.0402], always positive
     dim 7 (gripper_qpos[1]): range [-0.0405, -0.0028], always negative
   Row-by-row, dim7 approx -dim6 (e.g. row0: [0.03879, -0.03879]; within
   one episode's open->close->open trajectory, both channels track each
   other with opposite sign at every single timestep, not just on
   average). This is a MUCH stronger, directly-observed confirmation
   than the checkpoint's own aggregate mean/std stats alone -- exactly
   what this task asked to prioritize.

2. Root cause traced to an exact, non-guessed code location -- NOT a
   "negation step" in any conversion script, but a physical joint-model
   difference between the two simulators' bundled Panda gripper assets:
     robosuite/models/assets/grippers/panda_gripper.xml:
       finger_joint1: range="0.0 0.04"   (positive)
       finger_joint2: range="-0.04 0.0"  (NEGATIVE -- this IS the raw
                                          MuJoCo qpos range, straight from
                                          the asset's own joint axis
                                          definition)
     robot_sim/pybullet_panda_backend.py (this project, unmodified):
       both panda_finger_joint1/2 (from PyBullet's bundled
       franka_panda/panda.urdf) are POSITIVE-only [0, 0.04].
   robosuite's own gripper_qpos sensor (robosuite/robots/robot.py) is
   `[sim.data.qpos[x] for x in gripper_joint_pos_indexes]` -- a raw,
   unmodified MuJoCo qpos read. Because finger_joint2's OWN range is
   already negative in the asset file, that raw read is naturally
   negative; nothing downstream negates anything. Our own
   get_libero_observation_state() reads PyBullet's own two finger
   joints, BOTH of which are modeled with a positive-only range -- so
   there is no direct way to reproduce robosuite's exact per-joint
   convention there; the closest diagnostic stand-in is negating our own
   second finger's reading in the REQUEST we send (never in
   get_libero_observation_state() itself -- see this module's docstring
   "PRODUCTION UNCHANGED" guarantee below).

This script never modifies robot_sim/pybullet_panda_backend.py,
vla_server/model_loader.py, any checkpoint, or any config -- see
apply_gripper_condition()'s docstring and
benchmark/test_gripper_channel_sign_ab_experiment.py's dedicated
grep-based test asserting this file is never referenced from any
production directory.

Run (needs a live server, see run_state_semantics_diagnostic.py's module
docstring for the exact server startup commands; must run under the GPU
venv, e.g. .venv-vla, for the same reason that script does):

  .venv-vla/bin/python -m benchmark.run_gripper_channel_sign_ab_experiment \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json \\
    --mode one-step
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_counterfactual_direction_benchmark import DEFAULT_INSTRUCTIONS, DEFAULT_POSITIONS, _mean, _sign_match, _stdev
from benchmark.run_full_recycling_cell_demo import _cosine_similarity
from benchmark.run_state_semantics_diagnostic import load_checkpoint_state_stats
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/vla_backend_smolvla_libero_config.json"
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]
FAR_GRIPPER_CLOSE_THRESHOLD_M = 0.15

# The 4 required conditions. "current_positive_pair"/"mirrored_signed_pair"
# are DERIVED from a real backend's actual finger qpos (real_q1, real_q2);
# "checkpoint_mean_open"/"zero_pair" are fixed reference vectors,
# independent of the robot's actual physical gripper state.
DERIVED_CONDITIONS = ("current_positive_pair", "mirrored_signed_pair")
FIXED_CONDITIONS = {
    "checkpoint_mean_open": [0.0269, -0.0272],
    "zero_pair": [0.0, 0.0],
}
ALL_CONDITIONS = DERIVED_CONDITIONS + tuple(FIXED_CONDITIONS)

DEFAULT_SEEDS = [0, 42, 123]
DEFAULT_AB_INSTRUCTIONS = {"ko_full": DEFAULT_INSTRUCTIONS["ko_full"], "en_minimal": DEFAULT_INSTRUCTIONS["en_minimal"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--checkpoint-repo-id", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mode", choices=["one-step", "multi-step"], default="one-step")
    parser.add_argument("--steps-per-condition", type=int, default=5)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--output-dir", type=str, default="results/gripper_channel_sign_ab_experiment")
    return parser.parse_args()


def apply_gripper_condition(robot_state: dict, condition: str, real_q1=None, real_q2=None) -> dict:
    """Returns a NEW dict (never mutates robot_state) with ONLY
    gripper_qpos overridden -- ee_position/ee_orientation_axis_angle and
    everything else in robot_state (and the images sent alongside it)
    stay byte-identical across all 4 conditions, which is the whole
    point of this being a controlled A/B, not a confound. Diagnostic-only:
    called exclusively from this script, never from
    robot_sim/pybullet_panda_backend.py or vla_server/model_loader.py --
    see test_gripper_channel_sign_ab_experiment.py's grep-based
    regression check."""
    if condition == "current_positive_pair":
        pair = [real_q1, real_q2]
    elif condition == "mirrored_signed_pair":
        pair = [real_q1, -real_q2]
    elif condition in FIXED_CONDITIONS:
        pair = list(FIXED_CONDITIONS[condition])
    else:
        raise ValueError(f"Unknown gripper condition: {condition!r}")
    return {**robot_state, "gripper_qpos": pair}


def _predict_one(policy, main_image, wrist_image, robot_state, instruction, object_position, bin_position, seed,
                  step_index, strict, checkpoint_stats, label) -> dict:
    policy_input = PolicyInput(
        image=main_image,
        instruction=instruction,
        robot_state=robot_state,
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

    action_postprocess = info.get("action_postprocess") or {}
    canonical_after = action_postprocess.get("canonical_command") or {}
    metadata = canonical_after.get("metadata") or {}
    raw_model_action = metadata.get("raw_model_action")
    postprocessed_gripper = metadata.get("native_action_raw_values")
    gripper_raw_channel = raw_model_action[6] if raw_model_action else None
    postprocessed_gripper_channel = postprocessed_gripper[6] if postprocessed_gripper else None

    action_adapter = ActionAdapter()
    robot_command = action_adapter.convert(policy_output.action)
    commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]

    ee_position = robot_state["ee_position"]
    vector_to_object = [object_position[i] - ee_position[i] for i in range(3)]
    distance_before = sum((ee_position[i] - object_position[i]) ** 2 for i in range(3)) ** 0.5
    cosine_commanded = _cosine_similarity(commanded_translation, vector_to_object)
    sign_match_xyz = [_sign_match(commanded_translation[i], vector_to_object[i]) for i in range(3)]
    far_gripper_close = distance_before > FAR_GRIPPER_CLOSE_THRESHOLD_M and robot_command.gripper_command == "close"

    state_8d = list(robot_state["ee_position"]) + list(robot_state["ee_orientation_axis_angle"]) + list(robot_state["gripper_qpos"])
    normalized_state_8d = None
    gripper_zscore = [None, None]
    if checkpoint_stats is not None:
        mean = checkpoint_stats["observation_state_mean"]
        std = checkpoint_stats["observation_state_std"]
        normalized_state_8d = [(state_8d[i] - mean[i]) / std[i] if std[i] > 1e-9 else None for i in range(8)]
        gripper_zscore = [normalized_state_8d[6], normalized_state_8d[7]]

    return {
        "label": label,
        "seed": seed,
        "step": step_index,
        "instruction": instruction,
        "object_position": object_position,
        "exact_input_state_8d": state_8d,
        "normalized_state_8d": normalized_state_8d,
        "gripper_dim6_zscore": gripper_zscore[0],
        "gripper_dim7_zscore": gripper_zscore[1],
        "commanded_translation": commanded_translation,
        "cosine_commanded_vs_object": cosine_commanded,
        "sign_match_x": sign_match_xyz[0],
        "sign_match_y": sign_match_xyz[1],
        "sign_match_z": sign_match_xyz[2],
        "raw_gripper_channel": gripper_raw_channel,
        "postprocessed_gripper_channel": postprocessed_gripper_channel,
        "executed_gripper_command": robot_command.gripper_command,
        "distance_to_object": distance_before,
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


def run_one_step_grid(policy, positions, instructions, seeds, object_type, bin_position, strict, checkpoint_stats) -> list:
    rows = []
    for position_name, position in positions.items():
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        backend.set_object_type(object_type)
        backend.set_object_position(list(position))
        base_robot_state, _state_8d, object_position = build_robot_state(backend)
        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()

        for grip_physical_state, apply_grip in (("open", backend.open_gripper), ("closed", backend.close_gripper)):
            apply_grip()
            _refreshed_state, refreshed_8d, _obj = build_robot_state(backend)
            real_q1, real_q2 = refreshed_8d[6], refreshed_8d[7]
            for condition in DERIVED_CONDITIONS:
                override_state = apply_gripper_condition(base_robot_state, condition, real_q1, real_q2)
                for instruction_name, instruction in instructions.items():
                    for seed in seeds:
                        row = _predict_one(
                            policy, main_image, wrist_image, override_state, instruction, object_position,
                            bin_position, seed, 0, strict, checkpoint_stats,
                            f"{position_name}__{instruction_name}__{grip_physical_state}__{condition}",
                        )
                        row.update({"position_name": position_name, "instruction_name": instruction_name, "grip_physical_state": grip_physical_state, "condition": condition})
                        rows.append(row)

        for condition in FIXED_CONDITIONS:
            override_state = apply_gripper_condition(base_robot_state, condition)
            for instruction_name, instruction in instructions.items():
                for seed in seeds:
                    row = _predict_one(
                        policy, main_image, wrist_image, override_state, instruction, object_position,
                        bin_position, seed, 0, strict, checkpoint_stats,
                        f"{position_name}__{instruction_name}__fixed__{condition}",
                    )
                    row.update({"position_name": position_name, "instruction_name": instruction_name, "grip_physical_state": "fixed", "condition": condition})
                    rows.append(row)
        backend.shutdown()
    return rows


def run_multi_step_grid(policy, positions, instructions, seeds, object_type, bin_position, strict, checkpoint_stats,
                         steps_per_condition, steps_per_action) -> list:
    """Reduced, confirmatory grid: only the 2 DERIVED conditions (A, B),
    applying the model's FULL returned command (translation + rotation +
    gripper) for steps_per_condition real steps -- re-deriving
    (real_q1, real_q2) from the backend's actual, evolving finger qpos
    every single step, so "mirrored" always means "negate whatever the
    real second-finger reading currently is," not a value frozen from
    step 0."""
    rows = []
    for position_name, position in positions.items():
        for grip_physical_state in ("open", "closed"):
            for condition in DERIVED_CONDITIONS:
                for instruction_name, instruction in instructions.items():
                    for seed in seeds:
                        backend = PyBulletPandaBackend(gui=False)
                        backend.reset()
                        backend.set_object_type(object_type)
                        backend.set_object_position(list(position))
                        if grip_physical_state == "open":
                            backend.open_gripper()
                        else:
                            backend.close_gripper()

                        action_adapter = ActionAdapter()
                        for step_index in range(steps_per_condition):
                            robot_state, state_8d, object_position = build_robot_state(backend)
                            real_q1, real_q2 = state_8d[6], state_8d[7]
                            override_state = apply_gripper_condition(robot_state, condition, real_q1, real_q2)
                            main_image = backend.render_main_camera()
                            wrist_image = backend.render_wrist_camera()
                            row = _predict_one(
                                policy, main_image, wrist_image, override_state, instruction, object_position,
                                bin_position, seed, step_index, strict, checkpoint_stats,
                                f"multistep__{position_name}__{instruction_name}__{grip_physical_state}__{condition}",
                            )
                            row.update({
                                "position_name": position_name, "instruction_name": instruction_name,
                                "grip_physical_state": grip_physical_state, "condition": condition,
                            })
                            rows.append(row)
                            backend.apply_command(row["robot_command"], steps=steps_per_action)
                        backend.shutdown()
    return rows


def _strip_internal_fields(rows: list) -> list:
    return [{key: value for key, value in row.items() if key != "robot_command"} for row in rows]


# --- Automatic comparison ---


def compare_conditions(rows: list, condition_a="current_positive_pair", condition_b="mirrored_signed_pair") -> dict:
    """Per (position_name, instruction_name, grip_physical_state) cell,
    compares condition_a vs. condition_b's mean cosine/sign-accuracy/far-
    gripper-close-rate, then aggregates cell-level deltas -- so an
    improvement has to show up consistently across cells to count, not
    just in one lucky combination."""
    cell_keys = sorted({
        (row["position_name"], row["instruction_name"], row["grip_physical_state"])
        for row in rows if row["condition"] in (condition_a, condition_b)
    })

    cell_comparisons = {}
    for cell_key in cell_keys:
        position_name, instruction_name, grip_state = cell_key
        rows_a = [row for row in rows if row["position_name"] == position_name and row["instruction_name"] == instruction_name and row["grip_physical_state"] == grip_state and row["condition"] == condition_a]
        rows_b = [row for row in rows if row["position_name"] == position_name and row["instruction_name"] == instruction_name and row["grip_physical_state"] == grip_state and row["condition"] == condition_b]
        if not rows_a or not rows_b:
            continue

        def _cell_stats(cell_rows):
            cosines = [row["cosine_commanded_vs_object"] for row in cell_rows if row["cosine_commanded_vs_object"] is not None]
            xy_matches = [row[f"sign_match_{axis}"] for row in cell_rows for axis in ("x", "y") if row[f"sign_match_{axis}"] is not None]
            far_closes = [row["far_gripper_close"] for row in cell_rows]
            return {
                "mean_cosine": _mean(cosines),
                "xy_sign_accuracy": (sum(1 for m in xy_matches if m) / len(xy_matches)) if xy_matches else None,
                "far_gripper_close_rate": (sum(1 for v in far_closes if v) / len(far_closes)) if far_closes else None,
                "per_seed_cosine": {row["seed"]: row["cosine_commanded_vs_object"] for row in cell_rows},
            }

        stats_a = _cell_stats(rows_a)
        stats_b = _cell_stats(rows_b)
        cell_comparisons[f"{position_name}__{instruction_name}__{grip_state}"] = {
            "condition_a": stats_a, "condition_b": stats_b,
            "cosine_delta_b_minus_a": (stats_b["mean_cosine"] - stats_a["mean_cosine"]) if stats_a["mean_cosine"] is not None and stats_b["mean_cosine"] is not None else None,
            "xy_sign_accuracy_delta_b_minus_a": (stats_b["xy_sign_accuracy"] - stats_a["xy_sign_accuracy"]) if stats_a["xy_sign_accuracy"] is not None and stats_b["xy_sign_accuracy"] is not None else None,
            "far_gripper_close_rate_delta_b_minus_a": (stats_b["far_gripper_close_rate"] - stats_a["far_gripper_close_rate"]) if stats_a["far_gripper_close_rate"] is not None and stats_b["far_gripper_close_rate"] is not None else None,
        }

    cosine_deltas = [c["cosine_delta_b_minus_a"] for c in cell_comparisons.values() if c["cosine_delta_b_minus_a"] is not None]
    sign_acc_deltas = [c["xy_sign_accuracy_delta_b_minus_a"] for c in cell_comparisons.values() if c["xy_sign_accuracy_delta_b_minus_a"] is not None]
    far_close_deltas = [c["far_gripper_close_rate_delta_b_minus_a"] for c in cell_comparisons.values() if c["far_gripper_close_rate_delta_b_minus_a"] is not None]

    improved_cells = sum(1 for d in cosine_deltas if d > 0)
    return {
        "cell_comparisons": cell_comparisons,
        "num_cells": len(cell_comparisons),
        "cells_with_cosine_improvement": improved_cells,
        "cosine_delta_mean": _mean(cosine_deltas),
        "cosine_delta_std": _stdev(cosine_deltas),
        "xy_sign_accuracy_delta_mean": _mean(sign_acc_deltas),
        "far_gripper_close_rate_delta_mean": _mean(far_close_deltas),
        "reproducibility_fraction": (improved_cells / len(cosine_deltas)) if cosine_deltas else None,
    }


def count_gripper_switches(rows: list) -> int:
    """Number of open<->close transitions in a step-ordered sequence --
    only meaningful for --mode multi-step (a single one-step condition
    has nothing to "switch" against)."""
    ordered = sorted(rows, key=lambda row: row["step"])
    switches = 0
    for i in range(1, len(ordered)):
        if ordered[i]["executed_gripper_command"] != ordered[i - 1]["executed_gripper_command"]:
            switches += 1
    return switches


CONSISTENT_IMPROVEMENT_FRACTION_THRESHOLD = 0.7
SCOPE_EFFECT_SIZE_MARGIN = 0.1


def judge_consistent_improvement(comparison: dict) -> dict:
    fraction = comparison["reproducibility_fraction"]
    if fraction is None:
        return {"suspected": None, "reason": "no comparable cells with both conditions present"}
    suspected = fraction >= CONSISTENT_IMPROVEMENT_FRACTION_THRESHOLD and (comparison["cosine_delta_mean"] or 0) > 0
    return {
        "suspected": suspected,
        "reason": (
            f"mirrored_signed_pair improved cosine_commanded_vs_object in {comparison['cells_with_cosine_improvement']}/"
            f"{comparison['num_cells']} cells ({fraction:.0%}), mean delta={comparison['cosine_delta_mean']:.3f} -- "
            "consistent across positions/instructions/grip-states, not a single-cell fluke."
            if suspected
            else f"mirrored_signed_pair only improved cosine in {comparison['cells_with_cosine_improvement']}/"
            f"{comparison['num_cells']} cells ({fraction:.0%}) -- below the "
            f"{CONSISTENT_IMPROVEMENT_FRACTION_THRESHOLD:.0%} consistency bar, so this reads as unreliable/coincidental "
            "rather than a robust fix."
        ),
        "reproducibility_fraction": fraction,
    }


def judge_improvement_scope(comparison: dict) -> dict:
    cosine_delta = comparison["cosine_delta_mean"] or 0.0
    sign_delta = comparison["xy_sign_accuracy_delta_mean"] or 0.0
    far_close_delta = comparison["far_gripper_close_rate_delta_mean"] or 0.0

    direction_improved = cosine_delta > SCOPE_EFFECT_SIZE_MARGIN or sign_delta > SCOPE_EFFECT_SIZE_MARGIN
    gripper_improved = far_close_delta < -SCOPE_EFFECT_SIZE_MARGIN  # negative delta = fewer far-gripper-closes

    if direction_improved and gripper_improved:
        scope = "both"
    elif gripper_improved:
        scope = "gripper_only"
    elif direction_improved:
        scope = "translation_only"
    else:
        scope = "neither"

    return {
        "scope": scope,
        "reason": (
            f"cosine_delta={cosine_delta:+.3f}, xy_sign_accuracy_delta={sign_delta:+.3f}, "
            f"far_gripper_close_rate_delta={far_close_delta:+.3f} (negative = fewer premature closes) -- "
            f"scope classified as '{scope}'."
        ),
    }


def print_summary(comparison: dict, consistency: dict, scope: dict, gripper_switch_delta) -> None:
    print("\n=== A (current_positive_pair) vs. B (mirrored_signed_pair) ===")
    print(f"cells compared: {comparison['num_cells']}")
    print(f"cells with cosine improvement: {comparison['cells_with_cosine_improvement']} ({comparison['reproducibility_fraction']})")
    print(f"cosine_delta_mean: {comparison['cosine_delta_mean']}  (std={comparison['cosine_delta_std']})")
    print(f"xy_sign_accuracy_delta_mean: {comparison['xy_sign_accuracy_delta_mean']}")
    print(f"far_gripper_close_rate_delta_mean: {comparison['far_gripper_close_rate_delta_mean']}")
    if gripper_switch_delta is not None:
        print(f"gripper_switch_count_delta (multi-step only): {gripper_switch_delta}")

    print("\n=== Judgments ===")
    print(f"consistent_improvement: suspected={consistency['suspected']}")
    print(f"  reason: {consistency['reason']}")
    print(f"improvement_scope: {scope['scope']}")
    print(f"  reason: {scope['reason']}")


def run_all(args, policy=None) -> dict:
    if policy is None:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)

    try:
        checkpoint_stats = load_checkpoint_state_stats(args.checkpoint_repo_id)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        checkpoint_stats = None

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "one-step":
        rows = run_one_step_grid(policy, DEFAULT_POSITIONS, DEFAULT_AB_INSTRUCTIONS, args.seeds, args.object_type, DEFAULT_BIN_POSITION, args.strict, checkpoint_stats)
        gripper_switch_delta = None
    else:
        rows = run_multi_step_grid(
            policy, DEFAULT_POSITIONS, DEFAULT_AB_INSTRUCTIONS, args.seeds, args.object_type, DEFAULT_BIN_POSITION,
            args.strict, checkpoint_stats, args.steps_per_condition, args.steps_per_action,
        )
        switches_a = sum(
            count_gripper_switches([row for row in rows if row["condition"] == "current_positive_pair" and row["position_name"] == p and row["instruction_name"] == i and row["grip_physical_state"] == g and row["seed"] == s])
            for p in DEFAULT_POSITIONS for i in DEFAULT_AB_INSTRUCTIONS for g in ("open", "closed") for s in args.seeds
        )
        switches_b = sum(
            count_gripper_switches([row for row in rows if row["condition"] == "mirrored_signed_pair" and row["position_name"] == p and row["instruction_name"] == i and row["grip_physical_state"] == g and row["seed"] == s])
            for p in DEFAULT_POSITIONS for i in DEFAULT_AB_INSTRUCTIONS for g in ("open", "closed") for s in args.seeds
        )
        gripper_switch_delta = switches_b - switches_a

    comparison = compare_conditions(rows)
    consistency = judge_consistent_improvement(comparison)
    scope = judge_improvement_scope(comparison)

    result = {
        "mode": args.mode,
        "rows": _strip_internal_fields(rows),
        "comparison": comparison,
        "judgments": {"consistent_improvement": consistency, "improvement_scope": scope},
        "gripper_switch_count_delta": gripper_switch_delta,
    }

    log_path = output_dir / f"gripper_sign_ab_{args.mode}_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as log_file:
        json.dump(result, log_file, ensure_ascii=False, indent=2, default=str)

    print(f"=== Gripper channel sign A/B experiment ({args.mode}) -- {len(rows)} rows -- log: {log_path} ===")
    print_summary(comparison, consistency, scope, gripper_switch_delta)
    print(f"\nFull result JSON: {log_path}")
    result["log_path"] = str(log_path)
    return result


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
