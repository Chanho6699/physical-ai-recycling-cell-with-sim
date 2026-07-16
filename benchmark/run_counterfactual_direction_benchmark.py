"""Controlled counterfactual direction benchmark (v0).

Follow-up to benchmark/run_vla_action_direction_diagnostic.py. That
diagnostic already isolated the executor and adapter as innocent (commanded
translation == actual displacement; raw -> adapted -> commanded is wired
correctly) and confirmed HuggingFaceVLA/smolvla_libero's n_action_steps=1
means there is no stale action-chunk queue. What's left unknown is WHY the
model's own commanded direction is only weakly correlated with the real
object position, with a consistent wrong-sign tendency on x in the one
scene tested so far. This script isolates which of four candidate causes
that is, by varying exactly one thing at a time against a real, live
server:

  A. Object position counterfactual -- same camera/robot initial state,
     only the object's sim position changes (4 positions: left/right of
     center, +y/-y).
  B. Instruction wording -- Korean vs. 3 English variants, at every
     position.
  C. Seed -- 3 seeds per (position, instruction) cell, to separate a
     genuine directional bias from ordinary sampling noise.

It reuses run_vla_action_direction_diagnostic.py's own building blocks
(build_robot_state(), image_hash(), resolve()) and
run_full_recycling_cell_demo.py's cosine/distance helpers -- no
reimplementation, and neither production file nor any model
checkpoint/config is modified. No axis-sign "correction" is applied
anywhere in this file; it only measures and reports.

Judgment rules (see judge_* functions, each independently testable):
  - fixed_x_bias: mirroring the object left/right of center should flip
    the sign of vector_ee_to_object.x; if mean commanded x sign does NOT
    flip with it, that's a fixed bias, not scene-dependent reasoning.
  - fixed_y_bias: same idea for +y/-y.
  - language_issue: one instruction (or language) scoring much higher mean
    cosine than the others, consistently across positions/seeds.
  - domain_gap: cosine stays low/negative across every position AND every
    instruction -- nothing language- or bias-related explains it, points
    at the visual scene itself or a checkpoint/task mismatch.
  - seed_instability: within a single (position, instruction) cell, the
    3 seeds' cosine values disagree by more than a set threshold.

Run (needs a live server, e.g.
  VLA_MODEL_FAMILY=smolvla VLA_DTYPE=float32
  VLA_BACKEND_CONFIG_PATH=configs/vla_backend_smolvla_libero_config.json
  uvicorn vla_server.generic_vla_server:app --host 0.0.0.0 --port 9200
  then POST /load_model):

  python -m benchmark.run_counterfactual_direction_benchmark \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json \\
    --mode multi-step --steps-per-condition 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/vla_backend_smolvla_libero_config.json"
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]

# A. Object position counterfactual (spec's 4 minimum positions).
DEFAULT_POSITIONS = {
    "center_right": [0.42, 0.00, 0.05],
    "center_left": [0.27, 0.00, 0.05],
    "positive_y": [0.35, 0.18, 0.05],
    "negative_y": [0.35, -0.18, 0.05],
}

# B. Instruction comparison.
DEFAULT_INSTRUCTIONS = {
    "ko_full": "플라스틱 병을 플라스틱 수거함에 넣어줘",
    "en_full": "Pick up the bottle and place it in the bin.",
    "en_short": "Pick up the bottle.",
    "en_minimal": "Move the gripper toward the bottle.",
}

# C. Seed comparison.
DEFAULT_SEEDS = [0, 42, 123]

SIGN_EPSILON = 1e-3
FAR_GRIPPER_CLOSE_THRESHOLD_M = 0.15
LANGUAGE_ISSUE_COSINE_GAP = 0.3
DOMAIN_GAP_COSINE_CEILING = 0.1
SEED_INSTABILITY_STD_THRESHOLD = 0.35
FIXED_BIAS_MIN_SIGN_MATCH_GAP = 0.5  # see judge_fixed_axis_bias()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--bin-position", type=float, nargs=3, default=DEFAULT_BIN_POSITION, metavar=("X", "Y", "Z"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--mode",
        choices=["one-step", "multi-step"],
        default="multi-step",
        help="'one-step': a single inference call per condition, no backend.apply_command() at all -- the "
        "robot never moves, purely measuring the model's first reaction to each counterfactual scene. "
        "'multi-step' (default): up to --steps-per-condition real actions ARE applied, so executor-side "
        "cosine (commanded vs. actual displacement) can also be checked per condition, not just the "
        "model's raw output.",
    )
    parser.add_argument("--steps-per-condition", type=int, default=5)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--output-dir", type=str, default="results/counterfactual_direction_benchmark")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def _sign_match(commanded_value: float, vector_value: float):
    """None if the reference vector's component is negligible (sign
    comparison wouldn't be meaningful, e.g. object and EE nearly aligned
    on that axis); True/False otherwise. Never "corrects" anything -- a
    pure measurement."""
    if abs(vector_value) < SIGN_EPSILON:
        return None
    if abs(commanded_value) < SIGN_EPSILON:
        return False
    return (commanded_value > 0) == (vector_value > 0)


def run_condition(
    policy,
    position_name: str,
    position: list,
    instruction_name: str,
    instruction: str,
    seed: int,
    mode: str,
    steps_per_condition: int,
    steps_per_action: int,
    object_type: str,
    bin_position: list,
    strict: bool,
    backend=None,
) -> list:
    """Runs one (position, instruction, seed) cell -- 1 step in
    "one-step" mode, up to steps_per_condition in "multi-step" mode --
    and returns its rows (list of dict, one per step). backend= is
    injectable for tests; production callers leave it None (fresh
    PyBulletPandaBackend reset to the SAME canonical initial pose every
    single condition, exactly as the counterfactual design requires)."""
    owns_backend = backend is None
    if backend is None:
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()

    backend.set_object_type(object_type)
    backend.set_object_position(list(position))
    policy.reset()
    action_adapter = ActionAdapter()

    steps = 1 if mode == "one-step" else steps_per_condition
    rows = []
    for step_index in range(steps):
        robot_state, state_8d, object_position = build_robot_state(backend)
        ee_before = list(state_8d[0:3])
        vector_to_object = [object_position[i] - ee_before[i] for i in range(3)]
        distance_before = _distance_3d(ee_before, object_position)

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
            phase="move_to_object",
            images_by_role={"main": main_image, "wrist": wrist_image},
            seed=(seed + step_index) if seed is not None else None,
        )
        policy_output = policy.predict_action(policy_input)
        info = policy_output.info or {}

        compatibility_passed = (info.get("compatibility") or {}).get("passed")
        semantic_action_valid = bool(info.get("semantic_action_valid", True))
        degraded_input = bool(info.get("degraded_input", False))
        fallback_used = bool(info.get("fallback_used", False))
        server_latency_ms = info.get("inference_latency_ms")

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
                    f"--strict violated at position={position_name} instruction={instruction_name} "
                    f"seed={seed} step={step_index}: {'; '.join(violations)}. info={info}"
                )

        action_postprocess = info.get("action_postprocess") or {}
        canonical_after = action_postprocess.get("canonical_command") or {}
        metadata = canonical_after.get("metadata") or {}
        raw_model_action = metadata.get("raw_model_action")
        adapted_translation = canonical_after.get("translation_m")

        robot_command = action_adapter.convert(policy_output.action)
        commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]
        gripper_executed = robot_command.gripper_command

        cosine_commanded = _cosine_similarity(commanded_translation, vector_to_object)
        sign_match_xyz = [_sign_match(commanded_translation[i], vector_to_object[i]) for i in range(3)]
        far_gripper_close = distance_before > FAR_GRIPPER_CLOSE_THRESHOLD_M and gripper_executed == "close"

        row = {
            "position_name": position_name,
            "instruction_name": instruction_name,
            "instruction": instruction,
            "seed": seed,
            "step": step_index,
            "mode": mode,
            "object_position": object_position,
            "ee_position": ee_before,
            "vector_ee_to_object": vector_to_object,
            "raw_model_action": raw_model_action,
            "adapted_translation": adapted_translation,
            "commanded_translation": commanded_translation,
            "cosine_commanded_vs_object": cosine_commanded,
            "sign_match_x": sign_match_xyz[0],
            "sign_match_y": sign_match_xyz[1],
            "sign_match_z": sign_match_xyz[2],
            "gripper_command": gripper_executed,
            "distance_to_object_before": distance_before,
            "far_gripper_close": far_gripper_close,
            "server_latency_ms": server_latency_ms,
            "main_image_hash": image_hash(main_image),
            "wrist_image_hash": image_hash(wrist_image),
            "compatibility_passed": compatibility_passed,
            "semantic_action_valid": semantic_action_valid,
            "degraded_input": degraded_input,
            "fallback_used": fallback_used,
        }
        rows.append(row)

        if mode == "multi-step":
            backend.apply_command(robot_command, steps=steps_per_action)

    if owns_backend:
        backend.shutdown()
    return rows


def run_benchmark(args, policy=None, positions=None, instructions=None) -> dict:
    positions = positions or DEFAULT_POSITIONS
    instructions = instructions or DEFAULT_INSTRUCTIONS

    owns_policy = policy is None
    if policy is None:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"counterfactual_{timestamp}.jsonl"

    all_rows = []
    total_conditions = len(positions) * len(instructions) * len(args.seeds)
    condition_number = 0
    print(f"=== Counterfactual direction benchmark -- {total_conditions} conditions -- log: {log_path} ===")
    for position_name, position in positions.items():
        for instruction_name, instruction in instructions.items():
            for seed in args.seeds:
                condition_number += 1
                rows = run_condition(
                    policy,
                    position_name,
                    position,
                    instruction_name,
                    instruction,
                    seed,
                    args.mode,
                    args.steps_per_condition,
                    args.steps_per_action,
                    args.object_type,
                    args.bin_position,
                    args.strict,
                )
                all_rows.extend(rows)
                mean_cosine = _mean([row["cosine_commanded_vs_object"] for row in rows if row["cosine_commanded_vs_object"] is not None])
                print(
                    f"[{condition_number:03d}/{total_conditions}] pos={position_name:<13} "
                    f"instr={instruction_name:<10} seed={seed:<4} mean_cos={_fmt(mean_cosine)}"
                )

    with open(log_path, "w", encoding="utf-8") as log_file:
        for row in all_rows:
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_benchmark(all_rows, positions)
    summary_path = output_dir / f"counterfactual_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nPer-step log:  {log_path}")
    print(f"Summary JSON:  {summary_path}")

    return {"rows": all_rows, "summary": summary, "log_path": str(log_path), "summary_path": str(summary_path)}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2.0
    return values[mid]


def _stdev(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _fmt(value) -> str:
    return f"{value:+.3f}" if value is not None else "n/a"


def _sign_accuracy(rows, axis: str) -> float:
    values = [row[f"sign_match_{axis}"] for row in rows if row[f"sign_match_{axis}"] is not None]
    return (sum(1 for v in values if v) / len(values)) if values else None


def group_stats(rows: list) -> dict:
    cosines = [row["cosine_commanded_vs_object"] for row in rows if row["cosine_commanded_vs_object"] is not None]
    return {
        "num_rows": len(rows),
        "mean_cosine": _mean(cosines),
        "median_cosine": _median(cosines),
        "positive_cosine_fraction": (sum(1 for c in cosines if c > 0) / len(cosines)) if cosines else None,
        "sign_accuracy_x": _sign_accuracy(rows, "x"),
        "sign_accuracy_y": _sign_accuracy(rows, "y"),
        "sign_accuracy_z": _sign_accuracy(rows, "z"),
        "far_gripper_close_rate": (
            sum(1 for row in rows if row["far_gripper_close"]) / len(rows) if rows else None
        ),
    }


def summarize_benchmark(rows: list, positions: dict) -> dict:
    by_position = {}
    for position_name in {row["position_name"] for row in rows}:
        by_position[position_name] = group_stats([row for row in rows if row["position_name"] == position_name])

    by_instruction = {}
    for instruction_name in {row["instruction_name"] for row in rows}:
        by_instruction[instruction_name] = group_stats([row for row in rows if row["instruction_name"] == instruction_name])

    by_cell = {}
    seed_std_per_cell = {}
    for position_name in {row["position_name"] for row in rows}:
        for instruction_name in {row["instruction_name"] for row in rows}:
            cell_rows = [
                row for row in rows if row["position_name"] == position_name and row["instruction_name"] == instruction_name
            ]
            if not cell_rows:
                continue
            key = f"{position_name}__{instruction_name}"
            by_cell[key] = group_stats(cell_rows)
            per_seed_means = []
            for seed in sorted({row["seed"] for row in cell_rows}):
                seed_rows = [row for row in cell_rows if row["seed"] == seed]
                per_seed_means.append(_mean([row["cosine_commanded_vs_object"] for row in seed_rows]))
            seed_std_per_cell[key] = _stdev(per_seed_means)

    overall = group_stats(rows)

    fixed_x_bias = judge_fixed_axis_bias(rows, positions, axis="x", mirror_pair=("center_right", "center_left"))
    fixed_y_bias = judge_fixed_axis_bias(rows, positions, axis="y", mirror_pair=("positive_y", "negative_y"))
    language_issue = judge_language_issue(by_instruction)
    domain_gap = judge_domain_gap(overall, by_position, by_instruction)
    seed_instability = judge_seed_instability(seed_std_per_cell)

    return {
        "overall": overall,
        "by_position": by_position,
        "by_instruction": by_instruction,
        "by_cell": by_cell,
        "seed_std_per_cell": seed_std_per_cell,
        "judgments": {
            "fixed_x_bias": fixed_x_bias,
            "fixed_y_bias": fixed_y_bias,
            "language_issue": language_issue,
            "domain_gap": domain_gap,
            "seed_instability": seed_instability,
        },
    }


def judge_fixed_axis_bias(rows: list, positions: dict, axis: str, mirror_pair: tuple) -> dict:
    """Mirroring the object across center (e.g. center_right vs.
    center_left) flips the sign of vector_ee_to_object on `axis`. If the
    model's mean commanded sign on that axis does NOT flip along with it
    -- i.e. sign_accuracy is high in one mirror position and necessarily
    low in the other, because the commanded sign stayed put -- that's a
    fixed directional bias, independent of where the object actually is.
    """
    position_a, position_b = mirror_pair
    if position_a not in positions or position_b not in positions:
        return {"suspected": None, "reason": f"positions {mirror_pair} not both present in this run"}

    rows_a = [row for row in rows if row["position_name"] == position_a]
    rows_b = [row for row in rows if row["position_name"] == position_b]
    if not rows_a or not rows_b:
        return {"suspected": None, "reason": f"no rows for one of {mirror_pair}"}

    mean_signed_commanded_a = _mean([row["commanded_translation"][_axis_index(axis)] for row in rows_a])
    mean_signed_commanded_b = _mean([row["commanded_translation"][_axis_index(axis)] for row in rows_b])
    sign_accuracy_a = _sign_accuracy(rows_a, axis)
    sign_accuracy_b = _sign_accuracy(rows_b, axis)

    same_commanded_sign = (
        mean_signed_commanded_a is not None
        and mean_signed_commanded_b is not None
        and (mean_signed_commanded_a > 0) == (mean_signed_commanded_b > 0)
        and abs(mean_signed_commanded_a) > SIGN_EPSILON
        and abs(mean_signed_commanded_b) > SIGN_EPSILON
    )
    # vector_ee_to_object's sign on this axis is expected to differ
    # between the two mirror positions (that's the whole point of
    # mirroring it) -- confirm that actually happened in this run before
    # trusting the comparison.
    mean_vector_a = _mean([row["vector_ee_to_object"][_axis_index(axis)] for row in rows_a])
    mean_vector_b = _mean([row["vector_ee_to_object"][_axis_index(axis)] for row in rows_b])
    vector_actually_flipped = (
        mean_vector_a is not None and mean_vector_b is not None and (mean_vector_a > 0) != (mean_vector_b > 0)
    )

    suspected = bool(vector_actually_flipped and same_commanded_sign)
    reason = (
        f"vector_ee_to_object.{axis} flips sign between {position_a} ({mean_vector_a:+.3f}) and {position_b} "
        f"({mean_vector_b:+.3f}) as expected, but mean commanded_translation.{axis} does NOT "
        f"({position_a}={mean_signed_commanded_a:+.3f}, {position_b}={mean_signed_commanded_b:+.3f}) -- "
        f"fixed {axis}-axis bias suspected."
        if suspected
        else (
            f"commanded_translation.{axis} sign tracks the object's actual side "
            f"({position_a}={mean_signed_commanded_a}, {position_b}={mean_signed_commanded_b}) -- no fixed bias evidence."
            if vector_actually_flipped
            else f"vector_ee_to_object.{axis} did not actually flip sign between {position_a}/{position_b} in this "
            "run -- cannot evaluate fixed-bias on this axis from this data."
        )
    )
    return {
        "suspected": suspected if vector_actually_flipped else None,
        "reason": reason,
        "mean_commanded": {position_a: mean_signed_commanded_a, position_b: mean_signed_commanded_b},
        "mean_vector_to_object": {position_a: mean_vector_a, position_b: mean_vector_b},
        "sign_accuracy": {position_a: sign_accuracy_a, position_b: sign_accuracy_b},
    }


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def judge_language_issue(by_instruction: dict) -> dict:
    """One instruction (particularly ko_full vs. the en_* variants)
    scoring much higher mean cosine than the others suggests the
    instruction WORDING/LANGUAGE itself matters to this checkpoint,
    rather than the scene."""
    means = {name: stats["mean_cosine"] for name, stats in by_instruction.items() if stats["mean_cosine"] is not None}
    if len(means) < 2:
        return {"suspected": None, "reason": "fewer than 2 instructions with data"}

    best_name = max(means, key=means.get)
    worst_name = min(means, key=means.get)
    gap = means[best_name] - means[worst_name]
    ko_names = [name for name in means if name.startswith("ko")]
    en_names = [name for name in means if name.startswith("en")]
    ko_mean = _mean([means[name] for name in ko_names]) if ko_names else None
    en_mean = _mean([means[name] for name in en_names]) if en_names else None
    ko_en_gap = (en_mean - ko_mean) if (ko_mean is not None and en_mean is not None) else None

    suspected = gap > LANGUAGE_ISSUE_COSINE_GAP
    return {
        "suspected": suspected,
        "reason": (
            f"mean cosine spread across instructions is {gap:.3f} (best={best_name}:{means[best_name]:.3f}, "
            f"worst={worst_name}:{means[worst_name]:.3f}), above the {LANGUAGE_ISSUE_COSINE_GAP} threshold -- "
            "instruction wording/language appears to matter."
            if suspected
            else f"mean cosine spread across instructions is only {gap:.3f} -- instructions are not the "
            "dominant factor here."
        ),
        "mean_cosine_by_instruction": means,
        "ko_mean_cosine": ko_mean,
        "en_mean_cosine": en_mean,
        "en_minus_ko_gap": ko_en_gap,
    }


def judge_domain_gap(overall: dict, by_position: dict, by_instruction: dict) -> dict:
    """If mean cosine stays low across EVERY position and EVERY
    instruction (no condition stands out as "good"), a directional bias
    or a single bad instruction can't explain it -- points at the visual
    scene / checkpoint-task mismatch instead."""
    all_position_means = [stats["mean_cosine"] for stats in by_position.values() if stats["mean_cosine"] is not None]
    all_instruction_means = [stats["mean_cosine"] for stats in by_instruction.values() if stats["mean_cosine"] is not None]
    best_position_mean = max(all_position_means) if all_position_means else None
    best_instruction_mean = max(all_instruction_means) if all_instruction_means else None

    suspected = (
        overall["mean_cosine"] is not None
        and overall["mean_cosine"] < DOMAIN_GAP_COSINE_CEILING
        and (best_position_mean is None or best_position_mean < DOMAIN_GAP_COSINE_CEILING + 0.15)
        and (best_instruction_mean is None or best_instruction_mean < DOMAIN_GAP_COSINE_CEILING + 0.15)
    )
    return {
        "suspected": suspected,
        "reason": (
            f"overall mean cosine={overall['mean_cosine']:.3f} and no position/instruction stands out as clearly "
            "better -- consistent with a visual/domain gap or checkpoint-task mismatch rather than a fixable bias "
            "or wording issue."
            if suspected
            else f"overall mean cosine={overall['mean_cosine']} -- at least one position or instruction performs "
            "meaningfully better, so a blanket domain-gap explanation is not supported by itself."
        ),
        "overall_mean_cosine": overall["mean_cosine"],
        "best_position_mean_cosine": best_position_mean,
        "best_instruction_mean_cosine": best_instruction_mean,
    }


def judge_seed_instability(seed_std_per_cell: dict) -> dict:
    stds = [value for value in seed_std_per_cell.values() if value is not None]
    if not stds:
        return {"suspected": None, "reason": "no cell had more than 1 seed with data"}
    mean_std = sum(stds) / len(stds)
    high_variance_cells = {key: value for key, value in seed_std_per_cell.items() if value is not None and value > SEED_INSTABILITY_STD_THRESHOLD}
    suspected = mean_std > SEED_INSTABILITY_STD_THRESHOLD or len(high_variance_cells) > len(stds) / 2
    return {
        "suspected": suspected,
        "reason": (
            f"mean across-seed std of cosine is {mean_std:.3f} (> {SEED_INSTABILITY_STD_THRESHOLD}), and "
            f"{len(high_variance_cells)}/{len(stds)} cells exceed that threshold -- seed/sampling noise is a "
            "major contributor, not just scene/instruction."
            if suspected
            else f"mean across-seed std of cosine is {mean_std:.3f} -- seeds broadly agree within a cell, so "
            "sampling noise is not the dominant factor."
        ),
        "mean_seed_std": mean_std,
        "high_variance_cell_count": len(high_variance_cells),
        "total_cells_with_multiple_seeds": len(stds),
    }


def print_summary(summary: dict) -> None:
    print("\n=== Overall ===")
    for key, value in summary["overall"].items():
        print(f"{key}: {value}")

    print("\n=== By position ===")
    for name, stats in summary["by_position"].items():
        print(f"{name}: mean_cosine={_fmt(stats['mean_cosine'])} sign_acc(x,y,z)="
              f"({_fmt(stats['sign_accuracy_x'])},{_fmt(stats['sign_accuracy_y'])},{_fmt(stats['sign_accuracy_z'])})")

    print("\n=== By instruction ===")
    for name, stats in summary["by_instruction"].items():
        print(f"{name}: mean_cosine={_fmt(stats['mean_cosine'])} positive_fraction={stats['positive_cosine_fraction']}")

    print("\n=== Judgments ===")
    for name, judgment in summary["judgments"].items():
        print(f"{name}: suspected={judgment['suspected']}")
        print(f"  reason: {judgment['reason']}")


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
