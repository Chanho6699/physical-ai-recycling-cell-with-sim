"""VLA action direction diagnostic (v0).

Standalone diagnostic tool -- does NOT modify run_full_recycling_cell_demo.py
or any production code path. Its only job is to separate three possible
sources of "the robot isn't actually approaching the object" into three
layers, per step:

  1. MODEL      -- does the raw SmolVLA action (before any adapter/scale/
                   safety-filter touches it) already point away from the
                   object in the model's own translation channels?
  2. ADAPTER    -- does SmolVLALiberoActionAdapter's decode() (scale to
                   meters, axis mapping, gripper polarity) distort the
                   direction relative to what the model actually said?
  3. EXECUTOR   -- does PyBulletPandaBackend.apply_command() actually move
                   the end effector in the direction it was commanded?

It reuses the exact same real, already-tested pieces this project's
production closed loop uses -- PyBulletPandaBackend, RealVLAPolicyClient,
ActionAdapter, and the same info-dict fields
benchmark/run_full_recycling_cell_demo.py's real_vla_step_log already
threads through (canonical_command_pre_safety_filter/after,
raw_model_action, gripper_raw, etc, see vla_adapters/smolvla_adapter.py
and policy_semantics/adapters/smolvla_libero_adapter.py). It does not
call --strict-real-vla's code path in that file (this is a separate
script), but enforces the same conditions independently via
--strict (default on).

Also see this module's SERVER-SIDE INVESTIGATION NOTES below (module
docstring continued) for what was found reading
vla_server/model_loader.py + the installed lerobot package about the
SmolVLA action-chunk queue, and MAIN/WRIST CAMERA VERIFICATION NOTES for
what was found about the images_by_role -> observation.images.{image,image2}
mapping. Both are summarized in the final report, not just here.

Run:
  python -m benchmark.run_vla_action_direction_diagnostic \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json \\
    --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \\
    --object-position 0.3758 0.0204 0.05 \\
    --max-steps 20
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d, _unit_vector
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/vla_backend_smolvla_libero_config.json"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"

# A queued (not freshly inferred) SmolVLA action would return in a few ms
# (deque.popleft(), no GPU forward pass) -- a real forward pass observed
# in this project's own live runs takes ~600-850ms (see final report).
# This is a heuristic, not a certainty (see module docstring / final
# report for why HuggingFaceVLA/smolvla_libero's own config makes this
# unlikely in practice), which is why it's flagged, not asserted.
SUSPECTED_QUEUED_ACTION_LATENCY_MS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--object-position", type=float, nargs=3, default=[0.3758, 0.0204, 0.05], metavar=("X", "Y", "Z")
    )
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Raise immediately if compatibility_passed/semantic_action_valid/degraded_input/"
        "fallback_used are ever violated (default: on -- this diagnostic exists to isolate a "
        "real, non-degraded, non-fallback model's behavior, same spirit as "
        "run_full_recycling_cell_demo.py's --strict-real-vla).",
    )
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--save-images", action="store_true", help="Save every step's main/wrist frame to --output-dir.")
    parser.add_argument("--output-dir", type=str, default="results/vla_action_direction_diagnostic")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def image_hash(image: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(image).tobytes()).hexdigest()[:12]


def build_robot_state(backend: PyBulletPandaBackend) -> tuple:
    """(robot_state dict with the 3 LIBERO 8D-state fields merged in,
    state_8d list, object_position list) -- same construction
    run_full_recycling_cell_demo.py's --real-vla-observation-mode
    pybullet path uses."""
    state = backend.get_state()
    state_8d = backend.get_libero_observation_state()
    robot_state = {
        **state,
        "ee_position": list(state_8d[0:3]),
        "ee_orientation_axis_angle": list(state_8d[3:6]),
        "gripper_qpos": list(state_8d[6:8]),
    }
    return robot_state, state_8d, list(state["object_position"])


def run_diagnostic(args, policy=None, backend=None) -> dict:
    """policy/backend are injectable (default: real RealVLAPolicyClient /
    real PyBulletPandaBackend) purely so
    benchmark/test_vla_action_direction_diagnostic.py can exercise this
    exact function -- row-building, cosine wiring, strict enforcement,
    summary -- against a controllable fake policy without a live GPU
    server. Production callers (main() below) never pass these."""
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"diagnostic_{timestamp}.jsonl"
    images_dir = output_dir / f"images_{timestamp}"
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    owns_backend = backend is None
    if backend is None:
        backend = PyBulletPandaBackend(gui=args.gui)
        backend.reset()
        backend.set_object_type(args.object_type)
        backend.set_object_position(list(args.object_position))

    if policy is None:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)
    policy.reset()
    action_adapter = ActionAdapter()

    rows = []
    print(f"=== VLA action direction diagnostic -- log: {log_path} ===")
    for step_index in range(args.max_steps):
        robot_state, state_8d, object_position = build_robot_state(backend)
        ee_before = list(state_8d[0:3])
        vector_to_object = [object_position[i] - ee_before[i] for i in range(3)]
        distance_before = _distance_3d(ee_before, object_position)

        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()
        main_hash = image_hash(main_image)
        wrist_hash = image_hash(wrist_image)
        same_image = bool(np.array_equal(main_image, wrist_image))

        main_image_path = None
        wrist_image_path = None
        if args.save_images:
            main_image_path = str(images_dir / f"step_{step_index:03d}_main.png")
            wrist_image_path = str(images_dir / f"step_{step_index:03d}_wrist.png")
            save_rgb_image(main_image, main_image_path)
            save_rgb_image(wrist_image, wrist_image_path)

        policy_input = PolicyInput(
            image=main_image,
            instruction=args.instruction,
            robot_state=robot_state,
            task_goal={},
            target_object_position=object_position,
            bin_position=[0.3, 0.35, 0.05],
            step_index=step_index,
            phase="move_to_object",
            images_by_role={"main": main_image, "wrist": wrist_image},
            seed=(args.seed + step_index) if args.seed is not None else None,
        )

        observation_timestamp = time.time()
        policy_output = policy.predict_action(policy_input)
        info = policy_output.info or {}

        compatibility_passed = (info.get("compatibility") or {}).get("passed")
        semantic_action_valid = bool(info.get("semantic_action_valid", True))
        degraded_input = bool(info.get("degraded_input", False))
        fallback_used = bool(info.get("fallback_used", False))
        server_latency_ms = info.get("inference_latency_ms")
        suspected_queued_action = (
            server_latency_ms is not None and server_latency_ms < SUSPECTED_QUEUED_ACTION_LATENCY_MS
        )

        if args.strict:
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
                raise RuntimeError(f"--strict violated at step {step_index}: {'; '.join(violations)}. info={info}")

        action_postprocess = info.get("action_postprocess") or {}
        canonical_after = action_postprocess.get("canonical_command") or {}
        canonical_before = action_postprocess.get("canonical_command_pre_safety_filter") or {}
        metadata = canonical_after.get("metadata") or {}

        raw_model_action = metadata.get("raw_model_action")
        raw_model_translation = list(raw_model_action[0:3]) if raw_model_action else None
        adapted_canonical_translation = canonical_before.get("translation_m")
        safety_filtered_translation = canonical_after.get("translation_m")

        robot_command = action_adapter.convert(policy_output.action)
        commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]

        gripper_raw = metadata.get("gripper_raw")
        gripper_canonical = canonical_after.get("gripper_opening_01")
        gripper_executed = robot_command.gripper_command

        state_after = backend.apply_command(robot_command, steps=args.steps_per_action)
        ee_after = list(state_after["end_effector_position"])
        actual_ee_displacement = [ee_after[i] - ee_before[i] for i in range(3)]
        distance_after = _distance_3d(ee_after, object_position)
        distance_progress = distance_before - distance_after

        cosine_raw = _cosine_similarity(raw_model_translation, vector_to_object) if raw_model_translation else None
        cosine_commanded = _cosine_similarity(commanded_translation, vector_to_object)
        cosine_actual = _cosine_similarity(actual_ee_displacement, vector_to_object)

        row = {
            "step": step_index,
            "timestamp": observation_timestamp,
            "instruction": args.instruction,
            "ee_position_before": ee_before,
            "object_position": object_position,
            "vector_ee_to_object": vector_to_object,
            "normalized_vector_ee_to_object": _unit_vector(vector_to_object),
            "raw_model_action": raw_model_action,
            "raw_model_translation": raw_model_translation,
            "adapted_canonical_translation": adapted_canonical_translation,
            "safety_filtered_translation": safety_filtered_translation,
            "commanded_translation": commanded_translation,
            "ee_position_after": ee_after,
            "actual_ee_displacement": actual_ee_displacement,
            "distance_to_object_before": distance_before,
            "distance_to_object_after": distance_after,
            "distance_progress": distance_progress,
            "cosine_raw_vs_object": cosine_raw,
            "cosine_commanded_vs_object": cosine_commanded,
            "cosine_actual_vs_object": cosine_actual,
            "gripper_raw": gripper_raw,
            "gripper_canonical": gripper_canonical,
            "gripper_executed": gripper_executed,
            "main_image_path": main_image_path,
            "main_image_hash": main_hash,
            "wrist_image_path": wrist_image_path,
            "wrist_image_hash": wrist_hash,
            "main_image_shape": list(main_image.shape),
            "wrist_image_shape": list(wrist_image.shape),
            "main_wrist_identical": same_image,
            "degraded_input": degraded_input,
            "fallback_used": fallback_used,
            "compatibility_passed": compatibility_passed,
            "semantic_action_valid": semantic_action_valid,
            "server_latency_ms": server_latency_ms,
            "suspected_queued_action": suspected_queued_action,
        }
        rows.append(row)

        print(
            f"[step {step_index:02d}] dist={distance_before:.3f}->{distance_after:.3f} "
            f"(progress={distance_progress:+.3f}) cos_raw={_fmt(cosine_raw)} "
            f"cos_cmd={_fmt(cosine_commanded)} cos_act={_fmt(cosine_actual)} "
            f"gripper={gripper_executed} latency_ms={server_latency_ms} "
            f"{'[SUSPECTED_QUEUED_ACTION]' if suspected_queued_action else ''}"
        )

    with open(log_path, "w", encoding="utf-8") as log_file:
        for row in rows:
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if owns_backend:
        backend.shutdown()

    summary = summarize(rows)
    summary_path = output_dir / f"diagnostic_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nPer-step log:  {log_path}")
    print(f"Summary JSON:  {summary_path}")
    if args.save_images:
        print(f"Images:        {images_dir}")

    return {"rows": rows, "summary": summary, "log_path": str(log_path), "summary_path": str(summary_path)}


def _fmt(value) -> str:
    return f"{value:+.3f}" if value is not None else "n/a"


def summarize(rows: list) -> dict:
    if not rows:
        return {"num_steps": 0}

    distances = [row["distance_to_object_before"] for row in rows] + [rows[-1]["distance_to_object_after"]]
    total_distance_improvement = distances[0] - distances[-1]

    def _series(key):
        return [row[key] for row in rows if row[key] is not None]

    def _mean(values):
        return sum(values) / len(values) if values else None

    def _median(values):
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 0:
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        return ordered[mid]

    def _positive_fraction(values):
        return (sum(1 for value in values if value > 0) / len(values)) if values else None

    cosine_raw_series = _series("cosine_raw_vs_object")
    cosine_commanded_series = _series("cosine_commanded_vs_object")
    cosine_actual_series = _series("cosine_actual_vs_object")

    commanded_axis_sums = [sum(row["commanded_translation"][i] for row in rows) for i in range(3)]
    actual_axis_sums = [sum(row["actual_ee_displacement"][i] for row in rows) for i in range(3)]

    far_gripper_close_count = sum(
        1 for row in rows if row["distance_to_object_before"] > 0.15 and row["gripper_executed"] == "close"
    )

    mean_raw = _mean(cosine_raw_series)
    mean_commanded = _mean(cosine_commanded_series)
    mean_actual = _mean(cosine_actual_series)

    suspected_layer, suspected_reason = diagnose_layer(mean_raw, mean_commanded, mean_actual)

    return {
        "num_steps": len(rows),
        "total_distance_improvement": total_distance_improvement,
        "distance_before_first": distances[0],
        "distance_after_last": distances[-1],
        "cosine_raw_vs_object": {
            "mean": mean_raw,
            "median": _median(cosine_raw_series),
            "positive_fraction": _positive_fraction(cosine_raw_series),
        },
        "cosine_commanded_vs_object": {
            "mean": mean_commanded,
            "median": _median(cosine_commanded_series),
            "positive_fraction": _positive_fraction(cosine_commanded_series),
        },
        "cosine_actual_vs_object": {
            "mean": mean_actual,
            "median": _median(cosine_actual_series),
            "positive_fraction": _positive_fraction(cosine_actual_series),
        },
        "commanded_translation_axis_sums_xyz": commanded_axis_sums,
        "actual_displacement_axis_sums_xyz": actual_axis_sums,
        "far_gripper_close_count": far_gripper_close_count,
        "far_gripper_close_threshold_m": 0.15,
        "suspected_queued_action_steps": sum(1 for row in rows if row["suspected_queued_action"]),
        "most_suspected_layer": suspected_layer,
        "most_suspected_layer_reason": suspected_reason,
    }


def diagnose_layer(mean_raw, mean_commanded, mean_actual) -> tuple:
    """A simple, explicit decision rule -- not a statistical test.

    IMPORTANT (found while validating this script against a live server --
    see the final report): mean_raw is NOT a fair apples-to-apples
    direction comparison against a real-world meter vector. raw_model_action
    is captured BEFORE vla_server/model_loader.py's
    postprocessor_pipeline() call -- i.e. before LeRobot's own official
    per-channel MEAN_STD unnormalization. If the checkpoint's training
    data had different x/y/z delta variances, the *same* real-world
    direction looks different in this pre-postprocessor space purely from
    that per-axis scale, with no adapter or model bug involved. So this
    function anchors its verdict on mean_commanded/mean_actual (both
    already past the official postprocessor and past the adapter's own
    single uniform TRANSLATION_SCALE_M -- i.e. both are real, physically-
    comparable meter vectors) and only ever reports mean_raw as an
    informational note, never as the reason for a "model" or "adapter"
    verdict on its own.
    """
    if mean_commanded is None:
        return "unknown", "no commanded_translation recorded (no successful steps to diagnose)"

    raw_note = (
        f" (raw_model_action's own mean cosine was {mean_raw:.3f}, but that value is pre-official-postprocessor "
        "and confounded by this checkpoint's per-axis normalization scale -- informational only, not used below)"
        if mean_raw is not None
        else " (no raw_model_action was recorded at all)"
    )

    if mean_commanded < 0.1:
        return (
            "model",
            f"mean cosine(commanded_translation, vector_to_object)={mean_commanded:.3f} -- the real, "
            "physically-calibrated commanded direction (post official-postprocessor, post adapter scale) does "
            f"not point toward the object on average.{raw_note}",
        )

    if mean_actual is not None and abs(mean_commanded - mean_actual) > 0.3:
        return (
            "executor",
            f"mean cosine(commanded)={mean_commanded:.3f} vs mean cosine(actual displacement)={mean_actual:.3f} -- "
            "the backend's actual EE displacement diverges from what was commanded, even though the commanded "
            f"direction itself was reasonable -- check PyBulletPandaBackend.apply_command()'s axis convention.{raw_note}",
        )

    return (
        "model",
        f"commanded/actual cosines are similar and weakly-to-moderately positive (commanded={mean_commanded:.3f}, "
        f"actual={mean_actual}) -- the executor faithfully reproduces the commanded direction, so if the object "
        "still isn't being reliably approached, the model's own policy quality (not the plumbing) is the most "
        f"likely factor.{raw_note}",
    )


def print_summary(summary: dict) -> None:
    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    args = parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
