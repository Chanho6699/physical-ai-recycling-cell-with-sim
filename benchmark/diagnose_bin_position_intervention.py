"""Causal intervention test: is checkpoint 4000's gripper-release
decision conditioned on the bin's VISUAL position, or is it a
fixed timing/trajectory shortcut learned from train80's single,
never-varied bin position (see this task's chat report)?

SmolVLA is a vision-language-action model -- it never receives
bin_position as a numeric/text feature (confirmed by reading
vla_adapters/smolvla_adapter.py's build_model_input(): only instruction/
image/images_by_role/robot_state/step_index/phase/seed are forwarded,
never bin_position or target_object_position). The ONLY channel through
which the model could possibly know where the bin is is by SEEING it in
the rendered camera images. So this script actually moves the bin
object in the PyBullet simulator itself
(PyBulletPandaBackend.set_bin_position(), which both teleports the real
body -- so it appears in a different place in every subsequent render --
and updates the backend's own internal success-check position) rather
than just changing a bookkeeping value -- moving only the latter would
make this test meaningless (the model would have zero way to react).

Reuses the exact same fixed model_input construction (seed passed
through for deterministic flow-matching sampling, image JPEG round-trip
matching the real HTTP pipeline) already validated in
benchmark/diagnose_gripper_premature_release.py.

Conditions per (position, seed): control (bin at its original position)
plus 4 shifted conditions (+-0.05m in x [front/back] and y
[left/right]), independently, never combined.

Does not modify the model, threshold, decoder, dataset, or policy step
budget. Does not retrain. Only calls set_bin_position(), which is an
existing production method already used elsewhere in this project's own
benchmark scripts, never a new production code path.

Run:
  .venv-vla/bin/python -m benchmark.diagnose_bin_position_intervention \\
    --output results/gripper_diagnosis/bin_position_intervention.json
"""

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path

os.environ.setdefault("VLA_DEVICE", "cuda")
os.environ.setdefault("VLA_DTYPE", "float32")

import numpy as np
from PIL import Image

from action_adapter.adapter_v0 import RobotCommand
from benchmark.checkpoint_eval_positions import build_train_eval_positions, build_validation_eval_positions
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state
from policy.vla_image_preprocessor import encode_policy_image_for_vla
from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import get_manifest
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "outputs/train/smolvla_recycling_train80_v1/checkpoints/004000/pretrained_model"
ORIGINAL_BIN_POSITION = [0.3, 0.35, 0.05]
MAX_STEPS = 80
STEPS_PER_ACTION = 40
SHIFT_M = 0.05

# +x = "front" (away from robot base), -x = "back"; +y = "left", -y = "right"
# -- same axis convention as this project's own position grid
# (benchmark/train80_validation20_positions.py's GRID_Y: +y=left, -y=right).
BIN_SHIFT_CONDITIONS = {
    "control": [0.0, 0.0, 0.0],
    "front_+0.05x": [SHIFT_M, 0.0, 0.0],
    "back_-0.05x": [-SHIFT_M, 0.0, 0.0],
    "left_+0.05y": [0.0, SHIFT_M, 0.0],
    "right_-0.05y": [0.0, -SHIFT_M, 0.0],
}

# The (anchor, seed) pairs that produced a grasp in the checkpoint being
# tested's own prior 40-episode benchmark run -- prioritized per this
# task's explicit instruction ("가능하면 이전에 pick이 발생한 6개 seed를
# 우선 사용"). DEFAULT_PICKED_TARGETS matches the original train80
# 4000-step diagnosis (6 picks); v2 checkpoints pass their own picked
# list explicitly (see main()'s --picked-targets-from).
DEFAULT_PICKED_TARGETS = [
    ("train_x1_y0", 208201),
    ("train_x2_y3", 209200),
    ("train_x2_y3", 209201),
    ("val_west", 301301),
    ("val_near", 303302),
    ("val_near", 303303),
]

REAL_VLA_CONFIG = json.loads((PROJECT_ROOT / "configs/real_vla_backend_config.json").read_text(encoding="utf-8"))


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _round_trip_image(image: np.ndarray) -> np.ndarray:
    payload, _debug = encode_policy_image_for_vla(image, REAL_VLA_CONFIG)
    image_bytes = base64.b64decode(payload["data"])
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)


def load_checkpoint(checkpoint: str = DEFAULT_CHECKPOINT):
    from vla_server import model_loader

    model_loader._state["status"] = "unloaded"
    model_loader._state["model_family"] = None
    model_loader._state["model"] = None
    result = model_loader.load_model_once("smolvla", checkpoint, local_files_only=True)
    assert result["status"] == "loaded", result
    manifest = get_manifest(checkpoint)
    gate = CompatibilityGate.check(manifest)
    assert gate.passed, f"CompatibilityGate failed: {gate.reasons}"
    return manifest, model_loader


def run_episode(model_loader, manifest, anchor_name, position, seed, instruction, bin_position, condition_name):
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    backend.set_object_position(list(position))
    backend.set_bin_position(list(bin_position))
    adapter = SmolVLALiberoActionAdapter()

    rows = []
    ever_held = False
    first_grasp_step = None
    release_step = None
    grasp_object_position = None
    final_status = "running"

    for step_index in range(MAX_STEPS):
        robot_state, _, object_position = build_robot_state(backend)
        ee_position = list(robot_state["ee_position"])

        main_image = _round_trip_image(backend.render_main_camera())
        wrist_image = _round_trip_image(backend.render_wrist_camera())
        model_input = {
            "instruction": instruction,
            "image": main_image,
            "images_by_role": {"main": main_image, "wrist": wrist_image},
            "robot_state": robot_state,
            "task_goal": {},
            "target_object_position": object_position,
            "bin_position": bin_position,
            "step_index": step_index,
            "phase": "move_to_object",
            "seed": seed,
        }
        native = model_loader.run_inference("smolvla", model_input)
        raw_gripper = float(native.values[6])
        command = adapter.decode(native, manifest, context={"degraded_input": False})
        if command is None:
            rows.append({"step": step_index, "decode_failed": True})
            break

        gripper_opening_01 = command.gripper_opening_01
        gripper_label = "close" if gripper_opening_01 <= 0.5 else "open"

        robot_command = RobotCommand(
            target_dx=command.translation_m[0], target_dy=command.translation_m[1], target_dz=command.translation_m[2],
            target_droll=command.rotation_axis_angle_rad[0], target_dpitch=command.rotation_axis_angle_rad[1],
            target_dyaw=command.rotation_axis_angle_rad[2], gripper_command=gripper_label,
        )

        robot_state_after = backend.apply_command(robot_command, steps=STEPS_PER_ACTION)
        held_now = bool(robot_state_after["held_object"])
        final_status = robot_state_after["task_status"]

        if held_now and not ever_held:
            ever_held = True
            first_grasp_step = step_index
            grasp_object_position = list(object_position)
        if ever_held and not held_now and release_step is None and step_index != first_grasp_step:
            release_step = step_index
            release_ee_position = ee_position
            release_object_position = list(object_position)
            release_raw_gripper = raw_gripper

        rows.append({
            "step": step_index,
            "raw_native_gripper": raw_gripper,
            "gripper_opening_01_decoded": gripper_opening_01,
            "gripper_label": gripper_label,
            "held_object": held_now,
            "task_status": final_status,
            "object_position": list(object_position),
            "ee_position": ee_position,
        })

        if final_status == "success":
            break

    backend.shutdown()

    result = {
        "condition": condition_name,
        "bin_position": list(bin_position),
        "anchor_name": anchor_name,
        "seed": seed,
        "position": list(position),
        "num_steps": len(rows),
        "pick_success": ever_held,
        "first_grasp_step": first_grasp_step,
        "release_step": release_step,
        "final_task_status": final_status,
        "success": final_status == "success",
    }

    if ever_held and release_step is not None:
        result["grasp_object_position"] = grasp_object_position
        result["release_ee_position"] = release_ee_position
        result["release_object_position"] = release_object_position
        result["release_raw_gripper"] = release_raw_gripper
        result["distance_release_object_to_moved_bin"] = _distance_3d(release_object_position, bin_position)
        result["distance_release_object_to_original_bin"] = _distance_3d(release_object_position, ORIGINAL_BIN_POSITION)
        result["distance_release_ee_to_moved_bin"] = _distance_3d(release_ee_position, bin_position)
        result["distance_release_ee_to_original_bin"] = _distance_3d(release_ee_position, ORIGINAL_BIN_POSITION)

        carry_vector = [release_object_position[i] - grasp_object_position[i] for i in range(3)]
        vector_to_moved_bin = [bin_position[i] - grasp_object_position[i] for i in range(3)]
        vector_to_original_bin = [ORIGINAL_BIN_POSITION[i] - grasp_object_position[i] for i in range(3)]
        carry_progress_toward_moved_bin = (
            _distance_3d(grasp_object_position, bin_position) - _distance_3d(release_object_position, bin_position)
        )
        carry_progress_toward_original_bin = (
            _distance_3d(grasp_object_position, ORIGINAL_BIN_POSITION) - _distance_3d(release_object_position, ORIGINAL_BIN_POSITION)
        )
        result["carry_cosine_vs_moved_bin"] = _cosine_similarity(carry_vector, vector_to_moved_bin)
        result["carry_cosine_vs_original_bin"] = _cosine_similarity(carry_vector, vector_to_original_bin)
        result["carry_progress_toward_moved_bin_m"] = carry_progress_toward_moved_bin
        result["carry_progress_toward_original_bin_m"] = carry_progress_toward_original_bin

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--picked-targets", type=str, default=None,
        help="JSON list of [anchor_name, seed] pairs (this checkpoint's OWN picks from its prior "
        "40-episode benchmark run, per this task's 'use picks from the checkpoint under test' "
        "instruction). Defaults to the original train80 4000-step checkpoint's picks.",
    )
    args = parser.parse_args()

    picked_targets = DEFAULT_PICKED_TARGETS
    if args.picked_targets:
        picked_targets = [tuple(pair) for pair in json.loads(args.picked_targets)]

    manifest, model_loader = load_checkpoint(args.checkpoint)
    instruction = DEFAULT_INSTRUCTIONS["ko_full"]

    all_positions = {(p["anchor_name"], p["seed"]): p for p in build_train_eval_positions() + build_validation_eval_positions()}

    episodes = []
    total = len(picked_targets) * len(BIN_SHIFT_CONDITIONS)
    n = 0
    for anchor_name, seed in picked_targets:
        p = all_positions[(anchor_name, seed)]
        for condition_name, shift in BIN_SHIFT_CONDITIONS.items():
            n += 1
            bin_position = [ORIGINAL_BIN_POSITION[i] + shift[i] for i in range(3)]
            episode = run_episode(model_loader, manifest, anchor_name, p["position"], seed, instruction, bin_position, condition_name)
            episodes.append(episode)
            print(
                f"[{n:02d}/{total}] {anchor_name:16s} seed={seed:7d} condition={condition_name:14s} "
                f"pick={episode['pick_success']} grasp_step={episode['first_grasp_step']} release_step={episode['release_step']} "
                f"dist_to_moved_bin={episode.get('distance_release_object_to_moved_bin')}"
            )

    output = {"checkpoint": args.checkpoint, "max_steps": MAX_STEPS, "shift_m": SHIFT_M, "conditions": list(BIN_SHIFT_CONDITIONS.keys()), "episodes": episodes}
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
