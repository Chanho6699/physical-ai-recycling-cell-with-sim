"""In-process rollout diagnostic for the 4000-step checkpoint's
premature-release behavior (see this task's chat report). Runs the
SAME 40 episodes (train_eval + validation_eval positions/seeds from
benchmark/checkpoint_eval_positions.py) already used for the
zero-shot/2000/4000-step comparison, but talks to the loaded model
IN-PROCESS via vla_server.model_loader (same pattern as
benchmark/verify_checkpoint_loading.py) instead of through the HTTP
server + RealVLAPolicyClient -- because the HTTP response only ever
carries the FINAL decoded action, never the raw native network gripper
value, and this diagnosis specifically needs that raw value (to tell a
sudden flip from a gradual drift) alongside the fully-decoded
gripper_opening_01, the thresholded open/close label, and physical
distances -- all at once, every step.

This does not modify vla_server/model_loader.py, the adapter, the
decoder, or any threshold; it only reads values that already exist at
each pipeline stage (NativePolicyAction.values[6] -- raw; decode()'s
CanonicalRobotCommand.gripper_opening_01 -- decoded; <= 0.5 threshold --
the same one canonical_command.py's own to_legacy_action_list() uses).

Run:
  .venv-vla/bin/python -m benchmark.diagnose_gripper_premature_release \\
    --output results/gripper_diagnosis/checkpoint_4000_raw_gripper_rollout.json
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
from benchmark.run_full_recycling_cell_demo import _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state
from policy.vla_image_preprocessor import encode_policy_image_for_vla
from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import get_manifest
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_VLA_CONFIG = json.loads((PROJECT_ROOT / "configs/real_vla_backend_config.json").read_text(encoding="utf-8"))


def _round_trip_image(image: np.ndarray) -> np.ndarray:
    """Reproduces EXACTLY what a real request/response cycle does to an
    image (resize + lossy JPEG re-encode client-side, per
    configs/real_vla_backend_config.json's image_encoding section --
    the same config benchmark/run_checkpoint_comparison_benchmark.py's
    RealVLAPolicyClient already uses -- then base64/JPEG decode
    server-side, matching vla_server/generic_vla_server.py's
    decode_request_image() exactly). An earlier version of this script
    passed the raw, uncompressed, non-resized image straight to
    model_loader.run_inference() instead -- a genuinely DIFFERENT (not
    just re-run) input from what the production HTTP pipeline actually
    fed the model, which produced a different (though still
    deterministic) rollout. This closes that gap so the in-process path
    and the HTTP path see byte-identical images."""
    payload, _debug = encode_policy_image_for_vla(image, REAL_VLA_CONFIG)
    image_bytes = base64.b64decode(payload["data"])
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)
CHECKPOINT = "outputs/train/smolvla_recycling_train80_v1/checkpoints/004000/pretrained_model"
BIN_POSITION = [0.3, 0.35, 0.05]
MAX_STEPS = 80
STEPS_PER_ACTION = 40


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint():
    from vla_server import model_loader

    model_loader._state["status"] = "unloaded"
    model_loader._state["model_family"] = None
    model_loader._state["model"] = None
    result = model_loader.load_model_once("smolvla", CHECKPOINT, local_files_only=True)
    assert result["status"] == "loaded", result
    manifest = get_manifest(CHECKPOINT)
    gate = CompatibilityGate.check(manifest)
    assert gate.passed, f"CompatibilityGate failed: {gate.reasons}"
    return manifest, model_loader


def run_episode(model_loader, manifest, anchor_name, split_tag, position, seed, instruction):
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    backend.set_object_position(list(position))
    adapter = SmolVLALiberoActionAdapter()

    rows = []
    ever_held = False
    first_grasp_step = None
    release_step = None
    final_status = "running"

    for step_index in range(MAX_STEPS):
        robot_state, _, object_position = build_robot_state(backend)
        ee_position = list(robot_state["ee_position"])
        distance_to_object = _distance_3d(ee_position, object_position)
        distance_to_bin = _distance_3d(ee_position, BIN_POSITION)
        object_distance_to_bin = _distance_3d(object_position, BIN_POSITION)

        main_image = _round_trip_image(backend.render_main_camera())
        wrist_image = _round_trip_image(backend.render_wrist_camera())
        model_input = {
            "instruction": instruction,
            "image": main_image,
            "images_by_role": {"main": main_image, "wrist": wrist_image},
            "robot_state": robot_state,
            "task_goal": {},
            "target_object_position": object_position,
            "bin_position": BIN_POSITION,
            "step_index": step_index,
            "phase": "move_to_object",
            # Matches benchmark/run_checkpoint_comparison_benchmark.py's
            # PolicyInput(..., seed=seed) exactly -- model_loader.run_inference()
            # calls torch.manual_seed(seed) when present (see its own "Optional
            # per-step seed" comment), which is what made THAT run's SmolVLA
            # flow-matching sampling deterministic/reproducible per episode.
            # Omitting it here (as an earlier version of this script did)
            # left sampling un-seeded -- a different, non-reproducible
            # rollout each time, not a real behavioral difference.
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
        distance_from_threshold = abs(gripper_opening_01 - 0.5)

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
        if ever_held and not held_now and release_step is None and step_index != first_grasp_step:
            release_step = step_index

        rows.append({
            "step": step_index,
            "raw_native_gripper": raw_gripper,
            "gripper_opening_01_decoded": gripper_opening_01,
            "gripper_label": gripper_label,
            "distance_from_threshold": distance_from_threshold,
            "held_object": held_now,
            "task_status": final_status,
            "distance_to_object": distance_to_object,
            "distance_ee_to_bin": distance_to_bin,
            "distance_object_to_bin": object_distance_to_bin,
            "object_position": list(object_position),
            "ee_position": ee_position,
            "translation_m": list(command.translation_m),
        })

        if final_status == "success":
            break

    backend.shutdown()
    return {
        "anchor_name": anchor_name,
        "split_tag": split_tag,
        "seed": seed,
        "position": list(position),
        "num_steps": len(rows),
        "pick_success": ever_held,
        "first_grasp_step": first_grasp_step,
        "release_step": release_step,
        "final_task_status": final_status,
        "success": final_status == "success",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest, model_loader = load_checkpoint()
    instruction = DEFAULT_INSTRUCTIONS["ko_full"]

    train_eval = build_train_eval_positions()
    validation_eval = build_validation_eval_positions()
    all_positions = [("train_distribution", p) for p in train_eval] + [("validation_distribution", p) for p in validation_eval]

    episodes = []
    for n, (split_tag, p) in enumerate(all_positions, start=1):
        episode = run_episode(model_loader, manifest, p["anchor_name"], split_tag, p["position"], p["seed"], instruction)
        episodes.append(episode)
        print(
            f"[{n:02d}/{len(all_positions)}] {split_tag:20s} {p['anchor_name']:16s} seed={p['seed']:7d} "
            f"pick={episode['pick_success']} grasp_step={episode['first_grasp_step']} "
            f"release_step={episode['release_step']} status={episode['final_task_status']}"
        )

    output = {"checkpoint": CHECKPOINT, "max_steps": MAX_STEPS, "episodes": episodes}
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
