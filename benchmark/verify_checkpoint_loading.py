"""Per-checkpoint loading verification (see this task's chat report,
item 3). For a given local checkpoint directory, checks:
  - local checkpoint loads via vla_server.model_loader (no server
    process needed -- calls the same load_model_once()/run_inference()
    functions the real server uses, in-process)
  - PolicyManifest generation (policy_semantics.manifest.get_manifest())
  - native translation/rotation scale + native gripper range/semantics
  - CompatibilityGate.check() PASS/FAIL with full reasons
  - one real sample-batch inference through the loaded model + the
    production SmolVLALiberoActionAdapter, checking the resulting
    action for NaN/Inf

Never modifies vla_server/model_loader.py, policy_semantics/*, or
vla_adapters/*; only calls their existing public functions.

Run:
  .venv-vla/bin/python -m benchmark.verify_checkpoint_loading \\
    --checkpoint outputs/train/smolvla_recycling_train80_v1/checkpoints/000500/pretrained_model
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark.run_vla_action_direction_diagnostic import build_robot_state, resolve
from policy.policy_types import PolicyInput
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import get_manifest
from policy_semantics.native_policy_action import NativePolicyAction
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_checkpoint(checkpoint_path: str) -> dict:
    import os

    os.environ.setdefault("VLA_DEVICE", "cuda")
    os.environ.setdefault("VLA_DTYPE", "float32")
    from vla_server import model_loader

    checkpoint_path = str(resolve(checkpoint_path))
    result = {"checkpoint_path": checkpoint_path}

    # 1. Manifest
    manifest = get_manifest(checkpoint_path)
    result["manifest"] = {
        "model_id": manifest.model_id,
        "action_dimension": manifest.action_dimension,
        "native_translation_scale_m": manifest.native_translation_scale_m,
        "native_rotation_scale_rad": manifest.native_rotation_scale_rad,
        "native_action_clip_range": list(manifest.native_action_clip_range),
        "native_gripper_range": manifest.native_gripper_range,
        "native_gripper_min_means": manifest.native_gripper_min_means,
        "native_gripper_max_means": manifest.native_gripper_max_means,
        "gripper_convention": manifest.gripper_convention,
        "axis_convention_verified": manifest.axis_convention_verified,
        "notes": manifest.notes,
    }

    # 2. CompatibilityGate
    gate_result = CompatibilityGate.check(manifest)
    result["compatibility"] = {
        "passed": gate_result.passed,
        "reasons": gate_result.reasons,
        "checks": gate_result.checks,
    }

    # 3. Local load via model_loader (in-process, no server)
    model_loader._state["status"] = "unloaded"
    model_loader._state["model_family"] = None
    model_loader._state["model"] = None
    load_result = model_loader.load_model_once("smolvla", checkpoint_path, local_files_only=True)
    result["load_result"] = {"status": load_result["status"], "reason": load_result["reason"]}

    if load_result["status"] != "loaded":
        result["sample_inference"] = None
        return result

    # 4. Sample batch inference through a real PyBullet observation + the
    # production adapter, checking for NaN/Inf.
    from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    backend.set_object_position([0.345, 0.0, 0.05])
    robot_state, _state_8d, object_position = build_robot_state(backend)
    main_image = backend.render_main_camera()
    wrist_image = backend.render_wrist_camera()

    policy_input = PolicyInput(
        image=main_image,
        instruction="Pick up the plastic bottle and place it in the plastic bin.",
        robot_state=robot_state,
        task_goal={},
        target_object_position=object_position,
        bin_position=[0.3, 0.35, 0.05],
        step_index=0,
        phase="move_to_object",
        images_by_role={"main": main_image, "wrist": wrist_image},
    )
    model_input = {
        "instruction": policy_input.instruction,
        "image": policy_input.image,
        "images_by_role": policy_input.images_by_role,
        "robot_state": policy_input.robot_state,
        "task_goal": policy_input.task_goal,
        "target_object_position": policy_input.target_object_position,
        "bin_position": policy_input.bin_position,
        "step_index": policy_input.step_index,
        "phase": policy_input.phase,
    }
    raw_output = model_loader.run_inference("smolvla", model_input)
    backend.shutdown()

    # run_inference() for smolvla (with official processors loaded, the
    # normal path) already returns a NativePolicyAction -- see
    # vla_server/model_loader.run_inference()'s own docstring/branch.
    assert isinstance(raw_output, NativePolicyAction), f"Expected NativePolicyAction, got {type(raw_output)}"
    native = raw_output
    raw_values = list(native.values)
    raw_finite = bool(np.all(np.isfinite(np.array(raw_values, dtype=np.float64))))

    command = SmolVLALiberoActionAdapter().decode(native, manifest, context={"degraded_input": False})

    decoded_finite = None
    decoded = None
    if command is not None:
        decoded = {
            "translation_m": list(command.translation_m),
            "rotation_axis_angle_rad": list(command.rotation_axis_angle_rad),
            "gripper_opening_01": command.gripper_opening_01,
        }
        flat = list(command.translation_m) + list(command.rotation_axis_angle_rad) + [command.gripper_opening_01]
        decoded_finite = bool(np.all(np.isfinite(np.array(flat, dtype=np.float64))))

    result["sample_inference"] = {
        "raw_action_values": raw_values,
        "raw_action_finite": raw_finite,
        "decoded_command": decoded,
        "decoded_command_finite": decoded_finite,
        "decode_returned_none": command is None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = verify_checkpoint(args.checkpoint)
    print(json.dumps(result, indent=2, default=str))

    if args.output:
        output_path = resolve(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
