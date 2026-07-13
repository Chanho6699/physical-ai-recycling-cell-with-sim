"""Wrist-camera-aware VLA-ready control loop probe (v0).

Confirms that DummyOpenVLAPolicy actually receives a wrist-camera RGB
frame in PolicyInput.image on every step (not just some steps), which is
the whole point of --policy-observation-source wrist in
run_full_recycling_cell_demo.py. No YOLO, no ArUco, no external camera --
object_position is given directly, and only a handful of policy steps
run (default 10), just enough to confirm the observation loop shape.
"""

import argparse
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from robot_sim.pybullet_wrist_camera import PyBulletWristCamera

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRIST_CAMERA_CONFIG = "configs/wrist_camera_config.json"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-position", type=float, nargs=3, default=[0.40, -0.10, 0.05])
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--wrist-camera-config", type=str, default=DEFAULT_WRIST_CAMERA_CONFIG)

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    object_position = list(args.object_position)

    print("=== Wrist Camera Policy Input Loop Probe ===")

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        try:
            backend.reset()
        except Exception as exc:
            print(f"Panda backend reset failed: {exc}")
            print("FAIL")
            return

        backend.set_object_type(args.object_type)
        state = backend.set_object_position(object_position)
        bin_position = state["bin_position"]

        wrist_camera = PyBulletWristCamera(
            client_id=backend.client_id,
            robot_id=backend.robot_id,
            config_path=resolve(args.wrist_camera_config),
        )
        action_adapter = ActionAdapter()
        policy = DummyOpenVLAPolicy()
        policy.reset()

        task_goal_dict = {
            "instruction": DEFAULT_INSTRUCTION,
            "action": "pick_and_place",
            "target_object": args.object_type,
            "target_bin": "plastic_bin",
        }

        all_ok = True
        used_wrist_observation_steps = 0

        for step_index in range(args.steps):
            robot_state = backend.get_state()

            frame, _render_debug = wrist_camera.render()
            _, estimate_debug = wrist_camera.estimate_object_position_from_segmentation(
                frame, backend._object_id
            )
            policy_image = frame["rgb"]
            visual_observation = {
                "object_visible": estimate_debug["object_visible"],
                "object_pixel_count": estimate_debug["object_pixel_count"],
                "estimated_world_position": estimate_debug["estimated_world_position"],
            }

            policy_input = PolicyInput(
                image=policy_image,
                instruction=DEFAULT_INSTRUCTION,
                robot_state=robot_state,
                task_goal=task_goal_dict,
                target_object_position=object_position,
                bin_position=bin_position,
                step_index=step_index,
                phase=policy.phase,
                observation_source="wrist",
                visual_observation=visual_observation,
            )
            policy_output = policy.predict_action(policy_input)
            robot_command = action_adapter.convert(policy_output.action)
            backend.apply_command(robot_command, steps=args.steps_per_action)

            has_image = policy_image is not None
            image_shape = list(policy_image.shape) if hasattr(policy_image, "shape") else None
            used_image_input = bool((policy.last_info or {}).get("used_image_input"))
            object_visible = estimate_debug["object_visible"]

            print(f"step {step_index}: has_image={has_image} image_shape={image_shape} object_visible={object_visible}")

            if not (has_image and used_image_input and image_shape is not None):
                all_ok = False

            used_wrist_observation_steps += 1

        print(f"used_wrist_observation_steps: {used_wrist_observation_steps}")
        print("PASS" if all_ok else "FAIL")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
