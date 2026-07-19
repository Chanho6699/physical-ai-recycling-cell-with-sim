"""Scripted oracle policy behind the OpenVLA-style BasePolicy interface (v0).

DummyOpenVLAPolicy does not run any model -- it drives a fixed phase
state machine and outputs small clamped delta actions toward whichever
position the current phase targets. Kept behind BasePolicy so a real
OpenVLA policy (or a FastAPI dummy-server client) can be dropped in
later without changing the control loop that calls predict_action().

Phases:

  move_to_object -> close_gripper -> lift_object -> move_above_bin
  -> open_gripper -> done

move_to_object -> close_gripper -> move_to_bin -> open_gripper (the
original v0 phase list) turned out to be unreliable: closing the
gripper right at table height and then moving diagonally straight
toward the bin while still that low caused the held object to
graze/collide with the table and bin geometry, stalling the arm well
short of the target (see docs -- reproduced with
run_dummy_openvla_policy_control_demo.py's default run). Lifting to a
fixed carry_height before translating horizontally, then descending
only at the very end (open_gripper releases from above, it doesn't
need to descend further), avoids that collision path entirely.
"""

import math
from typing import Optional

from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput

DEFAULT_MAX_STEP_SIZE = 0.03
DEFAULT_POSITION_TOLERANCE = 0.03
DEFAULT_CARRY_HEIGHT = 0.18
DEFAULT_GRASP_Z_OFFSET = 0.015

# The bin is a solid box (see PyBulletPandaBackend, half-extent 0.03 on
# z), so its top surface sits above the stored bin_position's z. Placing
# requires the object to end up within the backend's PLACE_THRESHOLD
# (0.08) of bin_position when the gripper opens, but descending onto
# the box's exact stored z would drive the end effector into its top
# surface. This clearance targets a point above the box that is close
# enough to place successfully without colliding with it.
PLACE_APPROACH_CLEARANCE = 0.05

# move_above_bin's tight position_tolerance-based transition can, for an
# object grasped from an unusual/far-reach position (e.g. near the edge
# of the Real2Sim workspace), never quite settle within tolerance on
# every axis at once -- small residual IK error keeps flipping the stage
# target between carry_height and the descend height, which reads as
# the arm oscillating/rattling near the bin forever. These two knobs are
# a bounded escape hatch: open anyway once we're "close enough" by a
# looser threshold, or once we've spent too long in this phase at all.
DEFAULT_BIN_OPEN_DISTANCE_THRESHOLD = 0.09
DEFAULT_MAX_MOVE_ABOVE_BIN_STEPS = 40


class DummyOpenVLAPolicy(BasePolicy):
    def __init__(
        self,
        max_step_size: float = DEFAULT_MAX_STEP_SIZE,
        position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
        carry_height: float = DEFAULT_CARRY_HEIGHT,
        grasp_z_offset: float = DEFAULT_GRASP_Z_OFFSET,
        bin_open_distance_threshold: float = DEFAULT_BIN_OPEN_DISTANCE_THRESHOLD,
        max_move_above_bin_steps: int = DEFAULT_MAX_MOVE_ABOVE_BIN_STEPS,
        stabilization_steps: int = 0,
        stabilization_step_size: float = 0.01,
    ):
        """stabilization_steps (default 0 -- IDENTICAL behavior to before
        this parameter existed, every existing caller unaffected):
        inserts a "stabilize" phase between move_to_object and
        close_gripper, active only when > 0 -- see benchmark/
        collect_v3_recovery_smoke.py's chat report (V3 Recovery Data
        Collector task) for why: v2 training data was found to spend
        essentially zero frames dwelling near the object before closing
        (near_frames_in_last_5_before_close_mean ~1.0, see
        results/dataset_analysis/v2_approach_coverage.md), so the smoke
        collector needs a genuine, non-faked few-frame fine-correction
        window here rather than an instant tolerance-crossing -> close
        transition."""
        self.max_step_size = max_step_size
        self.position_tolerance = position_tolerance
        self.carry_height = carry_height
        self.grasp_z_offset = grasp_z_offset
        self.bin_open_distance_threshold = bin_open_distance_threshold
        self.max_move_above_bin_steps = max_move_above_bin_steps
        self.stabilization_steps = stabilization_steps
        self.stabilization_step_size = stabilization_step_size
        self.phase = "move_to_object"
        self.last_info: dict = {}
        self.move_above_bin_steps = 0
        self.stabilization_counter = 0

    def reset(self) -> None:
        self.phase = "move_to_object"
        self.last_info = {}
        self.move_above_bin_steps = 0
        self.stabilization_counter = 0

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        """Thin wrapper around _predict_phase_action(): runs the actual
        (image-blind) phase state machine unchanged, then records
        whether/what image input this step carried. DummyOpenVLAPolicy
        doesn't do any visual reasoning on policy_input.image -- this is
        just bookkeeping so a real VLA/visual policy implementing the
        same predict_action(PolicyInput) -> PolicyOutput interface can be
        dropped in later without the control loop around it changing.
        """
        policy_output = self._predict_phase_action(policy_input)

        image = policy_input.image
        image_input_info = {
            "used_image_input": image is not None,
            "image_shape": list(image.shape) if hasattr(image, "shape") else None,
            "observation_source": policy_input.observation_source,
        }
        if policy_input.visual_observation is not None:
            image_input_info["object_visible"] = policy_input.visual_observation.get("object_visible")
            image_input_info["estimated_world_position"] = policy_input.visual_observation.get(
                "estimated_world_position"
            )

        policy_output.info = {**(policy_output.info or {}), **image_input_info}
        self.last_info = policy_output.info
        return policy_output

    def _predict_phase_action(self, policy_input: PolicyInput) -> PolicyOutput:
        current_ee = policy_input.robot_state["end_effector_position"]
        robot_state = policy_input.robot_state

        # A plain if-chain (not elif) is deliberate: when a phase's
        # transition condition passes, execution falls through into the
        # next phase's block in the *same* call, instead of wasting a
        # whole step returning a zero action while only the phase label
        # changes.
        if self.phase == "move_to_object":
            grasp_target = self._grasp_target(policy_input.target_object_position)
            delta, distance = self._delta_to_target(current_ee, grasp_target)
            self.last_info = {"distance_to_target": distance, "target": grasp_target}
            if distance <= self.position_tolerance:
                if self.stabilization_steps > 0:
                    self.phase = "stabilize"
                    self.stabilization_counter = 0
                else:
                    self.phase = "close_gripper"
            else:
                return PolicyOutput(
                    action=[delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 0.0],
                    phase="move_to_object",
                    done=False,
                    info=self.last_info,
                )

        if self.phase == "stabilize":
            # A few genuine, smaller-magnitude corrective steps toward the
            # grasp target (real delta each frame, NOT a copied zero-action
            # -- see this method's own module-level "recovery" requirement)
            # before transitioning to close_gripper. Distance naturally
            # shrinks step over step as the residual error is corrected, so
            # consecutive frames are NOT identical even though they all aim
            # at the same fixed target.
            grasp_target = self._grasp_target(policy_input.target_object_position)
            raw_delta = [grasp_target[axis] - current_ee[axis] for axis in range(3)]
            distance = math.sqrt(sum(component ** 2 for component in raw_delta))
            clamped_delta = [
                max(-self.stabilization_step_size, min(self.stabilization_step_size, component))
                for component in raw_delta
            ]
            self.stabilization_counter += 1
            self.last_info = {
                "distance_to_target": distance, "target": grasp_target,
                "stabilization_step": self.stabilization_counter,
            }
            if self.stabilization_counter >= self.stabilization_steps:
                self.phase = "close_gripper"
            else:
                return PolicyOutput(
                    action=[clamped_delta[0], clamped_delta[1], clamped_delta[2], 0.0, 0.0, 0.0, 0.0],
                    phase="stabilize",
                    done=False,
                    info=self.last_info,
                )

        if self.phase == "close_gripper":
            held = bool(robot_state.get("held_object", False))
            task_status = robot_state.get("task_status")
            self.last_info = {"held_object": held, "task_status": task_status}
            if held or task_status == "grasped":
                self.phase = "lift_object"
            else:
                # Keep (re-)issuing the close command until the backend
                # confirms the grasp -- not necessarily a one-shot.
                return PolicyOutput(
                    action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    phase="close_gripper",
                    done=False,
                    info=self.last_info,
                )

        if self.phase == "lift_object":
            lift_target = [
                policy_input.target_object_position[0],
                policy_input.target_object_position[1],
                self.carry_height,
            ]
            delta, distance = self._delta_to_target(current_ee, lift_target)
            self.last_info = {"distance_to_target": distance, "target": lift_target}
            if distance <= self.position_tolerance:
                self.phase = "move_above_bin"
                self.move_above_bin_steps = 0
            else:
                return PolicyOutput(
                    action=[delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 1.0],
                    phase="lift_object",
                    done=False,
                    info=self.last_info,
                )

        if self.phase == "move_above_bin":
            self.move_above_bin_steps += 1
            bin_position = policy_input.bin_position or [0.0, 0.0, self.carry_height]
            xy_distance = math.sqrt(
                (bin_position[0] - current_ee[0]) ** 2 + (bin_position[1] - current_ee[1]) ** 2
            )
            in_descent_stage = xy_distance <= self.position_tolerance
            if not in_descent_stage:
                # Stage 1: travel laterally at a fixed carry height so the
                # held object never drags diagonally close to the table
                # or bin.
                stage_target = [bin_position[0], bin_position[1], self.carry_height]
            else:
                # Stage 2: xy is already aligned above the bin, so the
                # remaining motion is a straight vertical descent (no
                # diagonal risk) down to just above the bin's lid.
                stage_target = [
                    bin_position[0],
                    bin_position[1],
                    bin_position[2] + PLACE_APPROACH_CLEARANCE,
                ]

            delta, distance = self._delta_to_target(current_ee, stage_target)
            held = bool(robot_state.get("held_object", False))

            # Normal path: tight tolerance on the current stage target.
            reached_stage_target = distance <= self.position_tolerance
            # Escape hatch 1: xy (horizontal distance to the bin) and z
            # (vertical distance to the drop height) are checked
            # separately rather than as one combined 3D distance, so a
            # slightly-too-wide xy alone (still "near the bin") doesn't
            # get stuck failing the combined check forever. The vertical
            # leg is deliberately kept at position_tolerance, not looser:
            # the backend's PLACE_THRESHOLD (0.08) only leaves ~0.03 of
            # margin above PLACE_APPROACH_CLEARANCE (0.05) before a
            # release actually misses the bin, and 0.03 is exactly
            # position_tolerance -- loosening this leg would open the
            # gripper too high and turn a "success" into a "released".
            vertical_distance_to_drop = abs(current_ee[2] - (bin_position[2] + PLACE_APPROACH_CLEARANCE))
            near_bin_force_open = (
                held
                and in_descent_stage
                and xy_distance <= self.bin_open_distance_threshold
                and vertical_distance_to_drop <= self.position_tolerance
            )
            # Escape hatch 2: unconditional time-box so this phase can
            # never oscillate forever, even if still far from the bin.
            exceeded_max_steps = held and self.move_above_bin_steps >= self.max_move_above_bin_steps

            if reached_stage_target or near_bin_force_open or exceeded_max_steps:
                if reached_stage_target:
                    reason = None
                elif near_bin_force_open:
                    reason = "near_bin_force_open"
                else:
                    reason = "max_move_above_bin_steps_exceeded"

                self.last_info = {"distance_to_target": distance, "target": stage_target}
                if reason is not None:
                    self.last_info["phase_transition_reason"] = reason
                self.phase = "open_gripper"
            else:
                self.last_info = {"distance_to_target": distance, "target": stage_target}
                return PolicyOutput(
                    action=[delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 1.0],
                    phase="move_above_bin",
                    done=False,
                    info=self.last_info,
                )

        if self.phase == "open_gripper":
            task_status = robot_state.get("task_status")
            held = bool(robot_state.get("held_object", True))
            self.last_info = {"held_object": held, "task_status": task_status}
            if task_status == "success" or not held:
                self.phase = "done"
            else:
                return PolicyOutput(
                    action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    phase="open_gripper",
                    done=False,
                    info=self.last_info,
                )

        # self.phase == "done"
        return PolicyOutput(
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            phase="done",
            done=True,
            info=self.last_info,
        )

    def _grasp_target(self, target_object_position: list) -> list:
        return [
            target_object_position[0],
            target_object_position[1],
            target_object_position[2] + self.grasp_z_offset,
        ]

    def _delta_to_target(self, current_position: list, target_position: Optional[list]):
        if target_position is None:
            return [0.0, 0.0, 0.0], 0.0

        raw_delta = [target_position[axis] - current_position[axis] for axis in range(3)]
        distance = math.sqrt(sum(component ** 2 for component in raw_delta))
        clamped_delta = [
            max(-self.max_step_size, min(self.max_step_size, component)) for component in raw_delta
        ]
        return clamped_delta, distance
