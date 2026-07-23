"""Size-aware Scripted Expert V2.1 (see this task's chat report,
"물체 크기 기반 Scripted Expert V2.1"). Computes grasp waypoints from
object DIMENSIONS (height + footprint_xy or radius) instead of V1's
fixed per-episode offsets from object CENTER -- so the SAME code path
covers cube, rectangular box, and (new) upright cylinder without a
`if shape == ...: hardcode everything` branch per shape.

Does NOT touch orientation at all (unlike
benchmark/so101_expert_v2_orientation.py, preserved untouched and not
imported here -- see this task's own absolute principle 3: that file
and its results must be kept separate, not merged into V2.1). V2.1
always uses V1's own neutral (yaw=0) grasp orientation -- cube/box stay
axis-aligned exactly as V1 already assumes, and an upright cylinder is
yaw-symmetric, so orientation is irrelevant to it by construction (this
task's own absolute principle 10: cylinder success must NOT be read as
solving box arbitrary-yaw -- V2.1 makes no such claim and never invokes
the orientation-aware IK path at all).

V1 (benchmark/so101_scripted_expert.py) is NOT modified -- every
control-flow function used here (`gripper_phase`, `move_to_target`,
`run_bin_place_segment`, `evaluate_bin_place_success`,
`compute_bin_success_debug`) is V1's own, imported and reused AS-IS.
The backend gained one ADDITIVE, opt-in change this task
(`object_shape`/`object_radius` in scene_config + `_create_cylinder()`
-- default "box" reproduces every existing scene byte-for-byte,
verified via test_so101_expert_v1_regression.py after the change).
"""

import math
from dataclasses import dataclass
from typing import Optional

from benchmark.so101_scripted_expert import (
    FAILURE_GRASP_FAILED,
    FAILURE_IK_FAILED,
    FAILURE_LIFT_FAILED,
    FAILURE_OBJECT_DROPPED,
    LIFT_DISTANCE_M,
    MAX_MOVE_STEPS,
    PHASE_APPROACH,
    PHASE_GRASP,
    PHASE_LIFT,
    PHASE_PRE_GRASP,
    PHASE_TRANSPORT,
    So101ExpertError,
    gripper_phase,
    move_to_target,
    run_bin_place_segment,
)
from robot_sim.so101_pybullet_backend import GRASP_DISTANCE_THRESHOLD_M, So101PyBulletBackend

# --- Dimension-based clearance constants (see this task's chat report,
# "dimension 기반 계산식") -- calibrated so that, at object_height=0.04m
# (the EXISTING cube/box height, unchanged this task), the resulting
# pre_grasp_z/grasp_approach_z are numerically equal (within float
# tolerance) to V1's own hardcoded PRE_GRASP_OFFSET_M=[0,0,0.08] and
# APPROACH_OFFSET_M=[0,0,0.03] (both measured from object CENTER,
# height/2=0.02 above which is the object TOP):
#   PRE_GRASP_APPROACH_CLEARANCE_M + 0.02 == 0.08  ->  0.06
#   GRASP_APPROACH_CLEARANCE_M     + 0.02 == 0.03  ->  0.01
# Verified numerically in test_so101_expert_v2_size_aware.py's own
# cube/box compatibility check, not just asserted here.
PRE_GRASP_APPROACH_CLEARANCE_M = 0.06
GRASP_APPROACH_CLEARANCE_M = 0.01

# Diagnostic-only threshold (see this task's chat report, "성공 판정
# 보강") -- used by the evaluation script's own physical_success
# computation, NEVER by this module's control flow (V1's own
# grasp/lift/transport abort conditions are reused unchanged).
MINIMUM_PHYSICAL_LIFT_HEIGHT_M = 0.03


@dataclass
class ObjectMetadata:
    """What SizeAwareGraspPlanner needs to know about THIS episode's
    object -- read from the same scene_config/get_object_pose() values
    the caller already has, never inferred."""

    shape: str  # "box" or "cylinder"
    position: list  # [x, y, z] object CENTER, world frame
    height_m: float  # full height (2x half-extent)
    footprint_xy_half_extents: Optional[list] = None  # box only: [half_x, half_y]
    radius_m: Optional[float] = None  # cylinder only
    mass_kg: float = 0.05
    friction: Optional[float] = None  # None = PyBullet default (unset, matches existing cube/box -- see this task's own "물체 질량/마찰 randomization 금지")

    @property
    def recommended_grasp_axis(self) -> str:
        return "world_x_fixed_closing_axis" if self.shape in ("box", "cube") else "rotationally_symmetric"


@dataclass
class SizeAwareGraspPlan:
    object_top_z: float
    pre_grasp_position: list
    grasp_approach_position: list
    effective_object_width_m: float
    gripper_open_target: float
    gripper_close_target: float
    grasp_distance_threshold_m: float
    lift_success_threshold_m: float
    approach_clearance_m: float


class SizeAwareGraspPlanner:
    """Dimension-based grasp waypoint calculator (see this task's chat
    report, "Expert V2.1 목표"). No `if shape == "cube": ... elif
    shape == "box": ...` branch on VALUES -- the only shape-specific
    code is effective_object_width's source (footprint_xy[0]*2 for
    box, 2*radius for cylinder), exactly the "허용 예" this task's own
    prompt describes."""

    def plan(self, metadata: ObjectMetadata) -> SizeAwareGraspPlan:
        object_top_z = metadata.position[2] + metadata.height_m / 2.0
        pre_grasp_z = object_top_z + PRE_GRASP_APPROACH_CLEARANCE_M
        grasp_approach_z = object_top_z + GRASP_APPROACH_CLEARANCE_M

        if metadata.shape == "cylinder":
            effective_width = 2.0 * metadata.radius_m
        else:
            effective_width = 2.0 * metadata.footprint_xy_half_extents[0]

        return SizeAwareGraspPlan(
            object_top_z=object_top_z,
            pre_grasp_position=[metadata.position[0], metadata.position[1], pre_grasp_z],
            grasp_approach_position=[metadata.position[0], metadata.position[1], grasp_approach_z],
            effective_object_width_m=effective_width,
            gripper_open_target=1.0, gripper_close_target=0.0,
            grasp_distance_threshold_m=GRASP_DISTANCE_THRESHOLD_M,
            lift_success_threshold_m=MINIMUM_PHYSICAL_LIFT_HEIGHT_M,
            approach_clearance_m=PRE_GRASP_APPROACH_CLEARANCE_M,
        )


def run_pick_and_place_episode_v2_1(backend: So101PyBulletBackend, metadata: ObjectMetadata, transport_delta_xy: list, on_step=None) -> dict:
    """Size-aware counterpart of V1's own run_pick_and_place_episode() --
    SAME phase sequence/functions (gripper_phase/move_to_target/
    run_bin_place_segment), only pre_grasp/approach TARGETS are
    computed from `metadata` via SizeAwareGraspPlanner instead of V1's
    fixed PRE_GRASP_OFFSET_M/APPROACH_OFFSET_M. Orientation is never
    touched (always V1's own neutral/yaw=0 IK path)."""
    planner = SizeAwareGraspPlanner()
    plan = planner.plan(metadata)

    gripper_phase(backend, PHASE_PRE_GRASP, plan.gripper_open_target, on_step)
    pre_grasp_result = move_to_target(backend, plan.pre_grasp_position, PHASE_PRE_GRASP, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=on_step)
    approach_result = move_to_target(backend, plan.grasp_approach_position, PHASE_APPROACH, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=on_step)
    gripper_phase(backend, PHASE_GRASP, plan.gripper_close_target, on_step)
    grasp_succeeded = backend.is_grasped()
    if not grasp_succeeded:
        raise So101ExpertError("grasp was not established -- cannot proceed to lift/transport", FAILURE_GRASP_FAILED, phase=PHASE_GRASP)

    grasp_position, _ = backend.get_end_effector_pose()

    ee_pre_lift, _ = backend.get_end_effector_pose()
    lift_target = [ee_pre_lift[0], ee_pre_lift[1], ee_pre_lift[2] + LIFT_DISTANCE_M]
    lift_result = move_to_target(backend, lift_target, PHASE_LIFT, MAX_MOVE_STEPS, FAILURE_LIFT_FAILED, on_step=on_step, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during lift -- cannot proceed to transport", FAILURE_OBJECT_DROPPED, phase=PHASE_LIFT)

    object_lift_height = lift_result["final_ee_position"][2] - metadata.position[2]

    ee_lift_final = lift_result["final_ee_position"]
    transport_target = [ee_lift_final[0] + transport_delta_xy[0], ee_lift_final[1] + transport_delta_xy[1], ee_lift_final[2]]
    transport_result = move_to_target(backend, transport_target, PHASE_TRANSPORT, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=on_step, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during transport -- cannot proceed to release", FAILURE_OBJECT_DROPPED, phase=PHASE_TRANSPORT)

    if not backend.use_bin:
        raise RuntimeError("run_pick_and_place_episode_v2_1 currently only supports backend.use_bin=True (matches this task's own Stage 1A/1B bin-based scenes)")

    bin_place_result = run_bin_place_segment(backend, on_step=on_step)

    return {
        "grasp_plan": {
            "object_top_z": plan.object_top_z, "pre_grasp_position": plan.pre_grasp_position,
            "grasp_approach_position": plan.grasp_approach_position, "effective_object_width_m": plan.effective_object_width_m,
            "gripper_open_target": plan.gripper_open_target, "gripper_close_target": plan.gripper_close_target,
            "grasp_distance_threshold_m": plan.grasp_distance_threshold_m, "lift_success_threshold_m": plan.lift_success_threshold_m,
            "approach_clearance_m": plan.approach_clearance_m, "recommended_grasp_axis": metadata.recommended_grasp_axis,
        },
        "pre_grasp": pre_grasp_result, "approach": approach_result, "grasp_position_ee": grasp_position,
        "lift": lift_result, "object_lift_height": object_lift_height, "transport": transport_result,
        "bin_place_result": bin_place_result,
    }
