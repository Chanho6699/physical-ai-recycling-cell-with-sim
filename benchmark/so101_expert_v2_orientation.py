"""Orientation-aware Scripted Expert V2 (see this task's chat report,
"Orientation-aware Expert V2"). Handles rectangular-box grasps at a
non-zero object yaw -- V1 (benchmark/so101_scripted_expert.py) is
NEVER imported for its internal helpers in a way that changes its
behavior, and this module does NOT modify robot_sim/so101_pybullet_backend.py
at all (confirmed unnecessary this task -- every attribute this module
needs -- `robot_id`, `ee_link_index`, `arm_joint_indices`,
`joint_info_by_name`, `min_ee_height_m`, `client_id`,
`get_end_effector_pose()`, `apply_joint_target()`, `get_object_pose()`
-- is ALREADY public on So101PyBulletBackend). V2 calls
`p.calculateInverseKinematics()` itself, passing `targetOrientation`
(a parameter V1's own `compute_joint_target_from_ee_delta()` never
passes -- confirmed via direct source inspection this task), rather
than adding a hook to the shared backend file.

Pipeline (see this task's chat report, "기능 분리"):
  ObjectGraspMetadata          -- what this grasp needs to know (position/yaw/dims)
  OrientationAwareGraspPlanner -- turns that into target poses/orientations
  (move_to_target_with_orientation is the "ScriptedMotionPlanner" --
   the per-step IK+step loop, structurally identical to V1's own
   move_to_target() but orientation-aware)
  ExpertExecutionMonitor        -- per-step joint-jump/limit/orientation-error/collision tracking
  ExpertEvaluationRecorder       -- (see benchmark/evaluate_so101_expert_v2_yaw_grid.py) aggregates episode results

Orientation policy (this task's own "권장 정책"):
  pre_grasp -> approach -> grasp -> lift: object-yaw-aligned orientation held
  transition_to_neutral (NEW phase, inserted here, not a V1 phase):
    wrist rotates back toward the SAME yaw=0 orientation V1 always used,
    while still at lift height (no lateral motion during the rotation)
  transport onward: neutral orientation -- at that point this module
    delegates to V1's OWN run_bin_place_segment()/move_to_target()
    UNCHANGED (those functions are position-only and V1-calibrated;
    since orientation is already back to V1's own neutral by the time
    they run, no orientation-aware variant is needed for them at all).

180-degree grasp symmetry (this task's own "Yaw 대칭 처리"): a
rectangular box's short axis (local X) is unchanged by a 180 degree
yaw flip, so candidate_yaw = object_yaw and candidate_yaw = object_yaw
+ pi are equally valid grasp orientations. This module canonicalizes
to whichever candidate is a SMALLER absolute rotation away from the
arm's own neutral pose (wrist_roll = 0 at the start of every episode --
see robot_sim/so101_pybullet_backend.py's own NEUTRAL_ARM_POSITIONS),
so the wrist never takes an unnecessarily large turn.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pybullet as p

from benchmark.so101_scripted_expert import (
    APPROACH_OFFSET_M,
    CONVERGENCE_TOLERANCE_M,
    FAILURE_GRASP_FAILED,
    FAILURE_IK_FAILED,
    FAILURE_LIFT_FAILED,
    FAILURE_OBJECT_DROPPED,
    LIFT_DISTANCE_M,
    MAX_MOVE_STEPS,
    MAX_STEP_M,
    PHASE_APPROACH,
    PHASE_GRASP,
    PHASE_LIFT,
    PHASE_PRE_GRASP,
    PHASE_TRANSPORT,
    PRE_GRASP_OFFSET_M,
    STEP_ERROR_FAILURE_THRESHOLD_M,
    So101ExpertError,
    all_finite,
    check_joint_limits,
    gripper_phase,
    move_to_target,
    object_offset_in_ee_frame,
    run_bin_place_segment,
)
from robot_sim.so101_pybullet_backend import (
    ARM_JOINT_NAMES,
    IK_RESIDUAL_THRESHOLD,
    IK_SOLVER_ITERATIONS,
    So101PyBulletBackend,
)

# --- New V2-only phase label (never a V1 PHASE_* constant, never fed
# into any dataset schema/phase_id mapping -- purely a bookkeeping
# label for this module's own on_step calls during the reorientation). ---
PHASE_TRANSITION_TO_NEUTRAL = "transition_to_neutral"

# Max orientation-only reorientation steps for the neutral transition
# (no lateral motion during this phase -- see this module's own docstring).
MAX_REORIENT_STEPS = 30
ORIENTATION_CONVERGENCE_TOLERANCE_RAD = 0.05

FAILURE_ORIENTATION_UNREACHABLE = "orientation_unreachable"
FAILURE_EXCESSIVE_JOINT_JUMP = "excessive_joint_jump"

# Threshold used ONLY to classify a step-error abort's ROOT CAUSE (see
# move_to_target_with_orientation() below), never to change whether it
# aborts. Empirically established this task (see chat report, single-
# yaw sanity check + Jacobian analysis + bias-correction convergence
# test): shoulder_pan is the ARM's ONLY joint with a world-Z rotation
# component (angular Jacobian column ~[0,0,-1] at neutral pose), while
# wrist_roll's own axis is ~world-X (~[-1,0,0]) -- so an object-yaw
# rotation about world Z can ONLY be produced by shoulder_pan, which is
# also the joint that determines the arm's horizontal reach bearing.
# Requesting a target orientation whose yaw differs from what
# shoulder_pan-aimed-at-target-position already implies therefore pits
# orientation against position in the SAME joint's authority: IK trades
# them off rather than solving both. When a step-error abort occurs
# with orientation NEARLY satisfied (below this threshold), that is
# this specific structural coupling, not a generic IK divergence --
# classified as orientation_unreachable, never silently folded into
# the generic ik_failed/lift_failed bucket.
ORIENTATION_UNREACHABLE_ANGLE_THRESHOLD_RAD = math.radians(15.0)

# A per-step joint delta (radians) above this is flagged as an
# "excessive jump" for ExpertExecutionMonitor's own reporting -- chosen
# as roughly 3x MAX_STEP_M's own implied per-step arm displacement
# under normal position-only convergence (empirically, V1's own
# position-only steps move a small fraction of a radian per joint per
# step); this is a MONITORING threshold, not a hard abort condition
# (V1 itself has no such check either), so this module never raises
# because of it -- it only records `max_joint_jump`/
# `joint_limit_violation` for the caller/recorder to judge.
JOINT_JUMP_WARNING_THRESHOLD_RAD = 0.5


@dataclass
class ObjectGraspMetadata:
    """Everything Orientation-aware grasp planning needs to know about
    THIS episode's object -- nothing here is inferred from the live
    PyBullet scene beyond what the caller already read via
    backend.get_object_pose()."""

    object_position: list  # [x, y, z], world frame
    object_yaw_rad: float  # world-frame yaw the object was spawned at (0 = same as V1's own default)
    object_half_extent_x_m: float  # box's local-X half-extent (the SHORT/grasp axis in this project's box convention)
    object_half_extent_y_m: float  # box's local-Y half-extent (the LONG axis)
    gripper_closing_axis: str = "world_x"  # confirmed empirically (Stage 1B investigation) -- the gripper always closes along world X at V1's own neutral orientation
    preferred_grasp_axis: str = "local_x"  # this project's box convention: local X is always the short/graspable axis


@dataclass
class GraspPlan:
    """OrientationAwareGraspPlanner's own output -- see this task's own
    "다음 출력은 명확히 분리되어야 한다" requirement; every field listed
    there is present here, nothing bundled into an opaque blob."""

    grasp_position: list
    pre_grasp_position: list
    approach_position: list
    target_gripper_yaw: float
    selected_orientation_candidate: str  # "object_yaw" or "object_yaw_plus_pi" -- which 180-symmetric candidate was chosen
    target_end_effector_quaternion: list
    neutral_end_effector_quaternion: list
    effective_grasp_width_m: float
    approach_height_m: float


def _quat_angle_diff(q1: list, q2: list) -> float:
    """Smallest rotation angle (radians) between two quaternions
    (xyzw) -- used for ExpertExecutionMonitor's own orientation_error
    reporting and for the neutral-transition phase's own convergence
    check. Sign/hemisphere-safe (a quaternion and its negation
    represent the same rotation)."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def compose_world_z_rotation(yaw_rad: float, base_orientation: list) -> list:
    """R_target = Rz(yaw) (composed in WORLD frame) applied ON TOP OF
    base_orientation -- i.e. p.multiplyTransforms([0,0,0], Rz(yaw),
    [0,0,0], base_orientation)[1], the EXACT SAME composition order
    this codebase's own object_offset_in_ee_frame() (so101_scripted_expert.py)
    already uses for its own transform composition (never guessed --
    see this task's chat report, "추측으로 quaternion 곱셈 순서를
    결정하지 말고"). Derivation: the object's own yaw is applied via
    p.getQuaternionFromEuler([0,0,yaw]) (robot_sim/so101_pybullet_backend.py's
    own reset()), which sends the box's local-X axis to world
    (cos(yaw), sin(yaw), 0). For the gripper's closing axis (world X at
    yaw=0, confirmed empirically) to track that SAME direction, the
    identical additional WORLD-frame Z-rotation must be composed on top
    of the existing (yaw=0) EE orientation -- not composed in the EE's
    OWN local frame (which is itself already ~90 degrees off from world,
    a right-multiplication would rotate about the wrong axis entirely)."""
    rz = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])
    _, target_orientation = p.multiplyTransforms([0.0, 0.0, 0.0], rz, [0.0, 0.0, 0.0], base_orientation)
    return list(target_orientation)


def canonicalize_grasp_yaw(object_yaw_rad: float) -> tuple:
    """180-degree box-grasp symmetry (this task's own "Yaw 대칭 처리")
    -- returns (chosen_yaw, candidate_label). Candidates are object_yaw
    and object_yaw + pi, each wrapped to (-pi, pi]; the one with the
    SMALLER absolute value is chosen, since every episode's arm starts
    at wrist_roll=0 (NEUTRAL_ARM_POSITIONS) -- smaller absolute yaw
    means a smaller wrist rotation away from that known-good start."""
    def wrap(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    candidate_a = wrap(object_yaw_rad)
    candidate_b = wrap(object_yaw_rad + math.pi)
    if abs(candidate_a) <= abs(candidate_b):
        return candidate_a, "object_yaw"
    return candidate_b, "object_yaw_plus_pi"


class OrientationAwareGraspPlanner:
    """Turns ObjectGraspMetadata + the arm's OWN measured yaw=0
    orientation (read live from the backend right after reset(), never
    hardcoded -- see this module's own docstring) into a GraspPlan.
    Positions reuse V1's own PRE_GRASP_OFFSET_M/APPROACH_OFFSET_M
    constants UNCHANGED (this task's own "기존 V1의 grasp height,
    pre-grasp height... 불필요하게 바꾸지 말 것")."""

    def __init__(self, neutral_end_effector_quaternion: list):
        self.neutral_end_effector_quaternion = list(neutral_end_effector_quaternion)

    def plan(self, metadata: ObjectGraspMetadata) -> GraspPlan:
        chosen_yaw, candidate_label = canonicalize_grasp_yaw(metadata.object_yaw_rad)
        target_orientation = compose_world_z_rotation(chosen_yaw, self.neutral_end_effector_quaternion)

        object_position = metadata.object_position
        pre_grasp_position = [object_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)]
        approach_position = [object_position[i] + APPROACH_OFFSET_M[i] for i in range(3)]

        return GraspPlan(
            grasp_position=list(object_position),
            pre_grasp_position=pre_grasp_position,
            approach_position=approach_position,
            target_gripper_yaw=chosen_yaw,
            selected_orientation_candidate=candidate_label,
            target_end_effector_quaternion=target_orientation,
            neutral_end_effector_quaternion=self.neutral_end_effector_quaternion,
            effective_grasp_width_m=2.0 * metadata.object_half_extent_x_m,
            approach_height_m=APPROACH_OFFSET_M[2],
        )


class ExpertExecutionMonitor:
    """Per-episode accumulator for the diagnostic fields this task's
    chat report requires (section 5's verification list + section 9's
    failure-record schema) -- read-only observation, never influences
    control decisions (mirrors this project's own established
    diagnostic-logging convention, e.g. so101_smolvla_rollout.py's
    diagnostic_log)."""

    def __init__(self):
        self.max_joint_jump = 0.0
        self.joint_limit_violation = False
        self.orientation_error_max = 0.0
        self.collision_detected = False
        self._last_joint_positions = None

    def record_step(self, backend: So101PyBulletBackend, joint_targets: list, target_orientation: Optional[list] = None):
        if self._last_joint_positions is not None:
            jump = max(abs(a - b) for a, b in zip(joint_targets, self._last_joint_positions))
            self.max_joint_jump = max(self.max_joint_jump, jump)
        self._last_joint_positions = list(joint_targets)

        violations = check_joint_limits(backend, joint_targets)
        if violations:
            self.joint_limit_violation = True

        if target_orientation is not None:
            _current_ee_position, current_ee_orientation = backend.get_end_effector_pose()
            error = _quat_angle_diff(current_ee_orientation, target_orientation)
            self.orientation_error_max = max(self.orientation_error_max, error)

        contacts = p.getContactPoints(bodyA=backend.robot_id, physicsClientId=backend.client_id)
        # Filter out expected robot self/table contacts is NOT attempted here
        # (this project's own bin-contact diagnostics, e.g.
        # so101_scripted_expert.py's _check_bin_contacts(), already treat ANY
        # nonzero contact as worth recording rather than pre-filtering) --
        # ExpertExecutionMonitor reports whatever PyBullet itself reports,
        # same policy.
        if len(contacts) > 0:
            self.collision_detected = True


def compute_joint_target_with_orientation(backend: So101PyBulletBackend, target_position: list, target_orientation: list) -> dict:
    """Orientation-aware counterpart of
    So101PyBulletBackend.compute_joint_target_from_ee_delta() -- NEVER
    added to that class (see this module's own docstring for why no
    backend change was needed at all). Mirrors that method's own
    internal logic EXACTLY (same min_ee_height_m floor, same
    IK_SOLVER_ITERATIONS/IK_RESIDUAL_THRESHOLD, same per-joint clip)
    except this ALSO passes `targetOrientation` to
    p.calculateInverseKinematics() -- the one call V1 itself never
    makes (confirmed via direct source inspection this task)."""
    target_position = list(target_position)
    if not all(math.isfinite(v) for v in target_position):
        raise ValueError(f"Non-finite EE target position: {target_position}")
    target_position[2] = max(target_position[2], backend.min_ee_height_m)

    joint_poses = p.calculateInverseKinematics(
        backend.robot_id, backend.ee_link_index, target_position, targetOrientation=target_orientation,
        maxNumIterations=IK_SOLVER_ITERATIONS, residualThreshold=IK_RESIDUAL_THRESHOLD,
        physicsClientId=backend.client_id,
    )
    raw_arm_targets = list(joint_poses[: len(backend.arm_joint_indices)])

    clipped_arm_targets = []
    for name, raw_position in zip(ARM_JOINT_NAMES, raw_arm_targets):
        info = backend.joint_info_by_name[name]
        clipped_arm_targets.append(max(info["lower"], min(info["upper"], raw_position)))

    return {"target_position": target_position, "target_orientation": target_orientation, "arm_joint_targets": clipped_arm_targets}


def move_to_target_with_orientation(
    backend: So101PyBulletBackend, target_position: list, target_orientation: list, phase: str, max_steps: int,
    failure_reason: str, on_step=None, track_grasp: bool = False, monitor: Optional[ExpertExecutionMonitor] = None,
) -> dict:
    """Orientation-aware counterpart of V1's own move_to_target() --
    SAME step loop structure (MAX_STEP_M clamp, CONVERGENCE_TOLERANCE_M
    convergence check, STEP_ERROR_FAILURE_THRESHOLD_M/joint-limit
    failure checks, track_grasp bookkeeping -- all reused as VALUES/
    logic, not reimplemented from scratch) -- the ONLY functional
    difference is calling compute_joint_target_with_orientation()
    instead of backend.compute_joint_target_from_ee_delta(). Kept as
    its OWN function (not a modification of move_to_target() itself)
    per this task's own "V1 코드의 기본 동작을 바꿔 V2를 구현하지 말
    것"."""
    ee_start = object_start = grasp_constraint_id_start = initial_relative_offset = None
    max_relative_drift = 0.0
    grasp_maintained_all_steps = True
    constraint_valid_all_steps = True

    if track_grasp:
        ee_start, ee_orientation_start = backend.get_end_effector_pose()
        object_start = backend.get_object_position()
        grasp_constraint_id_start = backend.get_grasp_state()["grasp_constraint_id"]
        initial_relative_offset = object_offset_in_ee_frame(ee_start, ee_orientation_start, object_start)

    step_index = 0
    while step_index < max_steps:
        current_ee_position, _current_ee_orientation = backend.get_end_effector_pose()
        remaining = [target_position[i] - current_ee_position[i] for i in range(3)]
        remaining_norm = math.sqrt(sum(c ** 2 for c in remaining))
        if remaining_norm <= CONVERGENCE_TOLERANCE_M:
            break
        clamped_delta = [max(-MAX_STEP_M, min(MAX_STEP_M, c)) for c in remaining]
        step_target_position = [current_ee_position[i] + clamped_delta[i] for i in range(3)]

        computed = compute_joint_target_with_orientation(backend, step_target_position, target_orientation)
        if on_step is not None:
            gripper_current_normalized = backend.get_observation()["gripper_position_normalized"]
            on_step(phase, computed["arm_joint_targets"], gripper_current_normalized)
        if monitor is not None:
            monitor.record_step(backend, computed["arm_joint_targets"], target_orientation)

        obs = backend.apply_joint_target(computed["arm_joint_targets"])
        final_ee_position, final_ee_orientation = backend.get_end_effector_pose()
        position_error = math.sqrt(sum((final_ee_position[i] - computed["target_position"][i]) ** 2 for i in range(3)))
        obs["ee_delta_target_position"] = computed["target_position"]
        obs["ee_delta_position_error"] = position_error

        if not all_finite(obs["end_effector_position"]):
            raise So101ExpertError(f"[{phase}] non-finite EE position at step {step_index}", failure_reason, phase=phase)
        if obs["ee_delta_position_error"] > STEP_ERROR_FAILURE_THRESHOLD_M:
            step_orientation_error = _quat_angle_diff(final_ee_orientation, target_orientation)
            if step_orientation_error < ORIENTATION_UNREACHABLE_ANGLE_THRESHOLD_RAD:
                # Orientation was (nearly) satisfied but position was not --
                # this task's own empirically-confirmed signature of the
                # shoulder_pan/world-Z coupling (see this module's own
                # ORIENTATION_UNREACHABLE_ANGLE_THRESHOLD_RAD docstring),
                # not a generic IK divergence.
                raise So101ExpertError(
                    f"[{phase}] orientation satisfied ({math.degrees(step_orientation_error):.1f}deg error) but position unreachable "
                    f"({obs['ee_delta_position_error']:.4f}m error) at step {step_index} -- shoulder_pan/world-Z coupling",
                    FAILURE_ORIENTATION_UNREACHABLE, phase=phase,
                )
            raise So101ExpertError(f"[{phase}] abnormal IK step error {obs['ee_delta_position_error']:.4f}m at step {step_index}", failure_reason, phase=phase)
        violations = check_joint_limits(backend, obs["joint_positions"])
        if violations:
            raise So101ExpertError(f"[{phase}] joint limit violation at step {step_index}: {violations}", failure_reason, phase=phase)

        if track_grasp:
            object_now = backend.get_object_position()
            if not all_finite(object_now):
                raise So101ExpertError(f"[{phase}] non-finite object position at step {step_index}", failure_reason, phase=phase)
            if not backend.is_grasped():
                grasp_maintained_all_steps = False
            if backend.get_grasp_state()["grasp_constraint_id"] != grasp_constraint_id_start:
                constraint_valid_all_steps = False
            ee_now = obs["end_effector_position"]
            _ee_now_unused, ee_orientation_now = backend.get_end_effector_pose()
            current_relative_offset = object_offset_in_ee_frame(ee_now, ee_orientation_now, object_now)
            drift = math.sqrt(sum((current_relative_offset[i] - initial_relative_offset[i]) ** 2 for i in range(3)))
            max_relative_drift = max(max_relative_drift, drift)

        step_index += 1

    final_ee_position, final_ee_orientation = backend.get_end_effector_pose()
    final_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
    orientation_error = _quat_angle_diff(final_ee_orientation, target_orientation)
    result = {
        "target": target_position, "final_ee_position": final_ee_position, "error": final_error,
        "num_steps": step_index, "orientation_error_rad": orientation_error,
    }
    if track_grasp:
        result.update({
            "ee_start_position": ee_start, "object_start_position": object_start,
            "object_final_position": backend.get_object_position(),
            "max_relative_drift_m": max_relative_drift,
            "grasp_maintained_all_steps": grasp_maintained_all_steps,
            "constraint_valid_all_steps": constraint_valid_all_steps,
        })
    return result


def rotate_to_neutral_orientation(
    backend: So101PyBulletBackend, neutral_orientation: list, on_step=None, monitor: Optional[ExpertExecutionMonitor] = None,
) -> dict:
    """The "별도 transition phase" this task's chat report explicitly
    asks for (section 5, "갑작스러운 회전이 아니라 별도 transition
    phase를 추가하라") -- holds the CURRENT end-effector position fixed
    (no lateral motion) while rotating orientation back toward
    neutral_orientation over up to MAX_REORIENT_STEPS steps, so the
    reorientation itself is a gradual, monitored motion rather than an
    instantaneous jump baked into the first transport step."""
    current_ee_position, current_ee_orientation = backend.get_end_effector_pose()
    step_index = 0
    while step_index < MAX_REORIENT_STEPS:
        _pos, current_ee_orientation = backend.get_end_effector_pose()
        error = _quat_angle_diff(current_ee_orientation, neutral_orientation)
        if error <= ORIENTATION_CONVERGENCE_TOLERANCE_RAD:
            break
        computed = compute_joint_target_with_orientation(backend, current_ee_position, neutral_orientation)
        if on_step is not None:
            gripper_current_normalized = backend.get_observation()["gripper_position_normalized"]
            on_step(PHASE_TRANSITION_TO_NEUTRAL, computed["arm_joint_targets"], gripper_current_normalized)
        if monitor is not None:
            monitor.record_step(backend, computed["arm_joint_targets"], neutral_orientation)
        backend.apply_joint_target(computed["arm_joint_targets"])
        step_index += 1

    final_ee_position, final_ee_orientation = backend.get_end_effector_pose()
    return {
        "num_steps": step_index,
        "final_orientation_error_rad": _quat_angle_diff(final_ee_orientation, neutral_orientation),
        "final_ee_position": final_ee_position,
    }


def run_pick_and_place_episode_v2(
    backend: So101PyBulletBackend, object_yaw_rad: float, object_half_extent_x_m: float, object_half_extent_y_m: float,
    transport_delta_xy: list, on_step=None, monitor: Optional[ExpertExecutionMonitor] = None,
) -> dict:
    """Orientation-aware V2 episode entry point -- see this module's own
    docstring for the full phase-orientation policy. Delegates the
    ENTIRE post-reorientation portion (transport onward) to V1's own
    move_to_target()/run_bin_place_segment()/gripper_phase() UNCHANGED
    -- those functions are position-only and V1-calibrated, and by the
    time they run here, the wrist is already back at V1's own neutral
    orientation (see rotate_to_neutral_orientation() above), so no
    orientation-aware variant is needed for them at all."""
    if monitor is None:
        monitor = ExpertExecutionMonitor()

    object_position, _ = backend.get_object_pose()
    neutral_ee_position, neutral_ee_orientation = backend.get_end_effector_pose()

    metadata = ObjectGraspMetadata(
        object_position=list(object_position), object_yaw_rad=object_yaw_rad,
        object_half_extent_x_m=object_half_extent_x_m, object_half_extent_y_m=object_half_extent_y_m,
    )
    planner = OrientationAwareGraspPlanner(neutral_ee_orientation)
    plan = planner.plan(metadata)

    def wrapped_on_step(phase, arm_joint_targets, gripper_target_normalized):
        if on_step is not None:
            on_step(phase, arm_joint_targets, gripper_target_normalized)

    # --- pre_grasp / approach / grasp / lift: orientation-aware, held at plan's target orientation ---
    gripper_phase(backend, PHASE_PRE_GRASP, 1.0, wrapped_on_step)
    pre_grasp_result = move_to_target_with_orientation(
        backend, plan.pre_grasp_position, plan.target_end_effector_quaternion, PHASE_PRE_GRASP,
        MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=wrapped_on_step, monitor=monitor,
    )
    approach_result = move_to_target_with_orientation(
        backend, plan.approach_position, plan.target_end_effector_quaternion, PHASE_APPROACH,
        MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=wrapped_on_step, monitor=monitor,
    )
    gripper_phase(backend, PHASE_GRASP, 0.0, wrapped_on_step)
    grasp_succeeded = backend.is_grasped()
    if not grasp_succeeded:
        raise So101ExpertError("grasp was not established -- cannot proceed to lift/transport", FAILURE_GRASP_FAILED, phase=PHASE_GRASP)

    grasp_position, _ = backend.get_end_effector_pose()

    ee_pre_lift, _ = backend.get_end_effector_pose()
    lift_target = [ee_pre_lift[0], ee_pre_lift[1], ee_pre_lift[2] + LIFT_DISTANCE_M]
    lift_result = move_to_target_with_orientation(
        backend, lift_target, plan.target_end_effector_quaternion, PHASE_LIFT,
        MAX_MOVE_STEPS, FAILURE_LIFT_FAILED, on_step=wrapped_on_step, track_grasp=True, monitor=monitor,
    )
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during lift -- cannot proceed to transport", FAILURE_OBJECT_DROPPED, phase=PHASE_LIFT)

    # --- transition_to_neutral: rotate wrist back to V1's own neutral orientation, no lateral motion ---
    transition_result = rotate_to_neutral_orientation(backend, plan.neutral_end_effector_quaternion, on_step=wrapped_on_step, monitor=monitor)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during neutral-orientation transition", FAILURE_OBJECT_DROPPED, phase=PHASE_TRANSITION_TO_NEUTRAL)

    # --- transport onward: V1's OWN functions, UNCHANGED, position-only, now safe since orientation == V1's neutral ---
    ee_lift_final = lift_result["final_ee_position"]
    transport_target = [ee_lift_final[0] + transport_delta_xy[0], ee_lift_final[1] + transport_delta_xy[1], ee_lift_final[2]]
    transport_result = move_to_target(backend, transport_target, PHASE_TRANSPORT, 40, FAILURE_IK_FAILED, on_step=wrapped_on_step, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during transport -- cannot proceed to release", FAILURE_OBJECT_DROPPED, phase=PHASE_TRANSPORT)

    if not backend.use_bin:
        raise RuntimeError("run_pick_and_place_episode_v2 currently only supports backend.use_bin=True (matches this task's own Stage 1A/1B bin-based scenes)")

    bin_place_result = run_bin_place_segment(backend, on_step=wrapped_on_step)

    return {
        "object_yaw_rad": object_yaw_rad,
        "grasp_plan": {
            "grasp_position": plan.grasp_position, "pre_grasp_position": plan.pre_grasp_position,
            "target_gripper_yaw": plan.target_gripper_yaw, "selected_orientation_candidate": plan.selected_orientation_candidate,
            "target_end_effector_quaternion": plan.target_end_effector_quaternion,
            "effective_grasp_width_m": plan.effective_grasp_width_m, "approach_height_m": plan.approach_height_m,
        },
        "pre_grasp": pre_grasp_result, "approach": approach_result, "grasp_position_ee": grasp_position,
        "lift": lift_result, "transition_to_neutral": transition_result, "transport": transport_result,
        "bin_place_result": bin_place_result,
        "monitor": {
            "max_joint_jump": monitor.max_joint_jump, "joint_limit_violation": monitor.joint_limit_violation,
            "orientation_error_max_rad": monitor.orientation_error_max, "collision_detected": monitor.collision_detected,
        },
    }
