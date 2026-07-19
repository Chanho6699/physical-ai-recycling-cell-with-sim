"""SO-101 scripted pick-and-place expert -- SHARED between
benchmark/smoke_so101_pick_and_place.py and benchmark/collect_so101_episode.py
(see this task's chat report, "공통 Expert 모듈 정리"). Neither of those
files reimplements the phase sequence, waypoint/threshold constants, IK
target computation, gripper open/close, success judgment, or
failure_reason classification anymore -- both call run_pick_and_place_episode()
below.

All constants and control-flow here are moved AS-IS from the
already-validated benchmark/smoke_so101_pick_and_place.py (same values,
same thresholds, same step limits) -- no new behavior, no threshold
change. The only two behavioral differences from that file's original,
now-removed local implementation are:
  1. Each move/gripper step calls compute_joint_target_from_ee_delta()
     ONCE and applies that SAME target via apply_joint_target() (see
     robot_sim/so101_pybullet_backend.py's own docstrings) -- the
     previous recorder-side duplicate IK computation is gone.
  2. failure_reason is now classified into one of a fixed set (see
     FAILURE_* constants) instead of being left as a raw exception
     message string.

Does not import from or modify robot_sim/pybullet_panda_backend.py or
any V2/V3 pipeline file.
"""

import math
from collections import deque

import pybullet as p

from robot_sim.so101_pybullet_backend import ARM_JOINT_NAMES, So101PyBulletBackend

# --- Phase names (see this task's chat report, "phase 정보") --- Purely
# an analysis/bookkeeping label -- never fed into a policy observation.
PHASE_PRE_GRASP = "pre_grasp"
PHASE_APPROACH = "approach"
PHASE_GRASP = "grasp"
PHASE_LIFT = "lift"
PHASE_TRANSPORT = "transport"
PHASE_PLACE_DESCEND = "place_descend"
PHASE_RELEASE = "release"
PHASE_SETTLE = "settle"

PHASE_SEQUENCE = [
    PHASE_PRE_GRASP, PHASE_APPROACH, PHASE_GRASP, PHASE_LIFT,
    PHASE_TRANSPORT, PHASE_PLACE_DESCEND, PHASE_RELEASE, PHASE_SETTLE,
]
PHASE_ID_BY_NAME = {name: index for index, name in enumerate(PHASE_SEQUENCE)}
PHASE_NAME_BY_ID = {index: name for name, index in PHASE_ID_BY_NAME.items()}

# --- failure_reason categories (see this task's chat report, "실패 원인
# 세분화") --- Bucketing decisions (documented here, not hidden):
#   - IK_FAILED: abnormal IK step error / non-finite EE / joint-limit
#     violation during pre_grasp, approach, transport, or place_descend.
#   - LIFT_FAILED: the same class of IK/step-level exception, but
#     specifically during the lift phase (named separately per this
#     task's own required list).
#   - GRASP_FAILED: gripper closed but no grasp constraint was ever
#     established.
#   - OBJECT_DROPPED: a grasp that WAS established is lost (is_grasped()
#     goes False) during lift/transport/place_descend.
#   - PLACE_OUTSIDE_TARGET: released and settled, but the object's final
#     xy/height is outside the pass tolerance.
#   - SETTLE_FAILED: released, but final linear/angular velocity or
#     recent drift never dropped below the pass tolerance (independent
#     of xy/height).
FAILURE_IK_FAILED = "ik_failed"
FAILURE_GRASP_FAILED = "grasp_failed"
FAILURE_LIFT_FAILED = "lift_failed"
FAILURE_OBJECT_DROPPED = "object_dropped"
FAILURE_PLACE_OUTSIDE_TARGET = "place_outside_target"
FAILURE_SETTLE_FAILED = "settle_failed"

# --- Bin-specific failure_reason categories (see this task's chat
# report, "production place_success를 실제 bin 결과 기준으로 계산") --
# ONLY ever produced when backend.use_bin is True (see
# evaluate_bin_place_success() below); the flat-target FAILURE_* set
# above is completely untouched and still the only vocabulary used
# when backend.use_bin is False. FAILURE_SCENE_INVALID is included for
# completeness/testability of evaluate_bin_place_success() as a pure
# function -- in practice a bad scene is caught by
# So101PyBulletBackend.reset()'s own InvalidSceneLayoutError BEFORE
# run_pick_and_place_episode() is ever called, so this function itself
# never actually produces it from a real run.
FAILURE_SCENE_INVALID = "scene_invalid"
FAILURE_PLACE_WAYPOINT_FAILED = "place_waypoint_failed"
FAILURE_RELEASE_FAILED = "release_failed"
FAILURE_OBJECT_OUTSIDE_BIN = "object_outside_bin"
FAILURE_OBJECT_NOT_BELOW_RIM = "object_not_below_rim"
FAILURE_UNKNOWN_PLACE_FAILURE = "unknown_place_failure"


class So101ExpertError(RuntimeError):
    """Raised for any hard mid-episode failure -- carries a `failure_reason`
    (one of the FAILURE_* constants above) and the `phase` it occurred
    in, so callers can classify the failure without parsing the message
    string. Still a RuntimeError, so existing exception-catching call
    sites (e.g. smoke_so101_pick_and_place.py's own top-level
    try/except) work unchanged."""

    def __init__(self, message: str, failure_reason: str, phase: str = None):
        super().__init__(message)
        self.failure_reason = failure_reason
        self.phase = phase


# Waypoint / phase constants -- values UNCHANGED from
# smoke_so101_pick_and_place.py's own already-validated constants.
PRE_GRASP_OFFSET_M = [0.0, 0.0, 0.08]
APPROACH_OFFSET_M = [0.0, 0.0, 0.03]
FAR_OFFSET_M = [0.0, 0.0, 0.20]
LIFT_DISTANCE_M = 0.08
PLACE_APPROACH_HEIGHT_ABOVE_SURFACE_M = 0.03

# --- Bin-aware place path (see this task's chat report, "bin에 맞는
# 안전한 place 경로") --- ONLY used when backend.use_bin is True (see
# run_pick_and_place_episode()'s own branch below); the flat-target
# path above (PLACE_APPROACH_HEIGHT_ABOVE_SURFACE_M etc.) is completely
# untouched and still used verbatim when backend.use_bin is False.
# Clearances are relative to the bin's own rim_z (read from
# backend.get_bin_debug_info() at call time -- never a hardcoded
# absolute z here).
BIN_PRE_PLACE_CLEARANCE_M = 0.08
BIN_RELEASE_CLEARANCE_M = 0.03
BIN_RETREAT_CLEARANCE_M = 0.10
# Fixed wait after gripper-open before retreating (see this task's own
# "release 후 즉시 retreat하지 말고... 짧은 고정 step 대기를 둔다") --
# a plain fixed-step wait, not a state machine; object separation
# during it is reported as diagnostic info via the EXISTING
# grasp_constraint_id mechanism, not used as an early-exit condition
# (separation typically already happened inside gripper_phase()'s own
# set_gripper() call, before this wait even starts, so using it as an
# early exit would make the wait a no-op).
BIN_RELEASE_WAIT_STEPS = 45

# --- Bin place_success tolerances (see this task's chat report,
# "tolerance 상수") --- Both are small (same order of magnitude as this
# file's own existing CONVERGENCE_TOLERANCE_M=0.005) -- chosen from
# actual bin geometry margins, NOT inflated to raise a success rate:
# the 10-seed diagnostic benchmark's own measurements showed the
# object settling with ~0.06m of center-to-rim clearance and
# ~0.06m+ of xy clearance from the inner-bounds edge in every
# non-failing seed, so a 0.005m tolerance does not paper over any
# observed near-miss -- it only guards against float-level edge cases
# at the boundary itself.
BIN_CENTER_BELOW_RIM_TOLERANCE_M = 0.005
BIN_INNER_XY_TOLERANCE_M = 0.005

MAX_STEP_M = 0.02
CONVERGENCE_TOLERANCE_M = 0.005
STEP_ERROR_FAILURE_THRESHOLD_M = 0.03
MAX_MOVE_STEPS = 50
LIFT_MAX_STEPS = 20
TRANSPORT_MAX_STEPS = 40

# SETTLE_STEP_CHUNK/SETTLE_CHUNKS: legacy constants from the former
# fixed-360-step single-point settle check (see this task's chat
# report, "연속 안정화 기반 settle 판정"). No longer used to gate
# place_success -- kept only because SETTLE_STEP_CHUNK also defines the
# lookback window for the CONTINUOUS drift check below (DRIFT_WINDOW_STEPS),
# preserving the exact window size SETTLE_DRIFT_PASS_M was originally
# calibrated against.
SETTLE_STEP_CHUNK = 30
SETTLE_CHUNKS = 12

TARGET_XY_ERROR_PASS_M = 0.03
RESTING_HEIGHT_ERROR_PASS_M = 0.01
LINEAR_SPEED_PASS_MPS = 0.03
ANGULAR_SPEED_PASS_RADPS = 0.3
SETTLE_DRIFT_PASS_M = 0.003

# --- Continuous-stability settle judgment (see this task's chat
# report, "연속 안정화 기반 settle 판정") --- REPLACES the former
# fixed-360-step single-instant check. All threshold VALUES above
# (LINEAR_SPEED_PASS_MPS/ANGULAR_SPEED_PASS_RADPS/SETTLE_DRIFT_PASS_M/
# TARGET_XY_ERROR_PASS_M/RESTING_HEIGHT_ERROR_PASS_M) are unchanged --
# only WHEN they get checked, and for how long they must hold
# continuously, has changed.
MAX_SETTLE_STEPS = 1080
CONTINUOUS_STABLE_STEPS = 120
# Checked every physics step (this task's own stated priority: "가능하면
# 매 physics step 검사하는 쪽을 우선하라") -- not a coarser interval.
SETTLE_CHECK_INTERVAL_STEPS = 1
# Drift is displacement over the last DRIFT_WINDOW_STEPS steps (a
# sliding window, re-evaluated every check) -- same window size the
# former fixed-chunk check used, so SETTLE_DRIFT_PASS_M's calibration
# still means the same physical thing.
DRIFT_WINDOW_STEPS = SETTLE_STEP_CHUNK

JOINT_LIMIT_EPS = 1e-6


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def object_offset_in_ee_frame(ee_position: list, ee_orientation: list, object_position: list) -> list:
    ee_pos_inv, ee_orn_inv = p.invertTransform(ee_position, ee_orientation)
    local_position, _local_orientation = p.multiplyTransforms(ee_pos_inv, ee_orn_inv, object_position, [0, 0, 0, 1])
    return list(local_position)


def check_joint_limits(backend: So101PyBulletBackend, joint_positions: list) -> list:
    violations = []
    for name, pos in zip(ARM_JOINT_NAMES, joint_positions):
        info = backend.joint_info_by_name[name]
        if pos < info["lower"] - JOINT_LIMIT_EPS or pos > info["upper"] + JOINT_LIMIT_EPS:
            violations.append({"joint": name, "position": pos, "lower": info["lower"], "upper": info["upper"]})
    return violations


def gripper_phase(backend: So101PyBulletBackend, phase: str, gripper_target_normalized: float, on_step=None) -> dict:
    """Records (via on_step, if given) the CURRENT arm joint positions as
    the held target + the new gripper target, BEFORE actually applying
    it -- mirrors move_to_target()'s own observation_t -> action_t
    ordering for the gripper-only case."""
    current_joint_positions = backend.get_joint_positions()
    if on_step is not None:
        on_step(phase, current_joint_positions, gripper_target_normalized)
    return backend.set_gripper(gripper_target_normalized)


def move_to_target(
    backend: So101PyBulletBackend, target_position: list, phase: str, max_steps: int,
    failure_reason: str, on_step=None, track_grasp: bool = False,
) -> dict:
    """Unified stepper for ALL move phases (pre_grasp/approach/lift/
    transport/place_descend) -- merges smoke_so101_pick_and_place.py's
    former move_to_target() (track_grasp=False) and
    move_with_grasp_tracking() (track_grasp=True) into one function; the
    actual step loop (target convergence, MAX_STEP_M clamp, IK call,
    error/joint-limit checks) is identical between the two former
    functions, so merging changes no numeric behavior.

    Each step computes the absolute joint target via
    backend.compute_joint_target_from_ee_delta() ONCE, optionally hands
    it to on_step(phase, arm_joint_targets, gripper_target_normalized)
    for recording BEFORE applying, then applies that SAME target via
    backend.apply_joint_target() -- no second IK call (see this task's
    chat report, "IK 단일 계산 구조")."""
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
        current_ee_position, _ = backend.get_end_effector_pose()
        remaining = [target_position[i] - current_ee_position[i] for i in range(3)]
        remaining_norm = math.sqrt(sum(c ** 2 for c in remaining))
        if remaining_norm <= CONVERGENCE_TOLERANCE_M:
            break
        clamped_delta = [max(-MAX_STEP_M, min(MAX_STEP_M, c)) for c in remaining]

        computed = backend.compute_joint_target_from_ee_delta(clamped_delta)
        if on_step is not None:
            gripper_current_normalized = backend.get_observation()["gripper_position_normalized"]
            on_step(phase, computed["arm_joint_targets"], gripper_current_normalized)

        obs = backend.apply_joint_target(computed["arm_joint_targets"])
        final_ee_position, _ = backend.get_end_effector_pose()
        position_error = math.sqrt(sum((final_ee_position[i] - computed["target_position"][i]) ** 2 for i in range(3)))
        obs["ee_delta_target_position"] = computed["target_position"]
        obs["ee_delta_position_error"] = position_error

        if not all_finite(obs["end_effector_position"]):
            raise So101ExpertError(f"[{phase}] non-finite EE position at step {step_index}", failure_reason, phase=phase)
        if obs["ee_delta_position_error"] > STEP_ERROR_FAILURE_THRESHOLD_M:
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

    final_ee_position, _ = backend.get_end_effector_pose()
    final_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
    result = {"target": target_position, "final_ee_position": final_ee_position, "error": final_error, "num_steps": step_index}
    if track_grasp:
        result.update({
            "ee_start_position": ee_start, "object_start_position": object_start,
            "object_final_position": backend.get_object_position(),
            "max_relative_drift_m": max_relative_drift,
            "grasp_maintained_all_steps": grasp_maintained_all_steps,
            "constraint_valid_all_steps": constraint_valid_all_steps,
        })
    return result


def _bin_wall_body_ids(backend: So101PyBulletBackend) -> list:
    bin_info = backend.get_bin_debug_info()
    if bin_info is None:
        return []
    return [bin_info["body_ids"][name] for name in ("left_wall", "right_wall", "front_wall", "back_wall")]


def _check_bin_contacts(backend: So101PyBulletBackend, body_a_id: int) -> dict:
    """Read-only PyBullet contact query against the bin's own 4 wall
    bodies (see this task's chat report, "충돌 안전성 확인") -- does
    NOT raise or classify a failure_reason from this; purely
    observational/diagnostic. `max_normal_force` is a pure addition
    (see this task's chat report, "diagnostic benchmark") -- existing
    callers reading only "contact_count"/"contacts" are unaffected."""
    wall_ids = _bin_wall_body_ids(backend)
    contacts = []
    for wall_id in wall_ids:
        contacts.extend(p.getContactPoints(bodyA=body_a_id, bodyB=wall_id, physicsClientId=backend.client_id))
    max_normal_force = max((c[9] for c in contacts), default=0.0)
    return {"contact_count": len(contacts), "contacts": contacts, "max_normal_force": max_normal_force}


def run_bin_place_segment(backend: So101PyBulletBackend, on_step=None) -> dict:
    """Bin-specific place path -- pre-place (well above rim) -> nearly-
    vertical descend (x,y fixed at bin center) -> release -> fixed wait
    -> nearly-vertical retreat (see this task's chat report, "bin에
    맞는 안전한 place 경로"). Called ONLY from
    run_pick_and_place_episode()'s own `backend.use_bin` branch below --
    entirely separate from, and never invoked by, the flat-target path.

    Reuses move_to_target()/gripper_phase() EXACTLY as every other
    phase does -- no new motion planner. The lateral move (wherever
    transport left the EE -> bin center) happens ONLY at pre_place_z
    (well above rim_z + BIN_PRE_PLACE_CLEARANCE_M), so it never crosses
    a wall; both height changes after that (descend, retreat) target
    the SAME (x, y) as the waypoint before them, so move_to_target()'s
    own straight-line-to-target behavior makes them near-vertical by
    construction, not by a special-cased planner.

    Bin center/rim_z are read from backend.get_bin_debug_info() at call
    time (never hardcoded/duplicated here). PHASE_PLACE_DESCEND/
    PHASE_RELEASE (the EXISTING phase names) are reused for on_step's
    phase argument -- see this task's own "기존 phase_id 호환성을
    유지" -- bin-specific waypoint identity is only in this function's
    own returned "debug" dict, never in on_step's fixed 3-argument
    signature, so dataset schema/phase_id dimensionality are untouched."""
    bin_info = backend.get_bin_debug_info()
    if bin_info is None:
        raise RuntimeError("run_bin_place_segment() requires backend.use_bin=True")

    center_x, center_y, rim_z = bin_info["center_x"], bin_info["center_y"], bin_info["rim_z"]
    pre_place_target = [center_x, center_y, rim_z + BIN_PRE_PLACE_CLEARANCE_M]
    release_target = [center_x, center_y, rim_z + BIN_RELEASE_CLEARANCE_M]
    retreat_target = [center_x, center_y, rim_z + BIN_RETREAT_CLEARANCE_M]

    # sub_phase-tagged contact logs (see this task's chat report,
    # "diagnostic benchmark" -- section 4E's per-phase contact request)
    # -- PURELY ADDITIVE diagnostic data, distinct from the "phase"
    # string passed to on_step (which stays PHASE_PLACE_DESCEND/
    # PHASE_RELEASE for dataset/phase_id compatibility, unchanged).
    robot_bin_contact_log = []
    object_bin_contact_log = []
    _step_counters = {}

    def make_contact_checking_on_step(sub_phase: str):
        _step_counters[sub_phase] = 0

        def contact_checking_on_step(phase, arm_joint_targets, gripper_target_normalized):
            if on_step is not None:
                on_step(phase, arm_joint_targets, gripper_target_normalized)
            _step_counters[sub_phase] += 1
            step_index = _step_counters[sub_phase]

            robot_contact = _check_bin_contacts(backend, backend.robot_id)
            if robot_contact["contact_count"] > 0:
                robot_bin_contact_log.append({
                    "phase": phase, "sub_phase": sub_phase, "step": step_index,
                    "contact_count": robot_contact["contact_count"], "max_normal_force": robot_contact["max_normal_force"],
                })
            # The carried object is rigidly attached during rise/pre_place/
            # descend -- if IT clips a wall (not just the robot), that's
            # equally worth surfacing here (diagnostic only).
            object_contact = _check_bin_contacts(backend, backend.object_id)
            if object_contact["contact_count"] > 0:
                object_bin_contact_log.append({
                    "phase": phase, "sub_phase": sub_phase, "step": step_index,
                    "contact_count": object_contact["contact_count"], "max_normal_force": object_contact["max_normal_force"],
                })

        return contact_checking_on_step

    # Rise FIRST (straight up at the CURRENT x,y transport left the EE
    # at), THEN move laterally to the bin center -- both at/above
    # pre_place_z. move_to_target() interpolates in a straight 3D line
    # to its target, so simply setting pre_place_target's z high is NOT
    # by itself enough to guarantee a safe approach: if the starting
    # point is below rim_z, a single diagonal move straight to
    # pre_place_target would cut across at wall height partway through
    # (exactly the "사선으로 bin 벽을 가로질러 접근" this task
    # explicitly warns against). Splitting into rise-then-lateral
    # keeps the ENTIRE approach at or above pre_place_z throughout.
    current_ee_position, _ = backend.get_end_effector_pose()
    rise_target = [current_ee_position[0], current_ee_position[1], max(current_ee_position[2], pre_place_target[2])]
    rise_result = move_to_target(backend, rise_target, PHASE_PLACE_DESCEND, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=make_contact_checking_on_step("rise"), track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during bin pre-place rise", FAILURE_OBJECT_DROPPED, phase=PHASE_PLACE_DESCEND)

    pre_place_result = move_to_target(backend, pre_place_target, PHASE_PLACE_DESCEND, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=make_contact_checking_on_step("pre_place"), track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during bin pre-place", FAILURE_OBJECT_DROPPED, phase=PHASE_PLACE_DESCEND)

    descend_result = move_to_target(backend, release_target, PHASE_PLACE_DESCEND, LIFT_MAX_STEPS, FAILURE_IK_FAILED, on_step=make_contact_checking_on_step("descend"), track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during bin descend", FAILURE_OBJECT_DROPPED, phase=PHASE_PLACE_DESCEND)

    object_release_position = backend.get_object_position()
    object_bin_contact_before_release = _check_bin_contacts(backend, backend.object_id)
    # Real (not approximated/recomputed-from-half-extent) AABB, orientation-
    # aware -- see this task's chat report, "release 순간" diagnostics.
    object_aabb_before_release_min, object_aabb_before_release_max = p.getAABB(backend.object_id, physicsClientId=backend.client_id)
    ee_position_before_release, _ = backend.get_end_effector_pose()

    gripper_phase(backend, PHASE_RELEASE, 1.0, on_step)
    release_constraint_removed = backend.get_grasp_state()["grasp_constraint_id"] is None
    grasp_state_after_release = backend.get_grasp_state()
    object_position_immediately_after_release = backend.get_object_position()
    object_aabb_after_release_min, object_aabb_after_release_max = p.getAABB(backend.object_id, physicsClientId=backend.client_id)
    ee_position_after_release, _ = backend.get_end_effector_pose()

    wait_steps_used = 0
    for _ in range(BIN_RELEASE_WAIT_STEPS):
        backend.step(1)
        wait_steps_used += 1
        robot_contact = _check_bin_contacts(backend, backend.robot_id)
        if robot_contact["contact_count"] > 0:
            robot_bin_contact_log.append({
                "phase": PHASE_RELEASE, "sub_phase": "release_wait", "step": wait_steps_used,
                "contact_count": robot_contact["contact_count"], "max_normal_force": robot_contact["max_normal_force"],
            })
    object_separated_during_wait = backend.get_grasp_state()["grasp_constraint_id"] is None
    object_bin_contact_after_wait = _check_bin_contacts(backend, backend.object_id)

    retreat_result = move_to_target(backend, retreat_target, PHASE_RELEASE, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=make_contact_checking_on_step("retreat"))

    return {
        "rise_result": rise_result, "pre_place_result": pre_place_result, "descend_result": descend_result, "retreat_result": retreat_result,
        "object_release_position": object_release_position,
        "release_constraint_removed": release_constraint_removed, "grasp_state_after_release": grasp_state_after_release,
        "object_position_immediately_after_release": object_position_immediately_after_release,
        "release_wait_steps_used": wait_steps_used, "object_separated_during_wait": object_separated_during_wait,
        "debug": {
            "bin_center": [center_x, center_y], "rim_z": rim_z,
            "rise_target": rise_target, "rise_final_ee": rise_result["final_ee_position"], "rise_error_m": rise_result["error"],
            "rise_reached": rise_result["error"] <= CONVERGENCE_TOLERANCE_M,
            "pre_place_target": pre_place_target, "release_target": release_target, "retreat_target": retreat_target,
            "pre_place_final_ee": pre_place_result["final_ee_position"], "pre_place_error_m": pre_place_result["error"],
            "pre_place_reached": pre_place_result["error"] <= CONVERGENCE_TOLERANCE_M,
            "descend_final_ee": descend_result["final_ee_position"], "descend_error_m": descend_result["error"],
            "descend_reached": descend_result["error"] <= CONVERGENCE_TOLERANCE_M,
            "retreat_final_ee": retreat_result["final_ee_position"], "retreat_error_m": retreat_result["error"],
            "retreat_reached": retreat_result["error"] <= CONVERGENCE_TOLERANCE_M,
            "robot_bin_contact_log": robot_bin_contact_log,
            "robot_bin_contact_count_total": len(robot_bin_contact_log),
            "object_bin_contact_log": object_bin_contact_log,
            "object_bin_contact_before_release_count": object_bin_contact_before_release["contact_count"],
            "object_bin_contact_after_wait_count": object_bin_contact_after_wait["contact_count"],
            "ee_position_before_release": ee_position_before_release, "ee_position_after_release": ee_position_after_release,
            "object_aabb_before_release": {"min": list(object_aabb_before_release_min), "max": list(object_aabb_before_release_max)},
            "object_aabb_after_release": {"min": list(object_aabb_after_release_min), "max": list(object_aabb_after_release_max)},
        },
    }


def evaluate_bin_place_success(bin_success_debug: dict) -> tuple:
    """Pure function (see this task's chat report, "production 판정
    계산을 작은 pure helper 함수로 분리") -- no PyBullet/backend access
    at all, fully testable with a controlled dict (see
    benchmark/smoke_so101_bin_success_criterion.py). Returns
    (place_success, failure_reason, failure_phase).

    Priority order (see this task's chat report, section 4) -- first
    failing condition wins, single failure_reason per call:
      scene_invalid > place_waypoint_failed > release_failed >
      object_outside_bin > object_not_below_rim > settle_failed >
      unknown_place_failure
    (grasp_failed/lift_failed/transport_failed are NOT decided here --
    those already raise So101ExpertError before this function is ever
    reached; a 0N grazing robot-bin contact is NEVER checked here at
    all -- see this task's own "0N grazing contact를 production 실패
    조건으로 추가하지 않는다")."""
    d = bin_success_debug
    if not d["layout_validation_passed"]:
        return False, FAILURE_SCENE_INVALID, None
    if not d["manipulation_steps_completed"]:
        return False, FAILURE_UNKNOWN_PLACE_FAILURE, None
    if not d["place_waypoint_reached"]:
        return False, FAILURE_PLACE_WAYPOINT_FAILED, PHASE_PLACE_DESCEND
    if not d["object_separated"]:
        return False, FAILURE_RELEASE_FAILED, PHASE_RELEASE
    if not d["inside_inner_xy"]:
        return False, FAILURE_OBJECT_OUTSIDE_BIN, PHASE_PLACE_DESCEND
    if not d["object_center_below_rim"]:
        return False, FAILURE_OBJECT_NOT_BELOW_RIM, PHASE_PLACE_DESCEND
    if not d["settle_success"]:
        return False, FAILURE_SETTLE_FAILED, PHASE_SETTLE
    return True, None, None


def compute_bin_success_debug(
    backend: So101PyBulletBackend, bin_place_debug: dict, release_constraint_removed: bool, final_object_position: list,
    settle_success: bool, layout_validation_passed: bool,
) -> dict:
    """Gathers the live-backend-dependent inputs
    evaluate_bin_place_success() needs (see this task's chat report,
    section 3, "bin_success_debug") -- object AABB/bin inner bounds/
    rim_z are read from the SAME backend.get_bin_debug_info() API the
    rest of this module already uses, never hardcoded. Object-CENTER
    inside the inner bounds is the primary "inside_inner_xy" test; the
    live AABB is used only as a secondary guard against the object
    protruding past the inner boundary by more than
    BIN_INNER_XY_TOLERANCE_M (see this task's own "object AABB 일부가
    벽 바깥으로 심하게 돌출된 경우를 막을 수 있는 tolerance")."""
    bin_info = backend.get_bin_debug_info()
    rim_z = bin_info["rim_z"]
    inner_bounds = {
        "x_min": bin_info["inner_x_min"], "x_max": bin_info["inner_x_max"],
        "y_min": bin_info["inner_y_min"], "y_max": bin_info["inner_y_max"],
    }

    final_aabb_min, final_aabb_max = p.getAABB(backend.object_id, physicsClientId=backend.client_id)
    object_final_aabb = {"min": list(final_aabb_min), "max": list(final_aabb_max)}

    center_inside = (
        inner_bounds["x_min"] <= final_object_position[0] <= inner_bounds["x_max"]
        and inner_bounds["y_min"] <= final_object_position[1] <= inner_bounds["y_max"]
    )
    protrusion_x = max(inner_bounds["x_min"] - final_aabb_min[0], final_aabb_max[0] - inner_bounds["x_max"], 0.0)
    protrusion_y = max(inner_bounds["y_min"] - final_aabb_min[1], final_aabb_max[1] - inner_bounds["y_max"], 0.0)
    inside_inner_xy = center_inside and protrusion_x <= BIN_INNER_XY_TOLERANCE_M and protrusion_y <= BIN_INNER_XY_TOLERANCE_M

    center_rim_delta = final_object_position[2] - rim_z
    top_rim_delta = final_aabb_max[2] - rim_z
    object_center_below_rim = center_rim_delta < -BIN_CENTER_BELOW_RIM_TOLERANCE_M
    # Diagnostic-only (see this task's own "top-below-rim은 diagnostic
    # field로 계속 기록... 처음부터 top-below-rim을 필수 production
    # 조건으로 넣지 말 것") -- never gates place_success.
    object_top_below_rim = top_rim_delta < 0.0

    object_separated = bool(release_constraint_removed) and bool(bin_place_debug["object_separated_during_wait"])
    place_waypoint_reached = all([
        bin_place_debug["rise_reached"], bin_place_debug["pre_place_reached"],
        bin_place_debug["descend_reached"], bin_place_debug["retreat_reached"],
    ])
    # Reaching this point at all means grasp/lift/transport/rise/
    # pre_place/descend never raised So101ExpertError -- a mid-sequence
    # failure never gets here to be silently overwritten by a lucky
    # final pose (see this task's own "중간 단계 실패가 있었는데 최종
    # pose만 우연히 bin 안이면 성공 처리하지 말 것").
    manipulation_steps_completed = True

    failed_conditions = []
    if not layout_validation_passed:
        failed_conditions.append("layout_validation_passed")
    if not manipulation_steps_completed:
        failed_conditions.append("manipulation_steps_completed")
    if not place_waypoint_reached:
        failed_conditions.append("place_waypoint_reached")
    if not object_separated:
        failed_conditions.append("object_separated")
    if not inside_inner_xy:
        failed_conditions.append("inside_inner_xy")
    if not object_center_below_rim:
        failed_conditions.append("object_center_below_rim")
    if not settle_success:
        failed_conditions.append("settle_success")

    return {
        "layout_validation_passed": layout_validation_passed,
        "object_separated": object_separated,
        "inside_inner_xy": inside_inner_xy,
        "object_center_below_rim": object_center_below_rim,
        "object_top_below_rim": object_top_below_rim,
        "settle_success": settle_success,
        "manipulation_steps_completed": manipulation_steps_completed,
        "place_waypoint_reached": place_waypoint_reached,
        "object_final_xyz": final_object_position,
        "object_final_aabb": object_final_aabb,
        "bin_inner_bounds": inner_bounds,
        "rim_z": rim_z,
        "center_rim_delta": center_rim_delta,
        "top_rim_delta": top_rim_delta,
        "failed_conditions": failed_conditions,
    }


def run_pick_and_place_episode(
    backend: So101PyBulletBackend, transport_delta_xy: list, on_step=None,
    record_settle_trace: bool = False,
) -> dict:
    """approach -> grasp -> lift -> transport -> release -> settle (see
    this module's own docstring). Does NOT reset() -- caller decides
    when to reset. Return dict shape is UNCHANGED from
    smoke_so101_pick_and_place.py's own former run_episode() (same keys
    retained, same threshold VALUES), with pure additions:
    "failure_reason"/"failure_phase", "object_position_immediately_after_release",
    "settle_trace", and the new settle-diagnostic fields listed below.

    `on_step(phase, arm_joint_targets, gripper_target_normalized)`, if
    given, is called BEFORE every actual arm/gripper apply during
    pre_grasp/approach/grasp/lift/transport/place_descend/release --
    NOT during settle (settle is dynamics-only, no new arm/gripper
    command is issued, so there is nothing to record there).

    Settle judgment (see this task's chat report, "연속 안정화 기반
    settle 판정"): REPLACES the former fixed-360-step single-instant
    check. After release, steps SETTLE_CHECK_INTERVAL_STEPS (=1, i.e.
    every physics step) at a time, up to MAX_SETTLE_STEPS (1080),
    measuring linear/angular speed and drift (over a sliding
    DRIFT_WINDOW_STEPS-step window) at every check. A pass increments a
    consecutive-stable counter; any failure resets it to 0. Reaching
    CONTINUOUS_STABLE_STEPS (120) consecutive passes -> settled (loop
    stops early); never reaching it within MAX_SETTLE_STEPS -> timeout.
    Threshold VALUES (LINEAR_SPEED_PASS_MPS/ANGULAR_SPEED_PASS_RADPS/
    SETTLE_DRIFT_PASS_M) are unchanged.

    failure_reason priority (see this task's chat report, "판정 순서"):
      1. settle never achieves CONTINUOUS_STABLE_STEPS within
         MAX_SETTLE_STEPS -> FAILURE_SETTLE_FAILED (place_success=False,
         regardless of where the object ended up).
      2. settle succeeds, but the STABILIZED final position's xy/height
         is outside tolerance -> FAILURE_PLACE_OUTSIDE_TARGET.
      3. settle succeeds AND xy/height are within tolerance -> success.

    `record_settle_trace`, if True, appends one entry per physics-step
    check (step, linear/angular speed, drift, object position/
    orientation, contact_count, max_contact_normal_force, pass/fail,
    running consecutive_stable_count) to "settle_trace" -- diagnostic
    only, default False (zero extra list-building/contact-query
    overhead when not requested)."""
    object_position, _ = backend.get_object_pose()
    scene = backend.get_scene_state()
    surface_height = scene["table_top_z"]
    object_half_height = backend.scene_config["object_height"] / 2.0
    target_zone_center_xy = scene["target_zone_center_xy"]

    gripper_phase(backend, PHASE_PRE_GRASP, 1.0, on_step)
    pre_grasp_result = move_to_target(backend, [object_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)], PHASE_PRE_GRASP, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=on_step)
    approach_result = move_to_target(backend, [object_position[i] + APPROACH_OFFSET_M[i] for i in range(3)], PHASE_APPROACH, MAX_MOVE_STEPS, FAILURE_IK_FAILED, on_step=on_step)
    gripper_phase(backend, PHASE_GRASP, 0.0, on_step)
    grasp_succeeded = backend.is_grasped()
    if not grasp_succeeded:
        raise So101ExpertError("grasp was not established -- cannot proceed to lift/transport", FAILURE_GRASP_FAILED, phase=PHASE_GRASP)

    grasp_position, _ = backend.get_end_effector_pose()

    ee_pre_lift, _ = backend.get_end_effector_pose()
    lift_target = [ee_pre_lift[0], ee_pre_lift[1], ee_pre_lift[2] + LIFT_DISTANCE_M]
    lift_result = move_to_target(backend, lift_target, PHASE_LIFT, LIFT_MAX_STEPS, FAILURE_LIFT_FAILED, on_step=on_step, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during lift -- cannot proceed to transport", FAILURE_OBJECT_DROPPED, phase=PHASE_LIFT)

    ee_lift_final = lift_result["final_ee_position"]
    transport_target = [ee_lift_final[0] + transport_delta_xy[0], ee_lift_final[1] + transport_delta_xy[1], ee_lift_final[2]]
    transport_result = move_to_target(backend, transport_target, PHASE_TRANSPORT, TRANSPORT_MAX_STEPS, FAILURE_IK_FAILED, on_step=on_step, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was lost during transport -- cannot proceed to release", FAILURE_OBJECT_DROPPED, phase=PHASE_TRANSPORT)

    ee_transport_final = transport_result["final_ee_position"]

    # --- Place path branch (see this task's chat report, "bin에 맞는
    # 안전한 place 경로") --- backend.use_bin selects which path runs;
    # with it False (the default -- see So101PyBulletBackend's own
    # constructor), this is the EXACT SAME flat-target code that ran
    # before this task, untouched. bin_place_debug stays None on the
    # flat path -- a pure addition to the return dict below, nothing
    # existing reads or depends on it.
    bin_place_debug = None
    if backend.use_bin:
        bin_place_result = run_bin_place_segment(backend, on_step=on_step)
        descend_result = bin_place_result["descend_result"]
        object_release_position = bin_place_result["object_release_position"]
        release_constraint_removed = bin_place_result["release_constraint_removed"]
        grasp_state_after_release = bin_place_result["grasp_state_after_release"]
        object_position_immediately_after_release = bin_place_result["object_position_immediately_after_release"]
        bin_place_debug = {
            "rise_result": bin_place_result["rise_result"], "pre_place_result": bin_place_result["pre_place_result"], "retreat_result": bin_place_result["retreat_result"],
            "release_wait_steps_used": bin_place_result["release_wait_steps_used"],
            "object_separated_during_wait": bin_place_result["object_separated_during_wait"],
            **bin_place_result["debug"],
        }
    else:
        place_approach_target = [ee_transport_final[0], ee_transport_final[1], surface_height + PLACE_APPROACH_HEIGHT_ABOVE_SURFACE_M]
        descend_result = move_to_target(backend, place_approach_target, PHASE_PLACE_DESCEND, LIFT_MAX_STEPS, FAILURE_IK_FAILED, on_step=on_step, track_grasp=True)
        if not backend.is_grasped():
            raise So101ExpertError("grasp was lost during place-descend -- cannot proceed to release", FAILURE_OBJECT_DROPPED, phase=PHASE_PLACE_DESCEND)

        object_release_position = backend.get_object_position()

        gripper_phase(backend, PHASE_RELEASE, 1.0, on_step)
        release_constraint_removed = backend.get_grasp_state()["grasp_constraint_id"] is None
        grasp_state_after_release = backend.get_grasp_state()
        # Read-only query, zero physics/control side effect -- capturing
        # this costs nothing and does not change any existing behavior.
        object_position_immediately_after_release = backend.get_object_position()

    # --- Continuous-stability settle loop (see this function's own
    # docstring, "Settle judgment") --- order: (1) step, (2) measure,
    # (3) check all thresholds, (4) pass -> increment consecutive
    # count, (5) fail -> reset to 0, (6) reach CONTINUOUS_STABLE_STEPS
    # -> settled, (7) reach MAX_SETTLE_STEPS without that -> timeout.
    settle_trace = [] if record_settle_trace else None
    position_window = deque(maxlen=DRIFT_WINDOW_STEPS + 1)
    position_window.append(backend.get_object_position())

    consecutive_stable_count = 0
    max_consecutive_stable_count = 0
    settle_success = False
    settle_check_count = 0
    settle_steps_used = MAX_SETTLE_STEPS
    final_linear_speed = final_angular_speed = final_drift = None
    final_object_position = final_object_orientation = None

    for step_index in range(1, MAX_SETTLE_STEPS + 1):
        backend.step(SETTLE_CHECK_INTERVAL_STEPS)
        pos = backend.get_object_position()
        if not all_finite(pos):
            raise So101ExpertError("non-finite object position during settle", FAILURE_SETTLE_FAILED, phase=PHASE_SETTLE)
        position_window.append(pos)

        obj_pos, obj_orn = backend.get_object_pose()
        lin_v, ang_v = backend.get_object_velocity()
        lin_speed = math.sqrt(sum(v ** 2 for v in lin_v))
        ang_speed = math.sqrt(sum(v ** 2 for v in ang_v))

        # Drift needs a FULL DRIFT_WINDOW_STEPS of history to mean the
        # same thing SETTLE_DRIFT_PASS_M was calibrated against; before
        # that (only during the very first DRIFT_WINDOW_STEPS checks),
        # treat this check as not-yet-passing rather than computing
        # drift over an artificially short window.
        if len(position_window) > DRIFT_WINDOW_STEPS:
            window_start = position_window[0]
            drift = math.sqrt(sum((position_window[-1][i] - window_start[i]) ** 2 for i in range(3)))
            drift_measurable = True
        else:
            drift = None
            drift_measurable = False

        settle_check_count += 1
        passes = (
            drift_measurable
            and lin_speed <= LINEAR_SPEED_PASS_MPS
            and ang_speed <= ANGULAR_SPEED_PASS_RADPS
            and drift <= SETTLE_DRIFT_PASS_M
        )

        consecutive_stable_count = consecutive_stable_count + 1 if passes else 0
        max_consecutive_stable_count = max(max_consecutive_stable_count, consecutive_stable_count)

        final_linear_speed, final_angular_speed = lin_speed, ang_speed
        if drift_measurable:
            final_drift = drift
        final_object_position, final_object_orientation = obj_pos, obj_orn

        if settle_trace is not None:
            contacts = p.getContactPoints(bodyA=backend.object_id, physicsClientId=backend.client_id)
            settle_trace.append({
                "step": step_index, "linear_speed_mps": lin_speed, "angular_speed_radps": ang_speed,
                "drift_m": drift, "object_position": obj_pos, "object_orientation": obj_orn,
                "passes": passes, "consecutive_stable_count": consecutive_stable_count,
                "contact_count": len(contacts),
                "max_contact_normal_force": max((c[9] for c in contacts), default=0.0),
            })

        if consecutive_stable_count >= CONTINUOUS_STABLE_STEPS:
            settle_success = True
            settle_steps_used = step_index
            break

    settle_timeout = not settle_success
    final_consecutive_stable_steps = consecutive_stable_count

    target_xy_error = math.sqrt(
        (final_object_position[0] - target_zone_center_xy[0]) ** 2 + (final_object_position[1] - target_zone_center_xy[1]) ** 2
    )
    resting_height_error = abs((final_object_position[2] - object_half_height) - surface_height)

    object_in_target_zone = target_xy_error <= TARGET_XY_ERROR_PASS_M
    resting_height_ok = resting_height_error <= RESTING_HEIGHT_ERROR_PASS_M
    stably_settled = settle_success

    # --- place_success / failure_reason branch (see this task's chat
    # report, "production place_success를 실제 bin 결과 기준으로 계산")
    # --- backend.use_bin selects which judgment runs; with it False
    # (the default), this is the EXACT SAME flat-target logic that ran
    # before this task, untouched -- target_xy_error/resting_height_error
    # above are computed IDENTICALLY either way (target_zone_center_xy
    # already equals the bin's own center when a bin exists, so no
    # formula change was needed there), they are simply not used to
    # GATE place_success on the bin path (kept as a diagnostic/reference
    # metric only, see this task's own "target_xy_error 처리").
    # bin_success_debug stays None on the flat path -- a pure addition
    # to the return dict below.
    bin_success_debug = None
    if backend.use_bin:
        bin_success_debug = compute_bin_success_debug(
            backend, bin_place_debug, release_constraint_removed, final_object_position, settle_success, scene["layout_validation_passed"],
        )
        place_success, failure_reason, failure_phase = evaluate_bin_place_success(bin_success_debug)
    else:
        # failure_reason priority (see this task's chat report, "판정
        # 순서"): (1) settle itself never achieved CONTINUOUS_STABLE_STEPS
        # within MAX_SETTLE_STEPS -> settle_failed, checked FIRST and takes
        # precedence regardless of where the object ended up; (2) settle
        # succeeded but the STABILIZED final position is outside xy/height
        # tolerance -> place_outside_target; (3) both hold -> success.
        if not settle_success:
            failure_reason = FAILURE_SETTLE_FAILED
            failure_phase = PHASE_SETTLE
            place_success = False
        elif not object_in_target_zone or not resting_height_ok:
            failure_reason = FAILURE_PLACE_OUTSIDE_TARGET
            failure_phase = PHASE_PLACE_DESCEND
            place_success = False
        else:
            failure_reason = None
            failure_phase = None
            place_success = True

    object_final_position, object_final_orientation = final_object_position, final_object_orientation
    linear_speed, angular_speed, recent_drift = final_linear_speed, final_angular_speed, final_drift

    return {
        "initial_object_position": object_position,
        "grasp_position": grasp_position,
        "pre_grasp": pre_grasp_result,
        "approach": approach_result,
        "lift": lift_result,
        "transport": transport_result,
        "place_descend": descend_result,
        "object_release_position": object_release_position,
        "target_center_position": target_zone_center_xy,
        "release_constraint_removed": release_constraint_removed,
        "grasp_state_after_release": grasp_state_after_release,
        "object_final_position": object_final_position,
        "object_final_orientation": object_final_orientation,
        "object_target_xy_error_m": target_xy_error,
        "object_resting_height_error_m": resting_height_error,
        "object_final_linear_speed_mps": linear_speed,
        "object_final_angular_speed_radps": angular_speed,
        "object_recent_settle_drift_m": recent_drift,
        "object_in_target_zone": object_in_target_zone,
        "object_stably_settled": stably_settled,
        "resting_height_ok": resting_height_ok,
        "place_success": place_success,
        "failure_reason": failure_reason,
        "failure_phase": failure_phase,
        "object_position_immediately_after_release": object_position_immediately_after_release,
        "settle_trace": settle_trace,
        # --- New continuous-settle fields (see this task's chat report,
        # "결과 필드 추가") ---
        "settle_success": settle_success,
        "settle_steps_used": settle_steps_used,
        "continuous_stable_steps_required": CONTINUOUS_STABLE_STEPS,
        "max_consecutive_stable_steps": max_consecutive_stable_count,
        "final_consecutive_stable_steps": final_consecutive_stable_steps,
        "settle_check_count": settle_check_count,
        "final_linear_speed": final_linear_speed,
        "final_angular_speed": final_angular_speed,
        "final_drift": final_drift,
        "final_object_position": final_object_position,
        "final_xy_error": target_xy_error,
        "settle_timeout": settle_timeout,
        # None on the flat-target path (backend.use_bin=False) -- pure
        # addition, see run_bin_place_segment()'s own docstring.
        "bin_place_debug": bin_place_debug,
        # None on the flat-target path (backend.use_bin=False) -- pure
        # addition, see compute_bin_success_debug()'s own docstring.
        "bin_success_debug": bin_success_debug,
    }
