"""SO-101 PyBullet backend (minimal, v0) -- see this task's chat report
("So101PyBulletBackend 최소 버전 구현") and docs/so101_backend_design_proposal.md
for the interface rationale and the full Panda-vs-SO101 comparison.

Standalone: does not import, get imported by, or modify
robot_sim/pybullet_panda_backend.py, does not touch any V2/V3 pipeline
file, does not load any SmolVLA checkpoint. Reuses the exact URDF path,
joint names, EE link, neutral pose, IK settings, and move force already
validated in benchmark/inspect_so101_urdf.py / smoke_so101_joint_control.py
/ smoke_so101_ik.py (this task does not re-derive any of that).

Scope: reset -> observe -> command joint positions / EE-delta
(position-only IK) -> gripper open/close, on a scene with a support
surface + single object. Grasp is a distance/gripper-state-triggered
PyBullet fixed constraint (see _maybe_trigger_grasp()), same mechanism
PyBulletPandaBackend's own close_gripper() uses, implemented
independently here (no shared code, no import). Still no lift, bin,
place, camera, orientation IK, expert-policy/SmolVLA wiring, or ROS2 --
see the design-proposal doc for what's still missing before this could
replace PyBulletPandaBackend in the real recycling-cell loop.
"""

import math
from pathlib import Path
from typing import Optional

import pybullet as p
import pybullet_data

from robot_sim.camera_utils import capture_pybullet_camera

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = PROJECT_ROOT / "third_party" / "so101_arm" / "so101_new_calib.urdf"

# --- Front camera (see this task's chat report, "SO-101 Dataset Recorder")
# ---
# Fixed, world-space camera framing the support surface + object + target
# zone (surface centered at scene_config's own surface_center_xy, see
# DEFAULT_SCENE_CONFIG below) -- reuses robot_sim/camera_utils.py's plain
# capture_pybullet_camera() UNCHANGED (already robot-agnostic: takes only
# eye/target/up/physics_client_id, no Panda-specific assumption). No
# wrist camera is implemented here -- PyBulletWristCamera
# (robot_sim/pybullet_wrist_camera.py) is Panda-specific (hardcoded
# end_effector_link_index=11, offsets tuned for Panda's gripper geometry)
# and adapting it to SO-101's own EE link/geometry has not been
# validated, so it is deliberately NOT forced into existence here (see
# this task's own "wrist camera가 없으면 존재하지 않는 카메라를 억지로
# 만들지 않는다" instruction).
FRONT_CAMERA_WIDTH = 256
FRONT_CAMERA_HEIGHT = 256
FRONT_CAMERA_EYE = [0.9, -0.55, 0.55]
FRONT_CAMERA_TARGET = [0.391, 0.05, 0.08]  # roughly the support-surface/target-zone center
FRONT_CAMERA_UP = [0.0, 0.0, 1.0]
FRONT_CAMERA_FOV = 60.0
FRONT_CAMERA_NEAR = 0.1
FRONT_CAMERA_FAR = 3.0

# Identified BY NAME at load time in reset() -- never a hardcoded
# PyBullet joint index (see this task's requirement 3). This list is
# just which NAMES to look up, not an index assumption.
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
EE_LINK_NAME = "gripper_frame_link"

# Validated in benchmark/smoke_so101_joint_control.py: 0.0 on every arm
# joint converges cleanly and IS this URDF's own "new calibration"
# neutral (zero at the middle of each joint's range) -- not a Panda-style
# hand-picked home pose.
NEUTRAL_ARM_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0]

MOVE_FORCE = 10.0  # matches this URDF's own <limit effort="10" .../> on every actuated joint
DEFAULT_SETTLE_STEPS = 40  # chosen independently for SO-101 (not copied from Panda's DEFAULT_MOVE_STEPS=120) -- same order of magnitude validated in the smoke tests
IK_SOLVER_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5

# --- Scene metadata: support surface + object (see this task's chat
# report, "support surface 구조 정리") ---
# Generalized so a future caller can swap "table" for a different
# support surface (tray, shelf, ...) by overriding scene_config alone --
# nothing else in this file hardcodes a surface height in more than one
# place. surface_height is the ONE authoritative number; object spawn z
# and the EE minimum-height safety floor are both DERIVED from it (plus
# object_height / safety_margin), never independently hardcoded.
DEFAULT_SCENE_CONFIG = {
    "surface_height": 0.05,       # z (m) of the support surface's TOP face
    "surface_thickness": 0.05,    # box thickness for rendering/collision only -- does NOT affect surface_height
    "surface_footprint_xy": [0.15, 0.15],   # half-extents in x/y
    # Centered directly under the arm's own neutral EE position
    # ([0.391, 0, 0.226], see benchmark/smoke_so101_ik.py's own
    # measurement) so the default reach requires mostly a vertical
    # descent, not a large lateral move -- deliberate for the object-
    # approach/grasp smoke tests, not a workspace-coverage claim.
    "surface_center_xy": [0.391, 0.0],
    "object_height": 0.04,         # full object height (2x half-extent, box or cylinder)
    "object_footprint_xy": [0.02, 0.02],     # half-extents in x/y -- BOX shape only, ignored when object_shape="cylinder"
    # Additive, opt-in shape selector (see this task's chat report,
    # "Expert V2.1 cylinder 지원") -- default "box" reproduces EVERY
    # existing scene byte-for-byte (reset()'s object-creation branch
    # below only takes the cylinder path when this is explicitly
    # overridden to "cylinder"; no existing caller does that). "radius"
    # below is read ONLY on that opt-in path.
    "object_shape": "box",
    "object_radius": 0.02,        # half-extent-equivalent for object_shape="cylinder"; unused for "box"
    "safety_margin": 0.01,          # clearance added above surface_height for the EE minimum-height floor
    # Place target zone (see this task's chat report, "목표 표면 또는
    # target zone 추가") -- a PURELY VISUAL marker (no collision shape),
    # on the SAME support surface. Its center is computed in reset() as
    # object_position_xy + target_zone_offset_xy (never hardcoded to a
    # specific absolute coordinate); half-extent is comfortably larger
    # than the object's own footprint (0.02) so a well-placed object
    # visibly sits inside it.
    "target_zone_offset_xy": [0.05, 0.05],   # matches benchmark/smoke_so101_transport.py's own TRANSPORT_DELTA_XY convention
    "target_zone_half_extent": 0.05,
}

# --- Open-top bin V1 (see this task's chat report, "target marker를
# 실제 충돌이 있는 open-top bin으로 확장") --- Physical (collision-
# enabled) place container, built OPT-IN via So101PyBulletBackend's own
# `use_bin` constructor flag (default False) -- with it left at its
# default, reset() creates only the same purely-visual, collision-free
# target marker exactly as before, so every existing SO-101 smoke test/
# expert/benchmark run is byte-for-byte unaffected. This task's own
# scope is the bin's PHYSICAL STRUCTURE only -- scripted-expert
# waypoints, release height, and success judgment are untouched here.
BIN_INNER_WIDTH_M = 0.14
BIN_INNER_LENGTH_M = 0.14
BIN_WALL_HEIGHT_M = 0.08
BIN_WALL_THICKNESS_M = 0.004
BIN_BOTTOM_THICKNESS_M = 0.004

# Chosen to keep an object from unrealistically ice-skating (too low)
# or bouncing like rubber (too high/high restitution) inside the bin,
# without artificially inflating friction just to raise a future
# success rate (see this task's own "비현실적으로 큰 마찰값을 임의로
# 사용하지 말 것").
BIN_LATERAL_FRICTION = 0.7
BIN_ROLLING_FRICTION = 0.001
BIN_SPINNING_FRICTION = 0.001
BIN_RESTITUTION = 0.05

# Table/object visual colors -- named here (not left as inline literals
# at their _create_box() call sites) so a script measuring scene visual
# salience (see benchmark/measure_so101_bin_visual_salience.py) can
# import them instead of retyping the same numbers a second time.
TABLE_COLOR_RGBA = [0.55, 0.35, 0.2, 1.0]
OBJECT_COLOR_RGBA = [0.2, 0.6, 1.0, 1.0]

# Opaque, high-saturation magenta -- deliberately OUTSIDE this scene's
# existing blue/brown/gray palette (table is brown, object is blue,
# the default ground plane checker is pale gray/blue, the flat mode's
# own target marker is green), so the bin cannot be confused with any
# of them by hue alone. The OLD value (light blue, alpha=0.85, kept
# below commented out for the record) measured
# bin_object_rgb_distance~=61 and bin_background_rgb_distance~=212 on
# a 0-255*sqrt(3) scale (see
# benchmark/measure_so101_bin_visual_salience.py's own "before" run,
# results/so101_bin_visual_before_after/before/visibility_metrics.json)
# -- low specifically against the object because both were in the same
# blue hue family. This magenta was picked by computing straight-line
# RGB distance against all three existing surface colors at once
# (TABLE_COLOR_RGBA, OBJECT_COLOR_RGBA, and the pale background) and
# choosing a hue far from all of them, not just the one this task's
# report happened to flag first. alpha=1.0 (fully opaque) so the
# wall's rendered edge is crisp instead of blended with whatever is
# behind/inside it. Collision geometry (BIN_INNER_WIDTH_M/
# BIN_WALL_HEIGHT_M/BIN_WALL_THICKNESS_M above) is UNCHANGED -- only
# this rgbaColor value differs; see _create_box()'s own collision+
# visual shapes, both still built from the exact same half_extents.
# OLD_BIN_COLOR_RGBA_BEFORE_THIS_TASK = [0.55, 0.75, 0.95, 0.85]
BIN_COLOR_RGBA = [0.9, 0.1, 0.6, 1.0]

# --- Bin scene layout (see this task's chat report, "bin scene layout
# 문제") --- DEFAULT_SCENE_CONFIG's own target_zone_offset_xy=[0.05, 0.05]
# was sized for the OLD collision-free flat marker, where the object's
# spawn footprint being close to (or even under) the marker was
# harmless -- the marker has no collision shape. With a REAL
# collision-enabled bin (inner half-width 0.07m), that same offset
# puts the bin close enough to overlap the object's own spawn AABB (an
# object half-extent of 0.02m + bin inner half-width of 0.07m = 0.09m
# is the minimum single-axis clearance needed to avoid it -- two AABBs
# only overlap if EVERY axis overlaps, so clearing just one axis is
# enough).
#
# Empirically validated directly against this project's own SO-101
# IK/reach (see benchmark/smoke_so101_bin_place.py's own prior
# validation): a y-dominant split reaches with far better precision
# than an equal x/y split of the same total clearance -- (0.10, 0.10)
# left the bin-place waypoints ~0.03-0.06m short (occasionally even
# grazing a wall in transit), while (0.03, 0.10) converges to
# ~0.001m with zero robot-bin contact. Used ONLY when use_bin=True AND
# the caller did not explicitly set target_zone_offset_xy themselves
# (see So101PyBulletBackend.__init__'s own resolution order: explicit
# override > this bin default > DEFAULT_SCENE_CONFIG's flat default).
DEFAULT_BIN_TARGET_ZONE_OFFSET_XY = [0.03, 0.10]

# The DEFAULT table (surface_footprint_xy=[0.15, 0.15], object spawning
# at its exact center) is genuinely too small to fit this offset AND
# keep the bin's own outer footprint inside the table surface at the
# same time -- discovered by this task's own new table-bounds check
# (validate_initial_scene_layout()'s check C): clearing the
# object-bin overlap needs >= object_half(0.02) + bin_outer_half(0.074)
# = 0.094m of offset on at least one axis (a hard geometric fact for
# two axis-aligned boxes, not a tuning choice), but the DEFAULT table
# only allows up to table_half(0.15) - bin_outer_half(0.074) = 0.076m
# before the bin's own far edge runs off the table -- an 0.018m
# shortfall that no offset choice can close. Growing the table (bin
# mode only, same override-priority pattern as
# DEFAULT_BIN_TARGET_ZONE_OFFSET_XY) is the least invasive fix
# available within this task's own scope (bin size, expert waypoints,
# and object spawn position are all off-limits) -- 0.19m half-extent
# leaves a real ~0.016m margin at DEFAULT_BIN_TARGET_ZONE_OFFSET_XY.
DEFAULT_BIN_SURFACE_FOOTPRINT_XY = [0.19, 0.19]


class InvalidSceneLayoutError(RuntimeError):
    """Raised by So101PyBulletBackend.validate_initial_scene_layout()
    (see this task's chat report, "초기 scene layout validation") --
    carries the raw AABB/position data that triggered the failure so a
    caller doesn't need to re-derive it from a parsed message string."""

    def __init__(self, message: str, failure_type: str, details: dict):
        super().__init__(message)
        self.failure_type = failure_type
        self.details = details


def _aabb_from_pybullet(min_xyz: tuple, max_xyz: tuple) -> dict:
    return {"x_min": min_xyz[0], "x_max": max_xyz[0], "y_min": min_xyz[1], "y_max": max_xyz[1], "z_min": min_xyz[2], "z_max": max_xyz[2]}


def _aabb_boxes_overlap(a: dict, b: dict, epsilon: float = 1e-4) -> bool:
    """Real 3-axis AABB overlap (see this task's own "단순 중심 거리만
    사용하지 말고 실제 AABB overlap으로 확인") -- all three axes must
    overlap simultaneously; boxes merely touching (zero-gap, common for
    e.g. an object resting exactly on the table surface) are NOT
    considered overlapping. `epsilon` absorbs float-level noise in
    PyBullet's own reported AABBs without masking a real
    (multi-centimeter, in every observed failure case) overlap."""
    return (
        a["x_min"] < b["x_max"] - epsilon and a["x_max"] > b["x_min"] + epsilon
        and a["y_min"] < b["y_max"] - epsilon and a["y_max"] > b["y_min"] + epsilon
        and a["z_min"] < b["z_max"] - epsilon and a["z_max"] > b["z_min"] + epsilon
    )


def compute_bin_geometry(
    center_x: float, center_y: float, table_surface_z: float,
    inner_width: float = BIN_INNER_WIDTH_M, inner_length: float = BIN_INNER_LENGTH_M,
    wall_height: float = BIN_WALL_HEIGHT_M, wall_thickness: float = BIN_WALL_THICKNESS_M,
    bottom_thickness: float = BIN_BOTTOM_THICKNESS_M,
) -> dict:
    """Pure geometry computation (no PyBullet call, no side effect) -- kept
    separate from body creation (_create_open_top_bin()) so the
    dimensions/placement math is independently unit-testable (see this
    task's own "bin geometry 계산과 PyBullet body 생성을 가능한 한
    분리할 것").

    Box construction (see this task's chat report for the corner-
    overlap reasoning): the left/right walls span the FULL outer
    length (they own the four corners); the front/back walls span only
    the inner width and sit flush between the left/right walls -- no
    gap, no double-covered corner volume, and the bottom plate spans
    the full outer footprint under all four walls."""
    outer_width = inner_width + 2.0 * wall_thickness
    outer_length = inner_length + 2.0 * wall_thickness

    bottom_center_z = table_surface_z + bottom_thickness / 2.0
    bottom_top_z = table_surface_z + bottom_thickness
    wall_center_z = bottom_top_z + wall_height / 2.0
    rim_z = bottom_top_z + wall_height

    left_wall_x = center_x - inner_width / 2.0 - wall_thickness / 2.0
    right_wall_x = center_x + inner_width / 2.0 + wall_thickness / 2.0
    front_wall_y = center_y - inner_length / 2.0 - wall_thickness / 2.0
    back_wall_y = center_y + inner_length / 2.0 + wall_thickness / 2.0

    return {
        "center_x": center_x, "center_y": center_y,
        "inner_width": inner_width, "inner_length": inner_length,
        "outer_width": outer_width, "outer_length": outer_length,
        "wall_height": wall_height, "wall_thickness": wall_thickness, "bottom_thickness": bottom_thickness,
        "table_surface_z": table_surface_z,
        "bottom_center_z": bottom_center_z, "bottom_top_z": bottom_top_z,
        "wall_center_z": wall_center_z, "rim_z": rim_z,
        "inner_x_min": center_x - inner_width / 2.0, "inner_x_max": center_x + inner_width / 2.0,
        "inner_y_min": center_y - inner_length / 2.0, "inner_y_max": center_y + inner_length / 2.0,
        "outer_x_min": center_x - outer_width / 2.0, "outer_x_max": center_x + outer_width / 2.0,
        "outer_y_min": center_y - outer_length / 2.0, "outer_y_max": center_y + outer_length / 2.0,
        "bottom": {
            "half_extents": [outer_width / 2.0, outer_length / 2.0, bottom_thickness / 2.0],
            "position": [center_x, center_y, bottom_center_z],
        },
        "left_wall": {
            "half_extents": [wall_thickness / 2.0, outer_length / 2.0, wall_height / 2.0],
            "position": [left_wall_x, center_y, wall_center_z],
        },
        "right_wall": {
            "half_extents": [wall_thickness / 2.0, outer_length / 2.0, wall_height / 2.0],
            "position": [right_wall_x, center_y, wall_center_z],
        },
        "front_wall": {
            "half_extents": [inner_width / 2.0, wall_thickness / 2.0, wall_height / 2.0],
            "position": [center_x, front_wall_y, wall_center_z],
        },
        "back_wall": {
            "half_extents": [inner_width / 2.0, wall_thickness / 2.0, wall_height / 2.0],
            "position": [center_x, back_wall_y, wall_center_z],
        },
    }


def _surface_position(scene_config: dict) -> list:
    return [
        scene_config["surface_center_xy"][0], scene_config["surface_center_xy"][1],
        scene_config["surface_height"] - scene_config["surface_thickness"] / 2.0,
    ]


def _surface_half_extents(scene_config: dict) -> list:
    return [scene_config["surface_footprint_xy"][0], scene_config["surface_footprint_xy"][1], scene_config["surface_thickness"] / 2.0]


def _object_half_extents(scene_config: dict) -> list:
    return [scene_config["object_footprint_xy"][0], scene_config["object_footprint_xy"][1], scene_config["object_height"] / 2.0]


def _default_object_position(scene_config: dict) -> list:
    return [
        scene_config["surface_center_xy"][0], scene_config["surface_center_xy"][1],
        scene_config["surface_height"] + scene_config["object_height"] / 2.0,
    ]


def _min_ee_height(scene_config: dict) -> float:
    return scene_config["surface_height"] + scene_config["safety_margin"]


# Backward-compatible module-level constants, DERIVED from
# DEFAULT_SCENE_CONFIG (not a second hardcoded copy) -- kept because
# benchmark/smoke_so101_object_approach.py already imports these by name;
# an instance's actual geometry now comes from self.scene_config (see
# __init__), which these values exactly match by construction for the
# default scene (verified: table_top=0.05, object_z=0.07,
# min_ee_height=0.06 -- byte-identical to the pre-refactor hardcoded
# values, so "기존 기본 장면의 동작은 그대로 유지" holds numerically).
TABLE_POSITION = _surface_position(DEFAULT_SCENE_CONFIG)
TABLE_HALF_EXTENTS = _surface_half_extents(DEFAULT_SCENE_CONFIG)
TABLE_TOP_Z = DEFAULT_SCENE_CONFIG["surface_height"]
OBJECT_HALF_EXTENTS = _object_half_extents(DEFAULT_SCENE_CONFIG)
OBJECT_MASS = 0.05
DEFAULT_OBJECT_POSITION = _default_object_position(DEFAULT_SCENE_CONFIG)
OBJECT_SETTLE_STEPS = 60  # lets the object settle onto the table (tiny initial-contact wobble) before it's treated as "initial pose"
MIN_EE_HEIGHT_M = _min_ee_height(DEFAULT_SCENE_CONFIG)

# --- Grasp trigger thresholds (see this task's chat report, "grasp trigger") ---
GRASP_DISTANCE_THRESHOLD_M = 0.04   # EE-object distance must be within this to trigger a grasp -- looser than the object's own half-extent (0.02) but tighter than the pre-grasp offset (0.08), so pre-grasp cannot accidentally trigger a grasp
GRASP_GRIPPER_CLOSED_THRESHOLD = 0.15  # normalized gripper position (0=closed) must be at or below this -- generous margin above the ~0.015 typical settle value measured in smoke_so101_joint_control.py

# --- Release threshold (see this task's chat report, "release 동작") ---
GRIPPER_OPEN_RELEASE_THRESHOLD = 0.85  # normalized gripper position (1=open) must be at or above this to release an active grasp -- symmetric counterpart to GRASP_GRIPPER_CLOSED_THRESHOLD


def gripper_normalized_to_radians(value: float, lower: float, upper: float) -> float:
    """0.0=closed, 1.0=open (this task's external convention) -> this
    URDF's native revolute-joint radians. Kept as its own free function,
    independent of the class, specifically so it can be swapped for a
    real-hardware 0-100 command mapping later (see the vendored SO-101
    README's note that this URDF/MJCF pair does not yet reflect that
    hardware convention) without touching backend control-flow code."""
    value = max(0.0, min(1.0, value))
    return lower + value * (upper - lower)


def gripper_radians_to_normalized(radians: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return max(0.0, min(1.0, (radians - lower) / (upper - lower)))


class So101PyBulletBackend:
    def __init__(
        self, gui: bool = False, urdf_path=None, time_step: float = 1.0 / 240.0,
        object_position: Optional[list] = None, scene_config: Optional[dict] = None,
        use_bin: bool = False, show_target_marker: Optional[bool] = None,
        bin_center_override_xy: Optional[list] = None, object_yaw_rad: Optional[float] = None,
    ):
        self.gui = gui
        self.urdf_path = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
        self.time_step = time_step
        # None (default) -> DEFAULT_OBJECT_POSITION is used in reset();
        # a caller passing an explicit list here is the separation point
        # for future object-position randomization (see this task's
        # chat report) -- reset() itself never rolls its own randomness.
        self._object_position_override = list(object_position) if object_position is not None else None
        # None (default) -> reset() derives the bin center from THIS
        # episode's own (possibly randomized) object position + offset,
        # exactly as before this task ("coupled" mode -- see
        # DEFAULT_BIN_TARGET_ZONE_OFFSET_XY). An explicit [x, y] here
        # decouples the bin from wherever the object landed -- the bin
        # sits at this FIXED world position every episode regardless of
        # object_position (see this task's chat report, "새 독립
        # randomization" / fixed_bin_object_xy mode). Only meaningful
        # when use_bin=True; ignored otherwise.
        self._bin_center_override_xy = list(bin_center_override_xy) if bin_center_override_xy is not None else None
        # None (default) -> identity orientation (0 yaw), byte-identical
        # to every existing caller's behavior before this task. An
        # explicit radians value here is applied as a pure Z-axis
        # rotation at object spawn (see reset()'s own object creation) --
        # the grasp mechanism is a distance/gripper-state-triggered fixed
        # constraint (see _maybe_trigger_grasp()) that never reads
        # orientation, so this cannot by itself break grasp triggering.
        self._object_yaw_rad = float(object_yaw_rad) if object_yaw_rad is not None else 0.0
        # Shallow-merged with defaults so a caller can override e.g. just
        # surface_height without repeating every other key.
        user_scene_config = scene_config or {}
        self.scene_config = {**DEFAULT_SCENE_CONFIG, **user_scene_config}

        # Open-top bin V1 (see this task's chat report, "target marker를
        # 실제 충돌이 있는 open-top bin으로 확장") -- OPT-IN, default
        # False. With use_bin left at its default, reset() below takes
        # the exact same code path as before this task (visual-only
        # target marker, no bin bodies) -- every existing caller that
        # does not pass use_bin=True is unaffected.
        self.use_bin = use_bin

        # target_zone_offset_xy / surface_footprint_xy resolution
        # priority (see this task's chat report, "config 우선순위"):
        # explicit user scene_config > bin-specific default
        # (use_bin=True only) > flat default (DEFAULT_SCENE_CONFIG's
        # own values, UNCHANGED for use_bin=False regardless of this
        # block -- both checks below only ever fire when use_bin is
        # True). surface_footprint_xy also needs a bin-specific
        # default -- see DEFAULT_BIN_SURFACE_FOOTPRINT_XY's own
        # docstring for the geometric reason (the default table is too
        # small to fit the bin-safe offset AND keep the bin's own
        # outer footprint on the table at the same time).
        if "target_zone_offset_xy" not in user_scene_config and self.use_bin:
            self.scene_config["target_zone_offset_xy"] = list(DEFAULT_BIN_TARGET_ZONE_OFFSET_XY)
        if "surface_footprint_xy" not in user_scene_config and self.use_bin:
            self.scene_config["surface_footprint_xy"] = list(DEFAULT_BIN_SURFACE_FOOTPRINT_XY)

        self.min_ee_height_m = _min_ee_height(self.scene_config)
        # None (default) -> resolved in reset() to `not self.use_bin`
        # (marker shown when there's no bin to z-fight with; hidden by
        # default once a real bin exists -- see this task's own "bin
        # 사용 시 marker를 기본적으로 숨김"). An explicit True/False here
        # always overrides that default.
        self.show_target_marker = show_target_marker

        self.client_id = None
        self.robot_id = None
        self.joint_info_by_name = {}
        self.arm_joint_indices = []
        self.gripper_joint_index = None
        self.gripper_lower = None
        self.gripper_upper = None
        self.ee_link_index = None
        self.table_id = None
        self.object_id = None
        self.target_zone_id = None
        self.target_zone_center_xy = None
        self.bin_body_ids = None
        self.bin_geometry = None
        self._object_initial_pose = None
        self._layout_validation_result = None

        # Grasp state (see this task's chat report, "grasp 상태 추가") --
        # all reset to this same "nothing grasped" state in reset() too.
        self.grasp_constraint_id = None
        self._is_object_grasped = False
        self._grasped_object_id = None
        self._grasp_distance_at_trigger = None
        self._grasp_gripper_normalized_at_trigger = None

    def reset(self) -> dict:
        # Explicit constraint cleanup BEFORE tearing down the client (see
        # this task's "reset 시 기존 constraint를 제거" requirement) --
        # belt-and-suspenders on top of the fact that disconnecting the
        # physics client below already destroys every constraint/body in
        # it regardless, so this also stays correct if reset() is ever
        # changed to reuse a client instead of reconnecting.
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client_id)
            except p.error:
                pass
        self.grasp_constraint_id = None
        self._is_object_grasped = False
        self._grasped_object_id = None
        self._grasp_distance_at_trigger = None
        self._grasp_gripper_normalized_at_trigger = None

        # Also clear body ids from the PREVIOUS client -- without this,
        # the neutral-pose/gripper-open sequence below (which runs BEFORE
        # this episode's own table/object are recreated) would call
        # _maybe_trigger_grasp() with a stale self.object_id left over
        # from the last episode, which no longer exists in the freshly
        # (re)connected client below and crashes getBasePositionAndOrientation.
        self.table_id = None
        self.object_id = None
        self.target_zone_id = None
        self.target_zone_center_xy = None
        # Same reasoning as target_zone_id above -- these bodies live in
        # the PREVIOUS client too; the disconnect/reconnect below already
        # destroys them, this just keeps our own ID bookkeeping from
        # going stale (see this task's own "이전 body id가 남아 있으면
        # 정리해야 한다").
        self.bin_body_ids = None
        self.bin_geometry = None
        self._layout_validation_result = None

        if self.client_id is not None:
            p.disconnect(self.client_id)

        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        if not self.urdf_path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found at {self.urdf_path} -- see third_party/so101_arm/SOURCE.md")
        self.robot_id = p.loadURDF(str(self.urdf_path), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)

        self.joint_info_by_name = {}
        self.ee_link_index = None
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self.joint_info_by_name[name] = {"index": joint_index, "lower": info[8], "upper": info[9]}
            if link_name == EE_LINK_NAME:
                self.ee_link_index = joint_index

        missing = [n for n in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME] if n not in self.joint_info_by_name]
        if missing:
            raise RuntimeError(f"Expected joint name(s) not found in URDF {self.urdf_path}: {missing}")
        if self.ee_link_index is None:
            raise RuntimeError(f"EE link '{EE_LINK_NAME}' not found in URDF {self.urdf_path}")

        self.arm_joint_indices = [self.joint_info_by_name[n]["index"] for n in ARM_JOINT_NAMES]
        gripper_info = self.joint_info_by_name[GRIPPER_JOINT_NAME]
        self.gripper_joint_index = gripper_info["index"]
        self.gripper_lower = gripper_info["lower"]
        self.gripper_upper = gripper_info["upper"]

        # Neutral pose, then gripper fully open -- an explicit, well-
        # defined starting state (mirrors PyBulletPandaBackend.reset()'s
        # own "gripper starts open" convention, chosen independently here
        # for the same reason: an ambiguous mid-range resting angle would
        # make every downstream observation harder to interpret).
        self.command_joint_positions(NEUTRAL_ARM_POSITIONS, settle_steps=DEFAULT_SETTLE_STEPS * 3)
        self.set_gripper(1.0, settle_steps=DEFAULT_SETTLE_STEPS)

        # --- Scene: support surface (fixed) + single object -- built
        # AFTER the arm is at its known neutral pose, so there is no risk
        # of the object being spawned inside/overlapping a not-yet-settled
        # arm pose. Every geometric quantity here reads self.scene_config
        # (surface_height/object_height/safety_margin), never a bare
        # number -- see this task's chat report, "support surface 구조
        # 정리". self.min_ee_height_m is (re)computed here too, so a
        # scene_config override actually takes effect on the safety floor
        # command_end_effector_delta() enforces, not just on the visuals.
        self.min_ee_height_m = _min_ee_height(self.scene_config)
        surface_position = _surface_position(self.scene_config)
        surface_half_extents = _surface_half_extents(self.scene_config)
        object_half_extents = _object_half_extents(self.scene_config)

        self.table_id = self._create_box(surface_half_extents, surface_position, color=TABLE_COLOR_RGBA, mass=0.0)
        object_position = self._object_position_override if self._object_position_override is not None else _default_object_position(self.scene_config)
        object_orientation = p.getQuaternionFromEuler([0.0, 0.0, self._object_yaw_rad])
        # object_shape branch (see this task's chat report, "Expert V2.1
        # cylinder 지원") -- "box" (the default, and every scene_config
        # used before this task) takes the EXACT SAME _create_box() call
        # as before, byte-for-byte. Only an explicit "cylinder" override
        # takes the new path.
        if self.scene_config.get("object_shape", "box") == "cylinder":
            self.object_id = self._create_cylinder(
                self.scene_config["object_radius"], self.scene_config["object_height"] / 2.0,
                object_position, color=OBJECT_COLOR_RGBA, mass=OBJECT_MASS, orientation=object_orientation,
            )
        else:
            self.object_id = self._create_box(
                object_half_extents, object_position, color=OBJECT_COLOR_RGBA, mass=OBJECT_MASS, orientation=object_orientation,
            )
        self.step(OBJECT_SETTLE_STEPS)  # let any initial-contact wobble settle before recording "initial pose"
        self._object_initial_pose = self.get_object_pose()

        # Place target zone -- PURELY VISUAL (no collision shape), center
        # computed from THIS episode's own (possibly overridden) object
        # position, never a hardcoded absolute coordinate (see this
        # task's chat report). Uses the object's SETTLED position (after
        # OBJECT_SETTLE_STEPS above), not its pre-settle spawn position.
        settled_object_position = self._object_initial_pose[0]
        offset_xy = self.scene_config["target_zone_offset_xy"]
        if self._bin_center_override_xy is not None:
            # Fixed-bin mode (see this task's chat report,
            # "fixed_bin_object_xy") -- the bin sits at this fixed world
            # position regardless of where THIS episode's object landed.
            # scene_config["target_zone_offset_xy"] is RE-DERIVED from
            # the actual bin center and actual settled object position
            # ONLY in this branch -- it no longer means "the bin follows
            # the object by this fixed amount" once overridden, so
            # get_scene_state()/callers deriving a transport delta from
            # it (e.g. benchmark/collect_so101_bin_dataset.py) still see
            # the correct effective offset for THIS episode (see this
            # task's chat report, "object-bin 상대 offset이 사실상
            # 상수" limitation this mode exists to fix). Left OUT of
            # the coupled (else) branch below on purpose: recomputing it
            # there via subtraction is a mathematical no-op but NOT a
            # floating-point no-op (introduces ~1e-17 noise), which
            # broke smoke_so101_bin_scene_layout.py's exact-value
            # backward-compatibility checks the first time this was
            # tried.
            self.target_zone_center_xy = list(self._bin_center_override_xy)
            self.scene_config["target_zone_offset_xy"] = [
                self.target_zone_center_xy[0] - settled_object_position[0],
                self.target_zone_center_xy[1] - settled_object_position[1],
            ]
        else:
            self.target_zone_center_xy = [settled_object_position[0] + offset_xy[0], settled_object_position[1] + offset_xy[1]]

        # Open-top bin V1 -- built at the SAME center the flat marker
        # already uses (see this task's own "기존 target marker 중심
        # 좌표를 그대로 bin 내부 중심으로 사용"), only when opted in.
        if self.use_bin:
            self.bin_geometry = compute_bin_geometry(
                self.target_zone_center_xy[0], self.target_zone_center_xy[1], self.scene_config["surface_height"],
            )
            self.bin_body_ids = self._create_open_top_bin(self.bin_geometry)

        # Marker z sits just above whatever surface it would otherwise
        # z-fight with -- the bin's own bottom top face when a bin
        # exists, the bare table surface when it doesn't (unchanged
        # from before this task).
        marker_should_show = self.show_target_marker if self.show_target_marker is not None else not self.use_bin
        if marker_should_show:
            target_zone_half_extent = self.scene_config["target_zone_half_extent"]
            marker_z = (self.bin_geometry["bottom_top_z"] if self.bin_geometry else self.scene_config["surface_height"]) + 0.001
            target_zone_position = [self.target_zone_center_xy[0], self.target_zone_center_xy[1], marker_z]
            self.target_zone_id = self._create_visual_marker(
                [target_zone_half_extent, target_zone_half_extent, 0.001], target_zone_position, color=[0.1, 0.9, 0.1, 0.35],
            )

        # Validate BEFORE any expert phase runs, at 0 additional physics
        # steps past this point (see this task's chat report, "validation
        # 실행 시점") -- ONLY when use_bin=True, so use_bin=False's
        # behavior is unchanged in every case, not just the default
        # scene_config (see this task's own "기존 flat-target 기본 동작
        # 변경 금지"). Raises InvalidSceneLayoutError immediately on a
        # bad scene rather than silently proceeding into an episode that
        # would fail downstream for an opaque reason.
        if self.use_bin:
            self.validate_initial_scene_layout()
        else:
            self._layout_validation_result = None

        return self.get_observation()

    def _create_box(self, half_extents: list, position: list, color: list, mass: float = 0.0, orientation: Optional[list] = None) -> int:
        # None (default) -> identity quaternion via PyBullet's own
        # createMultiBody default -- every existing call site (table,
        # bin walls) omits this and is unaffected. Only the object's own
        # call site (see reset()) ever passes a non-identity value (see
        # this task's chat report, "object yaw randomization").
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id)
        kwargs = {"baseOrientation": orientation} if orientation is not None else {}
        return p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
            basePosition=position, physicsClientId=self.client_id, **kwargs,
        )

    def _create_cylinder(self, radius: float, half_height: float, position: list, color: list, mass: float = 0.0, orientation: Optional[list] = None) -> int:
        """Additive counterpart to _create_box() (see this task's chat
        report, "Expert V2.1 cylinder 지원") -- same structure (collision
        + visual, createMultiBody), GEOM_CYLINDER instead of GEOM_BOX.
        PyBullet's cylinder is upright (local Z) by construction, matching
        this task's own "upright cylinder만 사용" scope -- object_yaw_rad
        still applies via `orientation` exactly as it does for the box
        path, though it has no visible effect on an upright, rotationally-
        symmetric cylinder (used only for the yaw-invariance check this
        task's own Expert evaluation performs)."""
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=2.0 * half_height, physicsClientId=self.client_id)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=2.0 * half_height, rgbaColor=color, physicsClientId=self.client_id)
        kwargs = {"baseOrientation": orientation} if orientation is not None else {}
        return p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
            basePosition=position, physicsClientId=self.client_id, **kwargs,
        )

    def _create_visual_marker(self, half_extents: list, position: list, color: list) -> int:
        """Visual-only (no collision shape, mass=0) -- used for the place
        target zone, which must never physically interact with the arm
        or object (see this task's chat report, 'metadata 기반 target
        region이면 충분하다')."""
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id)
        return p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual_shape,
            basePosition=position, physicsClientId=self.client_id,
        )

    def _create_open_top_bin(self, geometry: dict) -> dict:
        """5 static bodies (bottom + 4 walls), each with BOTH a
        collision and a visual shape, baseMass=0.0 -- an actual
        open-top container, not a single solid box (see this task's
        chat report, "속이 찬 하나의 육면체로 만들면 안 된다"). Reuses
        the existing _create_box() helper (already: collision+visual+
        static) for each part, then applies BIN_* friction/restitution
        via p.changeDynamics() explicitly on every body (not left at
        PyBullet's own defaults)."""
        body_ids = {}
        for part_name in ("bottom", "left_wall", "right_wall", "front_wall", "back_wall"):
            part = geometry[part_name]
            body_id = self._create_box(part["half_extents"], part["position"], color=BIN_COLOR_RGBA, mass=0.0)
            p.changeDynamics(
                body_id, -1, lateralFriction=BIN_LATERAL_FRICTION, rollingFriction=BIN_ROLLING_FRICTION,
                spinningFriction=BIN_SPINNING_FRICTION, restitution=BIN_RESTITUTION, physicsClientId=self.client_id,
            )
            body_ids[part_name] = body_id
        body_ids["all"] = list(body_ids.values())
        return body_ids

    def get_bin_debug_info(self) -> Optional[dict]:
        """None when use_bin=False (nothing to report) -- otherwise the
        full geometry dict (center/inner/outer bounds/z-levels), body
        ids, and the ACTUAL dynamics values applied (not just the
        module-level constants -- see this task's own "PyBullet에 실제로
        적용된 값" requirement), read back via p.getDynamicsInfo() so
        this reports what PyBullet itself has, not merely what was
        requested."""
        if not self.use_bin or self.bin_geometry is None or self.bin_body_ids is None:
            return None

        applied_dynamics = {}
        for part_name in ("bottom", "left_wall", "right_wall", "front_wall", "back_wall"):
            info = p.getDynamicsInfo(self.bin_body_ids[part_name], -1, physicsClientId=self.client_id)
            applied_dynamics[part_name] = {
                "lateral_friction": info[1], "restitution": info[5],
                "rolling_friction": info[6], "spinning_friction": info[7],
            }

        return {**self.bin_geometry, "body_ids": {k: v for k, v in self.bin_body_ids.items() if k != "all"}, "applied_dynamics": applied_dynamics}

    def _table_surface_bounds_xy(self) -> dict:
        center_x, center_y = self.scene_config["surface_center_xy"]
        half_x, half_y = self.scene_config["surface_footprint_xy"]
        return {"x_min": center_x - half_x, "x_max": center_x + half_x, "y_min": center_y - half_y, "y_max": center_y + half_y}

    def validate_initial_scene_layout(self) -> dict:
        """Initial-scene sanity check (see this task's chat report,
        "초기 scene layout validation") -- called from reset() itself
        ONLY when use_bin=True (see reset()'s own call site), so
        use_bin=False behavior is completely untouched regardless of
        whether a custom scene_config would pass or fail these checks.
        Safely callable directly at any time otherwise (e.g. by a test,
        for a use_bin=False scene too -- the bin-specific checks (A, C)
        just report nothing to check in that case).

        Uses PyBullet's own LIVE p.getAABB() query for the object and
        every bin body -- reflects the actual simulated geometry at
        the moment this is called, not a hand-recomputed one from
        scene_config's stated intentions.

        Checks (see this task's chat report, section 3):
          A. object AABB does not overlap the bin bottom or any of the
             4 wall AABBs (real 3-axis overlap, not center-distance)
          B. object AABB is within the table's usable surface bounds
             (full AABB, not just its center)
          C. bin outer AABB is within the table's usable surface bounds
          D. object rests on the table surface within a small physics
             tolerance (not floating, not sunk through)
          E. no unexpected initial contact -- object-bin, object-robot
             (object-table contact is expected and allowed)

        Raises InvalidSceneLayoutError immediately on ANY failure --
        never returns a "failed" result silently. On success, returns
        a dict describing what was checked (also cached onto
        self._layout_validation_result and surfaced via
        get_scene_state())."""
        object_aabb_min, object_aabb_max = p.getAABB(self.object_id, physicsClientId=self.client_id)
        object_aabb = _aabb_from_pybullet(object_aabb_min, object_aabb_max)
        table_bounds = self._table_surface_bounds_xy()
        failures = []

        # A: object vs each bin body (bottom + 4 walls)
        if self.use_bin and self.bin_body_ids:
            for part_name in ("bottom", "left_wall", "right_wall", "front_wall", "back_wall"):
                part_min, part_max = p.getAABB(self.bin_body_ids[part_name], physicsClientId=self.client_id)
                part_aabb = _aabb_from_pybullet(part_min, part_max)
                if _aabb_boxes_overlap(object_aabb, part_aabb):
                    failures.append({
                        "failure_type": "object_bin_overlap", "wall_name": part_name,
                        "object_aabb": object_aabb, "part_aabb": part_aabb,
                    })

        # B: object AABB within table surface bounds (full AABB, not center)
        if not (
            table_bounds["x_min"] <= object_aabb["x_min"] and object_aabb["x_max"] <= table_bounds["x_max"]
            and table_bounds["y_min"] <= object_aabb["y_min"] and object_aabb["y_max"] <= table_bounds["y_max"]
        ):
            failures.append({"failure_type": "object_outside_table_bounds", "object_aabb": object_aabb, "table_bounds": table_bounds})

        # C: bin outer AABB within table surface bounds
        if self.use_bin and self.bin_geometry:
            bin_outer = {
                "x_min": self.bin_geometry["outer_x_min"], "x_max": self.bin_geometry["outer_x_max"],
                "y_min": self.bin_geometry["outer_y_min"], "y_max": self.bin_geometry["outer_y_max"],
            }
            if not (
                table_bounds["x_min"] <= bin_outer["x_min"] and bin_outer["x_max"] <= table_bounds["x_max"]
                and table_bounds["y_min"] <= bin_outer["y_min"] and bin_outer["y_max"] <= table_bounds["y_max"]
            ):
                failures.append({"failure_type": "bin_outside_table_bounds", "bin_outer_bounds": bin_outer, "table_bounds": table_bounds})

        # D: object resting on the table surface (small physics tolerance)
        surface_height = self.scene_config["surface_height"]
        resting_tolerance_m = 0.01  # same order of magnitude as this project's own RESTING_HEIGHT_ERROR_PASS_M
        if abs(object_aabb["z_min"] - surface_height) > resting_tolerance_m:
            failures.append({
                "failure_type": "object_not_on_table_surface", "object_bottom_z": object_aabb["z_min"],
                "surface_height": surface_height, "tolerance_m": resting_tolerance_m,
            })

        # E: no unexpected initial contact (object-table is fine/expected, not checked here)
        if self.use_bin and self.bin_body_ids:
            contacted_bin_parts = [
                part_name for part_name in ("bottom", "left_wall", "right_wall", "front_wall", "back_wall")
                if p.getContactPoints(bodyA=self.object_id, bodyB=self.bin_body_ids[part_name], physicsClientId=self.client_id)
            ]
            if contacted_bin_parts:
                failures.append({"failure_type": "unexpected_object_bin_contact", "contacted_parts": contacted_bin_parts})

        if p.getContactPoints(bodyA=self.object_id, bodyB=self.robot_id, physicsClientId=self.client_id):
            failures.append({"failure_type": "unexpected_object_robot_contact"})

        result = {"passed": len(failures) == 0, "failures": failures, "object_aabb": object_aabb, "table_bounds": table_bounds}
        self._layout_validation_result = result

        if failures:
            primary = failures[0]
            message = (
                f"InvalidSceneLayoutError: {primary['failure_type']}\n"
                f"object_aabb={object_aabb}\n"
                f"target_zone_offset_xy={self.scene_config['target_zone_offset_xy']}\n"
                f"object_spawn_xyz={self._object_initial_pose[0] if self._object_initial_pose else None}\n"
                f"bin_center_xyz={(self.target_zone_center_xy[0], self.target_zone_center_xy[1]) if self.use_bin and self.target_zone_center_xy else None}\n"
                f"table_bounds={table_bounds}\n"
                f"all_failures={failures}"
            )
            raise InvalidSceneLayoutError(message, primary["failure_type"], result)

        return result

    def _get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.client_id)
        return list(state[4]), list(state[5])

    def get_joint_positions(self) -> list:
        states = p.getJointStates(self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id)
        return [s[0] for s in states]

    def get_end_effector_pose(self) -> tuple:
        return self._get_ee_pose()

    def command_joint_positions(self, positions: list, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        if len(positions) != len(self.arm_joint_indices):
            raise ValueError(f"Expected {len(self.arm_joint_indices)} arm joint positions, got {len(positions)}")

        clipped = []
        for name, raw_position in zip(ARM_JOINT_NAMES, positions):
            if not math.isfinite(raw_position):
                raise ValueError(f"Non-finite joint position commanded for '{name}': {raw_position}")
            info = self.joint_info_by_name[name]
            clipped.append(max(info["lower"], min(info["upper"], raw_position)))

        p.setJointMotorControlArray(
            self.robot_id, self.arm_joint_indices, p.POSITION_CONTROL, targetPositions=clipped,
            forces=[MOVE_FORCE] * len(self.arm_joint_indices), physicsClientId=self.client_id,
        )
        self.step(settle_steps)
        return self.get_observation()

    def compute_joint_target_from_ee_delta(self, delta_position: list, delta_orientation=None) -> dict:
        """Pure computation, no side effect on the simulation -- runs the
        SAME IK call + safety floor + joint-limit clip that
        command_end_effector_delta() used to run inline (see this task's
        chat report, "IK 단일 계산 구조"). Split out so a caller (e.g. the
        shared Expert in benchmark/so101_scripted_expert.py) can compute
        the exact absolute joint target ONCE, record it, and apply that
        SAME target via apply_joint_target() below -- no second IK call.

        delta_orientation is accepted for interface symmetry with a
        future full-6DOF version, but is currently ignored, not silently
        misapplied (unchanged from command_end_effector_delta()'s prior
        own docstring note)."""
        current_ee_position, _current_ee_orientation = self._get_ee_pose()
        target_position = [current_ee_position[i] + delta_position[i] for i in range(3)]
        if not all(math.isfinite(v) for v in target_position):
            raise ValueError(f"Non-finite EE target computed from delta {delta_position}: {target_position}")

        # Safety floor: never command the EE below the table top + margin,
        # regardless of which delta produced this target (see this task's
        # "테이블 아래로 내려가지 않도록 최소 높이 제한" requirement) --
        # holds for every caller, not just the approach-smoke-test's own
        # step loop.
        target_position[2] = max(target_position[2], self.min_ee_height_m)

        joint_poses = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, target_position,
            maxNumIterations=IK_SOLVER_ITERATIONS, residualThreshold=IK_RESIDUAL_THRESHOLD,
            physicsClientId=self.client_id,
        )
        raw_arm_targets = list(joint_poses[: len(self.arm_joint_indices)])

        # Clipped HERE (not left to apply_joint_target()/command_joint_positions()
        # alone) so the value returned to the caller is EXACTLY the value
        # that will physically be applied -- a caller recording this as a
        # dataset action must never record a pre-clip value.
        # command_joint_positions() still clips again when this is
        # applied; re-clipping an already-clipped value is a no-op, so
        # this changes nothing about what actually gets commanded.
        clipped_arm_targets = []
        for name, raw_position in zip(ARM_JOINT_NAMES, raw_arm_targets):
            info = self.joint_info_by_name[name]
            clipped_arm_targets.append(max(info["lower"], min(info["upper"], raw_position)))

        return {"target_position": target_position, "arm_joint_targets": clipped_arm_targets}

    def apply_joint_target(self, arm_joint_targets: list, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        """Applies an already-computed absolute arm joint target (e.g.
        from compute_joint_target_from_ee_delta() above) -- a thin,
        purely-additive wrapper around the existing
        command_joint_positions() (same clip + finite-check + step +
        observation return), kept as its own method so a caller can
        apply the EXACT SAME target it already recorded, without
        recomputing IK."""
        return self.command_joint_positions(arm_joint_targets, settle_steps=settle_steps)

    def command_end_effector_delta(self, delta_position: list, delta_orientation=None, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        """Position-only IK (see docs/so101_backend_design_proposal.md).
        Now a thin wrapper over compute_joint_target_from_ee_delta() +
        apply_joint_target() (see this task's chat report, "IK 단일 계산
        구조") -- still exactly ONE IK call per invocation, same as
        before this refactor; behavior/return value are unchanged."""
        computed = self.compute_joint_target_from_ee_delta(delta_position, delta_orientation)
        observation = self.apply_joint_target(computed["arm_joint_targets"], settle_steps=settle_steps)

        final_ee_position, _ = self._get_ee_pose()
        position_error = math.sqrt(sum((final_ee_position[i] - computed["target_position"][i]) ** 2 for i in range(3)))
        observation["ee_delta_target_position"] = computed["target_position"]
        observation["ee_delta_position_error"] = position_error
        return observation

    def set_gripper(self, normalized_value: float, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        if not math.isfinite(normalized_value):
            raise ValueError(f"Non-finite gripper command: {normalized_value}")
        radians = gripper_normalized_to_radians(normalized_value, self.gripper_lower, self.gripper_upper)
        p.setJointMotorControlArray(
            self.robot_id, [self.gripper_joint_index], p.POSITION_CONTROL, targetPositions=[radians],
            forces=[MOVE_FORCE], physicsClientId=self.client_id,
        )
        self.step(settle_steps)
        self._maybe_trigger_grasp()
        self._maybe_release_grasp()
        return self.get_observation()

    def _maybe_release_grasp(self) -> None:
        """Checked after every set_gripper() call, alongside
        _maybe_trigger_grasp() (see this task's chat report, 'release
        동작') -- removes the active grasp constraint ONLY when
        currently grasping AND the gripper's REAL measured position is
        open at or above GRIPPER_OPEN_RELEASE_THRESHOLD. A no-op (not an
        error) if nothing is currently grasped -- opening the gripper
        when there was never a grasp, or reset()'s own gripper-open call
        before any object exists, must never crash or misbehave."""
        if not self._is_object_grasped:
            return

        gripper_state = p.getJointState(self.robot_id, self.gripper_joint_index, physicsClientId=self.client_id)
        gripper_normalized = gripper_radians_to_normalized(gripper_state[0], self.gripper_lower, self.gripper_upper)
        if gripper_normalized < GRIPPER_OPEN_RELEASE_THRESHOLD:
            return

        if self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client_id)
        self.grasp_constraint_id = None
        self._is_object_grasped = False
        self._grasped_object_id = None
        self._grasp_distance_at_trigger = None
        self._grasp_gripper_normalized_at_trigger = None

    def _maybe_trigger_grasp(self) -> None:
        """Checked after every set_gripper() call (see this task's chat
        report, 'grasp trigger') -- creates a PyBullet fixed constraint
        attaching the object to the EE link ONLY when all four
        conditions hold simultaneously: not already grasping, a valid
        object id, EE-object distance within GRASP_DISTANCE_THRESHOLD_M,
        and the gripper's REAL measured (not commanded) position closed
        at or below GRASP_GRIPPER_CLOSED_THRESHOLD. Any one condition
        failing is a silent no-op, not an error -- calling set_gripper()
        far from the object, or with the gripper open, is normal use,
        not a failure case."""
        if self._is_object_grasped:
            return
        if self.object_id is None:
            return

        ee_position, ee_orientation = self._get_ee_pose()
        object_position, object_orientation = self.get_object_pose()
        distance = math.sqrt(sum((ee_position[i] - object_position[i]) ** 2 for i in range(3)))

        gripper_state = p.getJointState(self.robot_id, self.gripper_joint_index, physicsClientId=self.client_id)
        gripper_normalized = gripper_radians_to_normalized(gripper_state[0], self.gripper_lower, self.gripper_upper)

        if distance > GRASP_DISTANCE_THRESHOLD_M:
            return
        if gripper_normalized > GRASP_GRIPPER_CLOSED_THRESHOLD:
            return

        # Attach at the object's CURRENT offset from the EE link (not
        # [0,0,0]), so creating the constraint doesn't snap/teleport the
        # object -- same reasoning PyBulletPandaBackend.close_gripper()
        # already documents for its own grasp constraint, applied here
        # independently (this file does not call or import that method).
        ee_pos_inv, ee_orn_inv = p.invertTransform(ee_position, ee_orientation)
        frame_position, frame_orientation = p.multiplyTransforms(ee_pos_inv, ee_orn_inv, object_position, object_orientation)

        self.grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id, parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.object_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=frame_position, parentFrameOrientation=frame_orientation,
            childFramePosition=[0, 0, 0], childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        self._is_object_grasped = True
        self._grasped_object_id = self.object_id
        self._grasp_distance_at_trigger = distance
        self._grasp_gripper_normalized_at_trigger = gripper_normalized

    def is_grasped(self) -> bool:
        return self._is_object_grasped

    def get_grasp_state(self) -> dict:
        return {
            "is_grasped": self._is_object_grasped,
            "grasp_constraint_id": self.grasp_constraint_id,
            "grasped_object_id": self._grasped_object_id,
            "grasp_distance_at_trigger": self._grasp_distance_at_trigger,
            "grasp_gripper_normalized_at_trigger": self._grasp_gripper_normalized_at_trigger,
        }

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_observation(self) -> dict:
        arm_positions = self.get_joint_positions()
        ee_position, ee_orientation = self._get_ee_pose()
        gripper_state = p.getJointState(self.robot_id, self.gripper_joint_index, physicsClientId=self.client_id)
        gripper_radians = gripper_state[0]
        gripper_normalized = gripper_radians_to_normalized(gripper_radians, self.gripper_lower, self.gripper_upper)

        return {
            "simulator": "pybullet_so101",
            "joint_positions": arm_positions,
            "end_effector_position": ee_position,
            "end_effector_orientation": ee_orientation,
            "gripper_position_normalized": gripper_normalized,
            "gripper_position_radians": gripper_radians,
        }

    def get_object_position(self) -> list:
        position, _orientation = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return list(position)

    def get_object_pose(self) -> tuple:
        position, orientation = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return list(position), list(orientation)

    def get_object_velocity(self) -> tuple:
        """(linear[3], angular[3]) m/s and rad/s -- used to judge whether
        a released object has settled (see this task's chat report)."""
        linear, angular = p.getBaseVelocity(self.object_id, physicsClientId=self.client_id)
        return list(linear), list(angular)

    def get_scene_state(self) -> dict:
        """Superset of get_observation() -- adds object pose, static
        table metadata, and the place target zone, all finite-checkable
        by the caller the same way get_observation()'s own fields
        already are."""
        observation = self.get_observation()
        object_position, object_orientation = self.get_object_pose()
        return {
            **observation,
            "object_position": object_position,
            "object_orientation": object_orientation,
            "table_position": list(TABLE_POSITION),
            "table_half_extents": list(TABLE_HALF_EXTENTS),
            "table_top_z": TABLE_TOP_Z,
            "target_zone_center_xy": list(self.target_zone_center_xy) if self.target_zone_center_xy else None,
            "target_zone_half_extent": self.scene_config["target_zone_half_extent"],
            "bin": self.get_bin_debug_info(),
            # --- Scene-layout metadata (see this task's chat report,
            # "scene metadata 보강") -- pure additions, no existing key
            # above is touched. ---
            "use_bin": self.use_bin,
            "target_zone_offset_xy": list(self.scene_config["target_zone_offset_xy"]),
            "object_spawn_position": list(self._object_initial_pose[0]) if self._object_initial_pose else None,
            "bin_center": list(self.target_zone_center_xy) if (self.use_bin and self.target_zone_center_xy) else None,
            "table_surface_bounds": self._table_surface_bounds_xy(),
            "object_aabb_initial": self._layout_validation_result["object_aabb"] if self._layout_validation_result else None,
            "bin_outer_bounds": (
                {"x_min": self.bin_geometry["outer_x_min"], "x_max": self.bin_geometry["outer_x_max"],
                 "y_min": self.bin_geometry["outer_y_min"], "y_max": self.bin_geometry["outer_y_max"]}
                if self.bin_geometry else None
            ),
            "layout_validation_passed": self._layout_validation_result["passed"] if self._layout_validation_result else None,
            "layout_validation_failures": self._layout_validation_result["failures"] if self._layout_validation_result else None,
        }

    def render_front_camera(self, width: int = FRONT_CAMERA_WIDTH, height: int = FRONT_CAMERA_HEIGHT):
        """Fixed, world-space camera (see module-level FRONT_CAMERA_* constants)
        -- never depends on the robot's current configuration. Returns an
        (H, W, 3) uint8 RGB array."""
        return capture_pybullet_camera(
            width=width, height=height, camera_eye=FRONT_CAMERA_EYE, camera_target=FRONT_CAMERA_TARGET,
            camera_up=FRONT_CAMERA_UP, fov=FRONT_CAMERA_FOV, near_val=FRONT_CAMERA_NEAR, far_val=FRONT_CAMERA_FAR,
            physics_client_id=self.client_id,
        )

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
