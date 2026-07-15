"""PolicyManifest -- what a checkpoint claims about its own input/output
semantics, registered once per model_id so CompatibilityGate (see
compatibility_gate.py) can check it before production use.

UNKNOWN is a first-class value here, not a missing-data bug: a field
being UNKNOWN means this project has not verified that detail yet, and
CompatibilityGate treats "claims a real value" and "honestly UNKNOWN"
very differently from "silently wrong" -- see compatibility_gate.py's
policy of refusing production whenever any semantically-relevant field
is UNKNOWN, per this task's explicit requirement not to guess.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

UNKNOWN = "UNKNOWN"


class ActionSpace(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_DELTA = "joint_delta"
    EE_DELTA = "ee_delta"
    EE_ABSOLUTE = "ee_absolute"
    UNKNOWN = "UNKNOWN"


@dataclass
class PolicyManifest:
    model_id: str
    revision: str  # UNKNOWN if not pinned to a specific verified snapshot
    source_embodiment: str  # e.g. "SO-100/SO-101", "LIBERO Franka Panda (robosuite)"

    required_camera_roles: List[str]  # e.g. ["main", "wrist"]
    state_fields: Dict[str, int]  # field name (or "UNKNOWN") -> dimension
    action_dimension: int

    action_space: ActionSpace
    relative_or_absolute: str  # "relative" | "absolute" | UNKNOWN
    rotation_representation: str  # "axis_angle" | "euler" | "quaternion" | UNKNOWN
    reference_frame: str  # "robot_base" | "world" | UNKNOWN

    gripper_included: bool
    gripper_index: Optional[int]  # index within the action vector, or None if not included
    gripper_convention: str  # e.g. "UNKNOWN -- open/close direction not verified"

    action_chunk_size: int

    normalization: str  # e.g. "MEAN_STD (dataset stats)" or "[-1, 1] clipped, unknown physical scale"
    official_processor_available: bool
    official_processor_wired: bool  # whether this project's loader actually calls it (see model_loader.py)

    # Whether this checkpoint's source-embodiment base-frame axis
    # convention has been independently confirmed to match
    # PANDA_TARGET_EMBODIMENT's (e.g. by a physical/simulated movement
    # test), as opposed to "both claim robot_base frame" being merely
    # asserted. False by default -- deliberately conservative, since two
    # simulators modeling the "same" robot can still disagree on which
    # way +X/+Y/+Z point without a direct cross-check.
    axis_convention_verified: bool = False

    notes: str = ""


@dataclass
class BackendCapabilities:
    supports_cartesian_translation: bool
    supports_cartesian_rotation: bool
    supports_gripper: bool
    rotation_representation: str
    reference_frame: str


# Mirrors robot_sim/pybullet_panda_backend.py's BACKEND_CAPABILITIES /
# PyBulletPandaBackend.get_capabilities() exactly -- kept as a plain-data
# constant here (never imported from robot_sim) so policy_semantics
# never gains a hard dependency on pybullet being installed:
# vla_server/model_loader.py imports this module on the VLA-serving side,
# which must keep working without pybullet installed (see
# docs/panda_axis_cross_verification.md for why pybullet/robosuite/mujoco
# were only added to .venv-vla for the one-off verification script, not
# as a standing dependency of the server). If
# robot_sim/pybullet_panda_backend.py's real capabilities ever change,
# update both together.
PANDA_BACKEND_CAPABILITIES = BackendCapabilities(
    supports_cartesian_translation=True,
    supports_cartesian_rotation=True,
    supports_gripper=True,
    rotation_representation="axis_angle",
    reference_frame="robot_base",
)


# This project's own robot -- see docs/07_pybullet_panda_backend.md /
# robot_sim/pybullet_panda_backend.py. CompatibilityGate compares every
# manifest against this, not a hardcoded per-checkpoint special case.
PANDA_TARGET_EMBODIMENT = PolicyManifest(
    model_id="physical-ai-recycling-cell/pybullet_panda_v0",
    revision="n/a",
    source_embodiment="Franka Panda (PyBullet simulation, 7-DOF arm + 2-finger gripper via IK to grasp target)",
    required_camera_roles=["wrist"],
    state_fields={"end_effector_position": 3, "end_effector_orientation": 4, "gripper_width": 1},
    action_dimension=7,
    action_space=ActionSpace.EE_DELTA,
    relative_or_absolute="relative",
    rotation_representation="axis_angle",
    reference_frame="robot_base",
    gripper_included=True,
    gripper_index=6,
    gripper_convention="1.0 = close, 0.0 = open (existing wire format threshold 0.5; see canonical_command.py "
    "for CanonicalRobotCommand's opposite gripper_opening polarity and the explicit conversion)",
    action_chunk_size=1,
    normalization="none (raw meters/radians, clipped by action_postprocess.max_translation_step/max_rotation_step)",
    official_processor_available=False,
    official_processor_wired=False,
    axis_convention_verified=True,  # this manifest IS the target; trivially "verified" against itself
    notes="Target embodiment this project's control loop actually drives -- every other manifest is checked against this one.",
)


_SMOLVLA_BASE_MANIFEST = PolicyManifest(
    model_id="lerobot/smolvla_base",
    # Snapshot actually verified against in this project (see
    # docs/vla_integration_spike_log.md / smolvla_cloud_loading_spike.md).
    revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    source_embodiment="SO-100/SO-101 (6-servo teleoperated arm)",
    required_camera_roles=["camera1", "camera2", "camera3"],
    state_fields={"observation.state": 6},
    action_dimension=6,
    action_space=ActionSpace.JOINT_POSITION,
    relative_or_absolute=UNKNOWN,  # not needed to reach the INCOMPATIBLE verdict -- see notes
    rotation_representation="n/a (joint-space, not Cartesian rotation)",
    reference_frame="n/a (joint-space)",
    gripper_included=True,
    gripper_index=5,
    gripper_convention="SO-100 gripper joint position (servo command), not an open/close fraction",
    action_chunk_size=50,
    normalization="MEAN_STD (LeRobot dataset stats, per SmolVLAConfig.normalization_mapping)",
    official_processor_available=True,
    official_processor_wired=False,
    notes=(
        "Confirmed via policy.config.input_features/output_features on the real loaded checkpoint: "
        "action shape=(6,), matching SO-100/SO-101's 6-servo joint-space convention "
        "(shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper), NOT "
        "[dx,dy,dz,droll,dpitch,dyaw,gripper]. Structurally INCOMPATIBLE with the Panda "
        "Cartesian-delta target -- no unit/scale conversion bridges a different robot's "
        "joint space to this project's end-effector space; that would need real forward/"
        "inverse kinematics for SO-100's specific arm, which this project does not have "
        "and is out of scope. The 6D-length match to a 7D-minus-gripper shape was coincidental."
    ),
)

_SMOLVLA_LIBERO_MANIFEST = PolicyManifest(
    model_id="HuggingFaceVLA/smolvla_libero",
    # Snapshot actually downloaded/inspected this session via hf_hub_download.
    revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    source_embodiment="LIBERO Franka Panda (robosuite/MuJoCo simulation, OSC_POSE controller, 20Hz)",
    required_camera_roles=["main", "wrist"],
    state_fields={"ee_position": 3, "ee_orientation_axis_angle": 3, "gripper_qpos": 2},
    action_dimension=7,
    action_space=ActionSpace.EE_DELTA,
    # Confirmed: lerobot/envs/libero.py's LiberoEnv.reset() sets
    # robot.controller.use_delta = True for control_mode="relative".
    relative_or_absolute="relative",
    # Confirmed: robosuite/controllers/parts/arm/osc.py,
    # OperationalSpaceController.compute_goal_ori()'s docstring: "delta
    # ... in axis-angle form [ax, ay, az]".
    rotation_representation="axis_angle",
    # Confirmed: robosuite/controllers/parts/arm/osc.py,
    # OperationalSpaceController.__init__'s docstring: input_ref_frame
    # "base": actions are wrt the robot body. LIBERO's env uses this
    # controller unmodified (see policy_semantics/adapters/
    # smolvla_libero_adapter.py's module docstring for exact citations).
    reference_frame="robot_base",
    gripper_included=True,
    gripper_index=6,
    # Confirmed: robosuite/models/grippers/panda_gripper.py,
    # PandaGripper.format_action() docstring, verbatim: "-1 => open,
    # 1 => closed". Deliberately NOT required to equal
    # PANDA_TARGET_EMBODIMENT.gripper_convention's string (opposite
    # polarity, 1.0=close/0.0=open) -- see compatibility_gate.py's
    # gripper_convention check: a *known* (not UNKNOWN) convention with a
    # verified conversion (policy_semantics/adapters/
    # smolvla_libero_adapter.py's SmolVLALiberoActionAdapter.decode())
    # is sufficient; the two sides don't need to share one polarity.
    gripper_convention="robosuite PandaGripper.format_action(): -1.0 = open, 1.0 = closed",
    action_chunk_size=50,
    # Confirmed via this checkpoint's real config.json (downloaded this
    # session): normalization_mapping.ACTION = "MEAN_STD". Its shipped
    # policy_postprocessor_step_1_unnormalizer_processor.safetensors
    # holds the actual per-dimension mean/std used to recover values in
    # lerobot/envs/libero.py's native Box(-1, 1, shape=(7,)) action
    # space -- see smolvla_libero_adapter.py for the confirmed physical
    # scale (0.05 m / 0.5 rad per unit) applied on top of that.
    normalization="MEAN_STD (dataset stats baked into the checkpoint's own shipped unnormalizer)",
    official_processor_available=True,
    official_processor_wired=True,  # vla_server/model_loader.py now loads and calls it -- see _load_smolvla_libero_processors()
    # Confirmed via benchmark/verify_panda_axis_convention.py: (1) a
    # +X/+Y/+Z-only delta action applied independently in both robosuite
    # (Panda + OSC_POSE) and PyBulletPandaBackend produces a
    # displacement dominated by (and same-signed as) the intended axis
    # in both simulators; (2) setting both simulators to identical
    # READY_JOINT_POSITIONS gives the same EE-position-relative-to-base
    # (within 2cm) via pure forward kinematics, no controller involved.
    # Both are real simulation results, not a config-file read -- see
    # docs/panda_axis_cross_verification.md/.json for the full numbers
    # and explicit tolerances.
    axis_convention_verified=True,
    notes=(
        "Same robot family (Franka Panda) and same action convention (6D EE delta + 1D "
        "gripper, robot_base frame, axis-angle rotation) as this project's PyBullet target "
        "-- confirmed this session directly from robosuite/LIBERO/LeRobot official source "
        "(see policy_semantics/adapters/smolvla_libero_adapter.py's module docstring for "
        "exact file/function citations), not inferred from watching the model's raw output. "
        "axis_convention_verified is now True (see docs/panda_axis_cross_verification.md) and "
        "PyBulletPandaBackend now actually applies rotation deltas (see "
        "robot_sim/pybullet_panda_backend.py v2) -- CompatibilityGate's remaining checks are "
        "now the only gate on this manifest; see compatibility_gate.py's backend_capabilities "
        "check for the rotation-support requirement this depends on."
    ),
)

MANIFEST_REGISTRY: Dict[str, PolicyManifest] = {
    _SMOLVLA_BASE_MANIFEST.model_id: _SMOLVLA_BASE_MANIFEST,
    _SMOLVLA_LIBERO_MANIFEST.model_id: _SMOLVLA_LIBERO_MANIFEST,
}


def get_manifest(model_id: str) -> PolicyManifest:
    """Never raises. Returns the registered manifest for model_id, or an
    all-UNKNOWN manifest if it isn't registered -- CompatibilityGate then
    refuses production for it the same way it would for a registered but
    unverified checkpoint, rather than the caller needing a separate
    not-found code path."""
    manifest = MANIFEST_REGISTRY.get(model_id)
    if manifest is not None:
        return manifest

    return PolicyManifest(
        model_id=model_id,
        revision=UNKNOWN,
        source_embodiment=UNKNOWN,
        required_camera_roles=[],
        state_fields={},
        action_dimension=-1,
        action_space=ActionSpace.UNKNOWN,
        relative_or_absolute=UNKNOWN,
        rotation_representation=UNKNOWN,
        reference_frame=UNKNOWN,
        gripper_included=False,
        gripper_index=None,
        gripper_convention=UNKNOWN,
        action_chunk_size=-1,
        normalization=UNKNOWN,
        official_processor_available=False,
        official_processor_wired=False,
        notes=f"No PolicyManifest registered for model_id={model_id!r} -- treated as fully unverified.",
    )
