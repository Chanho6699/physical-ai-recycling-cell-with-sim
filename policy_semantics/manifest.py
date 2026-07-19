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

import json
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This project's OWN single data-collection pipeline
# (benchmark/collect_recycling_dataset.py -> DummyOpenVLAPolicy) always
# clamps its translation delta to this exact, cited bound before saving
# it as the training action -- see policy/dummy_openvla_policy.py.
# Imported here (not re-typed as a bare literal) so this stays a single
# source of truth if that constant ever changes. Both policy.* modules
# this pulls in are lightweight (dataclasses/typing only, no torch/
# pybullet), consistent with this module's existing no-heavy-deps
# convention (see PANDA_BACKEND_CAPABILITIES's docstring below).
from policy.dummy_openvla_policy import DEFAULT_MAX_STEP_SIZE as _OWN_PIPELINE_MAX_STEP_SIZE_M

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

    # Native (raw, post-official-postprocessor, pre-CanonicalRobotCommand)
    # gripper value range this checkpoint's own action head actually
    # outputs -- e.g. LIBERO/robosuite's (-1.0, 1.0), or this project's
    # own collected-dataset convention (0.0, 1.0). Distinct from
    # gripper_convention (a free-text description) because
    # SmolVLALiberoActionAdapter.decode() (see
    # policy_semantics/adapters/smolvla_libero_adapter.py) needs these as
    # actual numbers to convert into CanonicalRobotCommand.gripper_opening_01
    # (1.0=open, 0.0=closed) -- see this task's chat report for the
    # confirmed bug this replaces: a single hardcoded (-1, 1) formula
    # applied uniformly regardless of which native scale a given
    # checkpoint's own postprocessor actually produces. None/UNKNOWN means
    # "not verified" -- CompatibilityGate's gripper_native_range_known
    # check refuses production for this checkpoint rather than guessing.
    native_gripper_range: Optional[Tuple[float, float]] = None  # (native_min, native_max)
    native_gripper_min_means: str = UNKNOWN  # "open" | "close" | UNKNOWN
    native_gripper_max_means: str = UNKNOWN  # "open" | "close" | UNKNOWN

    # Native (raw, post-official-postprocessor) translation/rotation
    # value -> physical meters/radians conversion this checkpoint's own
    # action head actually needs. LIBERO/robosuite's native action space
    # is Box(-1, 1); a native value of 1.0 there means
    # native_translation_scale_m meters (0.05) / native_rotation_scale_rad
    # radians (0.5) -- see policy_semantics/adapters/smolvla_libero_adapter.py's
    # module docstring for the robosuite controller citations. A
    # checkpoint whose OWN training data was already recorded in real
    # physical units (this project's own collect_recycling_dataset.py
    # pipeline: DummyOpenVLAPolicy's real-meter EE deltas, passed through
    # action_adapter.adapter_v0.ActionAdapter.convert() with
    # position_scale=1.0/rotation_scale=1.0 -- an identity pass-through,
    # confirmed via source and via this checkpoint's own postprocessor
    # stats living in [-0.03, 0.03], not LIBERO's [-0.9375, 0.9375]; see
    # this task's chat report) needs scale=1.0 (its postprocessed output
    # already IS the physical value) and no native clip range.
    # None/UNKNOWN means "not verified" -- CompatibilityGate's
    # translation_rotation_scale_known check refuses production for this
    # checkpoint rather than guessing, exactly like native_gripper_range
    # above. native_action_clip_range defaults to (-inf, inf) -- an
    # explicit, verified "no native-space clip applies" (distinct from
    # UNKNOWN: this project's own downstream PandaCommandSafetyFilter
    # still bounds the final command regardless, see safety_filter.py),
    # not a placeholder for missing information.
    native_translation_scale_m: Optional[float] = None
    native_rotation_scale_rad: Optional[float] = None
    native_action_clip_range: Tuple[float, float] = (float("-inf"), float("inf"))

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
    native_gripper_range=(0.0, 1.0),  # this project's own legacy wire format (RobotCommand's own gripper_command threshold)
    native_gripper_min_means="open",
    native_gripper_max_means="close",
    native_translation_scale_m=1.0,  # this project's own RobotCommand.target_dx/dy/dz are already real meters
    native_rotation_scale_rad=1.0,  # already real radians
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
    # Confirmed via this checkpoint's own real dataset stats (LIBERO's
    # meta/stats.json, downloaded this session): action[6] min=-1.0,
    # max=1.0 exactly -- NOT derived from this checkpoint's own
    # postprocessor safetensors, which only stores mean/std (pure
    # MEAN_STD normalization keeps no min/max) -- see this task's chat
    # report for why the numeric range has to come from an external,
    # independently-verified source for this specific checkpoint.
    native_gripper_range=(-1.0, 1.0),
    native_gripper_min_means="open",
    native_gripper_max_means="close",
    # robosuite/controllers/config/robots/default_panda.json (OSC_POSE,
    # Panda): output_max=[0.05,0.05,0.05,0.5,0.5,0.5], output_min the
    # negation -- confirmed via direct file read this session (see
    # policy_semantics/adapters/smolvla_libero_adapter.py's module
    # docstring). This checkpoint's native action space is Box(-1, 1);
    # a native value of 1.0 there means 0.05m (translation) / 0.5rad
    # (rotation) of physical motion.
    native_translation_scale_m=0.05,
    native_rotation_scale_rad=0.5,
    native_action_clip_range=(-1.0, 1.0),
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


def _read_postprocessor_action_min_max(path: Path) -> Optional[Tuple[list, list]]:
    """Reads a LOCAL checkpoint's own policy_postprocessor.json to find
    its unnormalizer_processor step's state_file (a structural lookup by
    registry_name, not a hardcoded filename -- LeRobot numbers these
    steps, e.g. policy_postprocessor_step_1_unnormalizer_processor.safetensors,
    and the exact number isn't guaranteed stable), then reads that
    safetensors file's full action.min/action.max tensors (all
    dimensions). Shared by _extract_gripper_native_range_from_postprocessor()
    (dim 6) and _verify_translation_scale_matches_own_pipeline() (dims 0:3).

    Returns None (never raises) if: the postprocessor has no such step,
    or that step's saved state doesn't include min/max at all (confirmed
    this session: a pure MEAN_STD-only unnormalizer -- e.g. the real
    HuggingFaceVLA/smolvla_libero base checkpoint's own shipped
    postprocessor -- never stores min/max, only mean/std; this project's
    own LOCALLY fine-tuned checkpoints do store the full stat set
    regardless of normalization mode, per LeRobot 0.6.0's
    make_pre_post_processors(), but that isn't guaranteed for every
    possible checkpoint). This is the checkpoint-intrinsic numeric
    source of truth -- it doesn't require the original training dataset
    directory to still exist on disk (train_config.json's dataset.root
    might have been moved/deleted since training)."""
    postprocessor_config_path = path / "policy_postprocessor.json"
    if not postprocessor_config_path.is_file():
        return None
    try:
        config = json.loads(postprocessor_config_path.read_text(encoding="utf-8"))
        state_file = None
        for step in config.get("steps", []):
            if step.get("registry_name") == "unnormalizer_processor":
                state_file = step.get("state_file")
                break
        if not state_file:
            return None

        from safetensors import safe_open

        with safe_open(str(path / state_file), framework="pt") as tensor_file:
            keys = set(tensor_file.keys())
            if "action.min" not in keys or "action.max" not in keys:
                return None
            action_min = tensor_file.get_tensor("action.min").tolist()
            action_max = tensor_file.get_tensor("action.max").tolist()
        return action_min, action_max
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def _extract_gripper_native_range_from_postprocessor(path: Path, gripper_index: int) -> Optional[Tuple[float, float]]:
    """gripper_index slice of _read_postprocessor_action_min_max() --
    see that function's docstring. Returns None if unavailable or the
    extracted range is degenerate (min >= max)."""
    min_max = _read_postprocessor_action_min_max(path)
    if min_max is None:
        return None
    action_min, action_max = min_max
    if gripper_index >= len(action_min) or gripper_index >= len(action_max):
        return None
    native_min = float(action_min[gripper_index])
    native_max = float(action_max[gripper_index])
    if not (native_min < native_max):
        return None
    return native_min, native_max


_OWN_PIPELINE_SCALE_TOLERANCE = 0.5  # +/-50% band around the known clamp bound -- generous enough for a
# handful of un-clamped samples near the tail of a short scripted episode, tight enough to reject
# a genuinely different native scale (e.g. LIBERO's ~0.9375, ~30x larger).


def _verify_translation_scale_matches_own_pipeline(path: Path) -> bool:
    """Corroborates -- does not blindly assume -- that this LOCAL
    checkpoint's training data was collected via this project's own
    pipeline (whose action[0:3] is already real meters, never a
    LIBERO-style normalized native value) by checking whether this
    checkpoint's OWN postprocessor action.max[0:3] is close to the
    known, cited DEFAULT_MAX_STEP_SIZE bound that pipeline always
    clamps to. Returns False (never guesses True) if the postprocessor
    has no min/max stats, or if the observed max is not close to that
    bound -- e.g. a checkpoint fine-tuned on genuinely LIBERO-native-
    scale data would correctly fail this and fall back to UNKNOWN rather
    than silently inheriting the wrong scale."""
    min_max = _read_postprocessor_action_min_max(path)
    if min_max is None:
        return False
    _action_min, action_max = min_max
    if len(action_max) < 3:
        return False
    for value in action_max[0:3]:
        if abs(abs(float(value)) - _OWN_PIPELINE_MAX_STEP_SIZE_M) > _OWN_PIPELINE_MAX_STEP_SIZE_M * _OWN_PIPELINE_SCALE_TOLERANCE:
            return False
    return True


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Generous bound for realistic fine-tune ancestry chains (a handful of
# --resume continuations at most) -- exists only to turn a pathological
# misconfiguration (or a genuine cycle that somehow slipped past cycle
# detection) into a clear, bounded failure instead of an unbounded walk.
_MAX_ANCESTRY_DEPTH = 8


class _AncestryResolutionError(Exception):
    """Raised internally while walking a local checkpoint's
    policy.pretrained_path chain back to a registered base manifest --
    always caught by _resolve_finetuned_manifest(), which converts it
    into a (None, reason) result. Never escapes get_manifest() (which
    must never raise); it exists purely to carry a specific, diagnosable
    failure reason (which path, which step) out of the ancestry walk."""


def _normalize_checkpoint_path(path_str: str) -> Path:
    """Resolves path_str the same way every other benchmark/*.py script
    in this project already resolves a possibly-relative checkpoint
    path: relative to this project's own root (not whatever the current
    process's cwd happens to be), then fully resolved (symlinks/`..`
    collapsed) so two different spellings of the same directory compare
    equal for cycle detection. This project only ever runs on Linux/WSL
    (see docs -- no Windows deployment target), so no separate backslash/
    drive-letter handling is needed; Path.resolve() already normalizes
    POSIX path spelling differences (`./`, redundant slashes, symlinks)."""
    path = Path(path_str)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _resolve_base_manifest_via_ancestry(model_id: str) -> Tuple[PolicyManifest, List[str]]:
    """Follows a LOCAL checkpoint's policy.pretrained_path (as written by
    lerobot_train.py into its saved train_config.json) through however
    many local fine-tune/--resume hops it takes to reach an
    already-registered MANIFEST_REGISTRY entry (e.g. checkpoint 4000 ->
    checkpoint 2000 -> "HuggingFaceVLA/smolvla_libero") -- rather than
    only following a single hop, which is all a plain --policy.path
    fine-tune ever needs but which a --resume continuation's own saved
    config breaks (lerobot_train.py records policy.pretrained_path as
    the checkpoint it actually resumed from, a local path, not the
    original Hub base -- confirmed this session against a real resumed
    checkpoint's train_config.json).

    Never hardcodes a checkpoint number, directory name, or specific
    chain depth -- it just keeps reading each train_config.json's own
    policy.pretrained_path until that value is either a registered
    model_id or another local directory to keep following.

    Raises _AncestryResolutionError (never returns a partial/guessed
    result) on: a cycle (a path revisited within this walk), a missing
    train_config.json, an unparseable/missing policy.pretrained_path, or
    exceeding _MAX_ANCESTRY_DEPTH -- each message names the exact path
    and step index that failed, plus the chain walked so far, so a
    caller can diagnose exactly where lineage resolution broke down
    instead of silently falling back to an unverified guess.

    Returns (registered_base_manifest, full_chain_as_strings) on
    success, where full_chain_as_strings is every path/model_id visited,
    ending in the resolved registered model_id.
    """
    visited: List[Path] = []
    current = _normalize_checkpoint_path(model_id)

    for depth in range(_MAX_ANCESTRY_DEPTH + 1):
        if current in visited:
            raise _AncestryResolutionError(
                f"cycle detected in checkpoint ancestry chain at step {depth}: {current} was "
                f"already visited (chain so far: {[str(p) for p in visited]})"
            )
        visited.append(current)

        if not current.is_dir():
            raise _AncestryResolutionError(
                f"ancestry step {depth}: {current!r} is neither a registered base model id nor "
                f"an existing local checkpoint directory (chain: {[str(p) for p in visited]})"
            )

        train_config_path = current / "train_config.json"
        if not train_config_path.is_file():
            raise _AncestryResolutionError(
                f"ancestry step {depth}: {train_config_path} does not exist "
                f"(chain: {[str(p) for p in visited]})"
            )

        try:
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            parent_id = train_config["policy"]["pretrained_path"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise _AncestryResolutionError(
                f"ancestry step {depth}: failed to read policy.pretrained_path from "
                f"{train_config_path}: {exc!r} (chain: {[str(p) for p in visited]})"
            ) from exc

        if not isinstance(parent_id, str) or not parent_id:
            raise _AncestryResolutionError(
                f"ancestry step {depth}: {train_config_path}'s policy.pretrained_path is missing "
                f"or not a non-empty string ({parent_id!r}) (chain: {[str(p) for p in visited]})"
            )

        registered = MANIFEST_REGISTRY.get(parent_id)
        if registered is not None:
            return registered, [str(p) for p in visited] + [parent_id]

        current = _normalize_checkpoint_path(parent_id)

    raise _AncestryResolutionError(
        f"checkpoint ancestry chain exceeded max depth ({_MAX_ANCESTRY_DEPTH}) without reaching "
        f"a registered base model id (chain: {[str(p) for p in visited]})"
    )


def _resolve_finetuned_manifest(model_id: str) -> Tuple[Optional[PolicyManifest], Optional[str]]:
    """Derives a manifest for a LOCAL checkpoint that this project
    fine-tuned itself, by reading which base checkpoint it was
    fine-tuned FROM -- rather than requiring a new hardcoded model_id
    entry (or a path-shaped heuristic) per fine-tuning run. Follows
    however many local --resume hops it takes (see
    _resolve_base_manifest_via_ancestry()) to reach an already-registered
    manifest -- a direct, single-hop --policy.path fine-tune (chain
    length 1) and a multi-generation --resume chain (length N) are
    handled by the exact same walk, not two separate code paths.

    Fine-tuning/resuming only changes weights, never observation/action
    shape, gripper convention, coordinate frame, or action scaling
    (LeRobot's `--policy.path`/`--resume` load carries the full base
    PreTrainedConfig forward unchanged unless a CLI override is given,
    and none of ours touch those fields -- see lerobot/configs/train.py's
    _resolve_pretrained_from_cli()), so inheriting the RESOLVED
    registered base manifest's semantic-shape claims is safe at any chain
    depth; only `revision` and `axis_convention_verified`-supporting
    provenance are NOT claimed to be independently re-verified for this
    specific local checkpoint.

    Numeric semantics (native gripper range, native translation/rotation
    scale) are NEVER read from an intermediate parent in the chain --
    only from THIS checkpoint's (model_id's) own postprocessor files,
    exactly as before ancestry-following existed; a checkpoint that
    happens to share ancestry with another does not necessarily share
    its exact numeric normalizer stats (e.g. re-fine-tuning could in
    principle shift them), so each checkpoint is re-verified independently.

    Returns (None, None) if model_id isn't an existing local directory
    with a train_config.json at all (the same "not applicable" case
    get_manifest() has always fallen through on). Returns (None,
    failure_reason) if it IS a local fine-tuned checkpoint but ancestry
    resolution failed (cycle/missing config/exceeded depth/etc) --
    get_manifest() surfaces failure_reason in the resulting all-UNKNOWN
    manifest's notes rather than silently guessing. Never raises.
    """
    path = _normalize_checkpoint_path(model_id)
    train_config_path = path / "train_config.json"
    if not path.is_dir() or not train_config_path.is_file():
        return None, None

    try:
        base_manifest, ancestry_chain = _resolve_base_manifest_via_ancestry(model_id)
    except _AncestryResolutionError as exc:
        return None, str(exc)

    has_processor_files = (path / "policy_preprocessor.json").is_file() and (
        path / "policy_postprocessor.json"
    ).is_file()

    # Numeric native gripper range is extracted FRESH from this exact
    # checkpoint's own postprocessor file -- never inherited from the
    # base manifest's numbers, since fine-tuning on this project's own
    # dataset genuinely changes them (confirmed this session:
    # HuggingFaceVLA/smolvla_libero's native range is (-1.0, 1.0);
    # outputs/train/smolvla_recycling_smoke_v0/.../pretrained_model's own
    # postprocessor reports (0.0, 1.0) -- a real, checkpoint-specific
    # difference a blind inheritance would have silently gotten wrong).
    native_gripper_range = None
    if base_manifest.gripper_index is not None:
        native_gripper_range = _extract_gripper_native_range_from_postprocessor(path, base_manifest.gripper_index)

    # Polarity DIRECTION (which end means open vs. close), by contrast,
    # genuinely is safe to inherit -- normalizer stats alone can never
    # reveal semantic polarity (a raw value of 1.0 could mean "open" in
    # one dataset and "close" in another), so it cannot be re-derived
    # from this checkpoint's files at all. It's inherited here ONLY
    # because this project has exactly one data-collection pipeline
    # (benchmark/collect_recycling_dataset.py -> action_adapter.adapter_v0
    # .ActionAdapter's fixed threshold: raw gripper >= 0.5 means "close"),
    # which places "open" at the LOW end and "close" at the HIGH end --
    # the same DIRECTION the base LIBERO manifest already declares, even
    # though the absolute numbers differ. Fine-tuning (adjusting weights)
    # cannot change this direction. Left UNKNOWN (not inherited) whenever
    # the numeric range itself couldn't be verified above, so a checkpoint
    # this function can't fully characterize never silently claims a
    # polarity it has no evidence for.
    if native_gripper_range is not None:
        native_gripper_min_means = base_manifest.native_gripper_min_means
        native_gripper_max_means = base_manifest.native_gripper_max_means
    else:
        native_gripper_min_means = UNKNOWN
        native_gripper_max_means = UNKNOWN

    # Translation/rotation native scale: same "cannot be derived from
    # stats alone, but a structural pipeline fact is safe to use"
    # reasoning as gripper polarity above -- except here it's actively
    # CORROBORATED (not just assumed) via
    # _verify_translation_scale_matches_own_pipeline()'s numeric check
    # against DEFAULT_MAX_STEP_SIZE, so a checkpoint fine-tuned on
    # data from some OTHER (e.g. still LIBERO-native-scale) source
    # correctly does NOT inherit this project's own identity scale.
    own_pipeline_scale_verified = _verify_translation_scale_matches_own_pipeline(path)
    if own_pipeline_scale_verified:
        native_translation_scale_m = PANDA_TARGET_EMBODIMENT.native_translation_scale_m
        native_rotation_scale_rad = PANDA_TARGET_EMBODIMENT.native_rotation_scale_rad
        native_action_clip_range = PANDA_TARGET_EMBODIMENT.native_action_clip_range
    else:
        native_translation_scale_m = None
        native_rotation_scale_rad = None
        native_action_clip_range = (float("-inf"), float("inf"))

    base_model_id = ancestry_chain[-1]
    manifest = replace(
        base_manifest,
        model_id=model_id,
        revision=UNKNOWN,
        official_processor_available=has_processor_files,
        official_processor_wired=has_processor_files,
        native_gripper_range=native_gripper_range,
        native_gripper_min_means=native_gripper_min_means,
        native_gripper_max_means=native_gripper_max_means,
        native_translation_scale_m=native_translation_scale_m,
        native_rotation_scale_rad=native_rotation_scale_rad,
        native_action_clip_range=native_action_clip_range,
        notes=(
            f"Locally fine-tuned from {base_model_id!r} via ancestry chain {ancestry_chain} "
            f"(each hop's own train_config.json: policy.pretrained_path, followed until a "
            f"registered base model id was reached -- see _resolve_base_manifest_via_ancestry()). "
            f"Observation/action shape, gripper convention direction, coordinate frame, and "
            f"axis-convention claims are inherited from {base_model_id!r}'s manifest unchanged -- "
            f"fine-tuning/resuming does not alter any of those, only the weights. "
            f"native_gripper_range is NOT inherited from any ancestor -- it is read fresh from "
            f"THIS checkpoint's own postprocessor safetensors "
            f"({'found: ' + str(native_gripper_range) if native_gripper_range else 'NOT FOUND -- gripper_native_range_known will fail CompatibilityGate'}). "
            f"native_translation_scale_m/native_rotation_scale_rad "
            f"({'verified against DEFAULT_MAX_STEP_SIZE using this checkpoint own stats: identity scale, no native clip' if own_pipeline_scale_verified else 'NOT verified against this checkpoint own stats -- translation_rotation_scale_known will fail CompatibilityGate'}). "
            f"This exact checkpoint's weights/behavior have NOT been independently re-verified "
            f"beyond that inheritance."
        ),
    )
    return manifest, None


def _unknown_manifest_field_defaults(model_id: str) -> dict:
    """The same all-UNKNOWN field values get_manifest()'s final fallback
    has always used -- factored out so _load_explicit_local_manifest()
    can start from identical defaults for whichever fields a
    policy_manifest.json doesn't declare."""
    return {
        "model_id": model_id,
        "revision": UNKNOWN,
        "source_embodiment": UNKNOWN,
        "required_camera_roles": [],
        "state_fields": {},
        "action_dimension": -1,
        "action_space": ActionSpace.UNKNOWN,
        "relative_or_absolute": UNKNOWN,
        "rotation_representation": UNKNOWN,
        "reference_frame": UNKNOWN,
        "gripper_included": False,
        "gripper_index": None,
        "gripper_convention": UNKNOWN,
        "action_chunk_size": -1,
        "normalization": UNKNOWN,
        "official_processor_available": False,
        "official_processor_wired": False,
    }


_EXPLICIT_MANIFEST_FILENAME = "policy_manifest.json"
_KNOWN_MANIFEST_FIELD_NAMES = {f.name for f in fields(PolicyManifest)}


def _normalize_declared_manifest_fields(declared: dict) -> dict:
    """Converts JSON-friendly shapes (action_space as a plain string,
    range fields as 2-element lists) into what PolicyManifest's
    dataclass fields actually expect, and drops any key that isn't one
    of PolicyManifest's own field names (a typo'd or forward-looking key
    in a hand-written policy_manifest.json should never raise or, worse,
    silently pass an unrelated kwarg into the dataclass constructor)."""
    normalized = {k: v for k, v in declared.items() if k in _KNOWN_MANIFEST_FIELD_NAMES}
    if isinstance(normalized.get("action_space"), str):
        try:
            normalized["action_space"] = ActionSpace(normalized["action_space"])
        except ValueError:
            normalized["action_space"] = ActionSpace.UNKNOWN
    for range_field in ("native_gripper_range", "native_action_clip_range"):
        if normalized.get(range_field) is not None:
            normalized[range_field] = tuple(normalized[range_field])
    return normalized


def _read_explicit_manifest_json(model_id: str) -> Optional[dict]:
    """Reads model_id/policy_manifest.json (a NEW, project-defined
    sidecar -- not part of LeRobot's own checkpoint schema) if present,
    returning its raw declared fields (only the ones the file actually
    specifies), or None if the file doesn't exist or fails to parse.
    This is how a checkpoint -- SmolVLA-lineage or not (ACT, Diffusion
    Policy, any custom policy) -- can ship its OWN semantics directly,
    with zero dependency on train_config.json/policy.pretrained_path or
    any registered base manifest at all. Never raises."""
    path = _normalize_checkpoint_path(model_id)
    manifest_path = path / _EXPLICIT_MANIFEST_FILENAME
    if not path.is_dir() or not manifest_path.is_file():
        return None
    try:
        declared = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(declared, dict):
        return None
    return _normalize_declared_manifest_fields(declared)


def _build_manifest_from_declared_fields(model_id: str, declared_fields: dict) -> PolicyManifest:
    """A full PolicyManifest built from declared_fields (see
    _read_explicit_manifest_json()) layered over the same all-UNKNOWN
    defaults the final hard-fail fallback already uses for anything
    NOT declared -- so a policy_manifest.json can be partial (declare
    only what it actually knows) without ever needing to spell out
    every field, and get_manifest() can tell "genuinely UNKNOWN" apart
    from "this file just didn't mention it" the same way everywhere
    else in this module already does."""
    fields = _unknown_manifest_field_defaults(model_id)
    fields.update(declared_fields)
    fields["model_id"] = model_id
    return PolicyManifest(**fields)


def _is_explicit_manifest_self_sufficient(manifest: PolicyManifest) -> bool:
    """True when a policy_manifest.json alone declares enough for
    CompatibilityGate's structural/semantic checks to run WITHOUT any
    ancestry/base-lineage resolution at all -- the case this task's
    chat report calls out explicitly: ACT, Diffusion Policy, or any
    other custom policy family that has no relationship whatsoever to
    this project's SmolVLA/LIBERO manifests must still be able to
    register a fully independent manifest for itself. Mirrors (but does
    not import, to avoid a circular dependency) the structural fields
    CompatibilityGate.check() itself gates on."""
    return (
        manifest.source_embodiment != UNKNOWN
        and manifest.action_space != ActionSpace.UNKNOWN
        and manifest.relative_or_absolute != UNKNOWN
        and manifest.rotation_representation != UNKNOWN
        and manifest.reference_frame != UNKNOWN
        and manifest.gripper_convention != UNKNOWN
        and manifest.action_dimension > 0
    )


def get_manifest(model_id: str) -> PolicyManifest:
    """Never raises. Resolves model_id through 4 priority tiers, each
    only attempted if the previous one didn't apply:

      1. Explicit manifest: an exact MANIFEST_REGISTRY entry (a
         hardcoded, project-registered model_id -- e.g.
         "HuggingFaceVLA/smolvla_libero").
      2. Explicit local manifest: model_id/policy_manifest.json, if
         present and self-sufficient (see
         _is_explicit_manifest_self_sufficient()) -- used AS-IS, with NO
         ancestry/lineage resolution attempted at all. This is the path
         for ACT/Diffusion Policy/any custom policy family: it never has
         to pretend to be a SmolVLA fine-tune to register a manifest.
      3. Ancestry fallback: local SmolVLA-family fine-tune/--resume
         chain resolution (see _resolve_finetuned_manifest()) -- for
         legacy/resumed checkpoints that have no policy_manifest.json of
         their own. If a policy_manifest.json IS present but was not
         self-sufficient (tier 2 didn't apply), its declared fields
         still WIN over whatever ancestry resolution derives for those
         same fields -- a partial self-declaration is allowed to
         override individual claims (e.g. "I know my own gripper
         convention even though I inherited my action shape") without
         requiring a full, standalone manifest.
      4. UNKNOWN / hard-fail: CompatibilityGate refuses production the
         same way it always has, with a notes string explaining exactly
         which tier failed and why.
    """
    manifest = MANIFEST_REGISTRY.get(model_id)
    if manifest is not None:
        return manifest

    declared_fields = _read_explicit_manifest_json(model_id)
    if declared_fields:
        explicit_local_manifest = _build_manifest_from_declared_fields(model_id, declared_fields)
        if _is_explicit_manifest_self_sufficient(explicit_local_manifest):
            return explicit_local_manifest

    finetuned_manifest, ancestry_failure_reason = _resolve_finetuned_manifest(model_id)
    if finetuned_manifest is not None:
        if declared_fields:
            finetuned_manifest = replace(finetuned_manifest, **declared_fields)
        return finetuned_manifest

    notes = f"No PolicyManifest registered for model_id={model_id!r} -- treated as fully unverified."
    if declared_fields:
        notes += (
            f" A {_EXPLICIT_MANIFEST_FILENAME} was found but was not self-sufficient (missing "
            f"one or more of source_embodiment/action_space/relative_or_absolute/"
            f"rotation_representation/reference_frame/gripper_convention/action_dimension), and "
            f"ancestry resolution also failed: {ancestry_failure_reason}"
        )
    elif ancestry_failure_reason is not None:
        notes += f" This IS a local checkpoint, but ancestry resolution failed: {ancestry_failure_reason}"

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
        notes=notes,
    )
