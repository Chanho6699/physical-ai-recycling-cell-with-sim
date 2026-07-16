"""SmolVLALiberoActionAdapter / SmolVLALiberoObservationAdapter -- the
first real, production-capable PolicyAdapter for
HuggingFaceVLA/smolvla_libero.

Every numeric constant here is sourced from official code actually read
this session, not inferred from the model's output values -- see the
per-constant comments for exact file/function locations (all read from
lerobot==0.6.0 and robosuite, installed in .venv-vla):

  - relative (delta) control, "base"-frame actions, axis-angle rotation
    delta: robosuite/controllers/parts/arm/osc.py,
    OperationalSpaceController.__init__ docstring ("input_ref_frame:
    ... 'base': actions are wrt to the robot body") and
    compute_goal_ori() docstring ("delta ... in axis-angle form
    [ax, ay, az]"). LIBERO's own env (lerobot/envs/libero.py,
    LiberoEnv.reset(): `robot.controller.use_delta = True` for
    control_mode="relative") uses this controller unmodified.

  - controller action scale (raw [-1, 1] -> physical meters/radians):
    robosuite/controllers/config/robots/default_panda.json and
    robosuite/controllers/config/default/parts/osc_pose.json, both
    output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5] / output_min=[-0.05,
    ..., -0.5, ...]. Confirmed this is what LIBERO actually loads (not
    a LIBERO-specific override) via
    Lifelong-Robot-Learning/LIBERO's env_wrapper.py calling
    `suite.load_controller_config(default_controller=controller)` --
    robosuite's own default loader, not a LIBERO-authored config.
    robosuite/controllers/parts/controller.py's Controller.scale_action()
    is the exact linear formula applied: with input range [-1, 1] and
    output range [-0.05, 0.05] (translation) / [-0.5, 0.5] (rotation),
    this reduces to `physical = raw * 0.05` / `physical = raw * 0.5`.

  - gripper convention: robosuite/models/grippers/panda_gripper.py,
    PandaGripper.format_action() docstring, verbatim: "Maps continuous
    action into binary output -1 => open, 1 => closed".

  - control period: lerobot/envs/libero.py's control_freq=20 (LIBERO's
    documented 20 Hz control frequency) -> 1/20 = 0.05s per action step.

  - native action space bounds: lerobot/envs/libero.py,
    ACTION_LOW = -1.0, ACTION_HIGH = 1.0, ACTION_DIM = 7 (the
    LiberoEnv.action_space every SmolVLA-on-LIBERO checkpoint is
    trained/evaluated against).

NOT yet verified (kept UNKNOWN in policy_semantics/manifest.py, still
blocking CompatibilityGate as of this change -- see that module):
  - Whether robosuite's MuJoCo Panda base-frame axis convention is
    numerically identical to this project's PyBullet
    franka_panda/panda.urdf base-frame axis convention. Both model the
    same physical Franka Panda base frame, which is a strong reason to
    expect they agree, but that is an assumption, not a verified fact
    (no direct byte-for-byte cross-simulator check has been run).
  - Whether HuggingFaceVLA/smolvla_libero's dataset_stats (baked into
    its shipped policy_postprocessor_step_1_unnormalizer_processor.safetensors)
    actually recovers values in exactly LiberoEnv's [-1, 1] range for
    this specific fine-tune, vs. some other post-training scale drift.
"""

from typing import Optional

from policy_semantics.canonical_command import CanonicalRobotCommand
from policy_semantics.canonical_observation import CanonicalObservation
from policy_semantics.interfaces import ActionAdapter, ObservationAdapter
from policy_semantics.manifest import PolicyManifest
from policy_semantics.native_policy_action import NativePolicyAction

ADAPTER_NAME = "SmolVLALiberoActionAdapter"
ADAPTER_VERSION = "v0"

# robosuite/controllers/config/robots/default_panda.json (OSC_POSE, Panda)
NATIVE_ACTION_LOW = -1.0
NATIVE_ACTION_HIGH = 1.0
TRANSLATION_SCALE_M = 0.05  # output_max[0:3] in default_panda.json
ROTATION_SCALE_RAD = 0.5  # output_max[3:6] in default_panda.json

# lerobot/envs/libero.py: control_freq=20 (Hz)
LIBERO_CONTROL_FREQ_HZ = 20
DURATION_S = 1.0 / LIBERO_CONTROL_FREQ_HZ

# Camera role -> this checkpoint's actual observation key, confirmed via
# its real config.json (input_features): observation.images.image (main/
# agentview) and observation.images.image2 (wrist/robot0_eye_in_hand).
CAMERA_ROLE_TO_KEY = {
    "main": "observation.images.image",
    "wrist": "observation.images.image2",
}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SmolVLALiberoObservationAdapter(ObservationAdapter):
    def build_preprocessor_input(self, observation: CanonicalObservation, manifest: PolicyManifest) -> dict:
        """CanonicalObservation -> the raw dict LeRobot's own official
        SmolVLA preprocessor pipeline (make_smolvla_pre_post_processors's
        input_steps, or the equivalent loaded via
        PolicyProcessorPipeline.from_pretrained(..., "policy_preprocessor.json"))
        expects: {"observation.images.<key>": array, "observation.state":
        array, "task": str}. Missing camera roles are reported (not
        zero-filled here -- the caller decides how to degrade, e.g. by
        setting CanonicalRobotCommand.degraded_input)."""
        missing = observation.missing_camera_roles(manifest.required_camera_roles)

        images = {}
        for role, key in CAMERA_ROLE_TO_KEY.items():
            if role in observation.images_by_role:
                images[key] = observation.images_by_role[role]

        return {
            "images": images,
            "observation.state": observation.robot_state,
            "task": observation.instruction,
            "missing_camera_roles": missing,
        }


class SmolVLALiberoActionAdapter(ActionAdapter):
    def decode(
        self, native_action: NativePolicyAction, manifest: PolicyManifest, context: dict
    ) -> Optional[CanonicalRobotCommand]:
        if not native_action.postprocessor_used:
            # A raw, un-postprocessed model tensor is not in LiberoEnv's
            # [-1, 1] action space at all -- decoding it with this
            # adapter's fixed scale factors would silently fabricate
            # meaning. Refuse instead of guessing (same policy as
            # BaseVLAAdapter.normalize_model_output()'s action=None contract).
            return None

        values = list(native_action.values)
        if len(values) != 7:
            return None

        safety_clipped = any(v < NATIVE_ACTION_LOW or v > NATIVE_ACTION_HIGH for v in values)
        clipped = [_clip(v, NATIVE_ACTION_LOW, NATIVE_ACTION_HIGH) for v in values]

        translation_m = tuple(component * TRANSLATION_SCALE_M for component in clipped[0:3])
        rotation_axis_angle_rad = tuple(component * ROTATION_SCALE_RAD for component in clipped[3:6])

        raw_gripper = clipped[6]
        # -1 (open) -> 1.0, +1 (closed) -> 0.0 (PandaGripper.format_action() polarity, see module docstring)
        gripper_opening_01 = _clip((1.0 - raw_gripper) / 2.0, 0.0, 1.0)

        degraded_input = bool(context.get("degraded_input", False))

        return CanonicalRobotCommand(
            translation_m=translation_m,
            rotation_axis_angle_rad=rotation_axis_angle_rad,
            gripper_opening_01=gripper_opening_01,
            duration_s=DURATION_S,
            source_policy=native_action.source_policy,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            safety_clipped=safety_clipped,
            degraded_input=degraded_input,
            metadata={
                "native_action_values": clipped,
                "native_action_raw_values": values,
                "native_action_space": "Box(-1, 1, shape=(7,)) -- lerobot/envs/libero.py",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "control_freq_hz": LIBERO_CONTROL_FREQ_HZ,
                "gripper_raw": raw_gripper,
                "gripper_convention": "robosuite PandaGripper.format_action(): -1=open, 1=closed",
                # Forwarded from NativePolicyAction.metadata (set by
                # vla_server/model_loader.py's _run_smolvla_libero_inference)
                # -- the truly raw, pre-official-postprocessor model tensor
                # and which images/state source were actually used this
                # step. Without this, CanonicalRobotCommand.to_info_dict()
                # only ever showed the post-postprocessor action
                # (native_action_raw_values above), never what the model
                # itself actually output before unnormalization.
                "raw_model_action": native_action.metadata.get("raw_model_action"),
                "images_source": native_action.metadata.get("images_source"),
                "state_source": native_action.metadata.get("state_source"),
            },
        )
