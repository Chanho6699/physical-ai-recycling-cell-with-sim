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

import math
from typing import Optional, Tuple

from policy_semantics.canonical_command import CanonicalRobotCommand
from policy_semantics.canonical_observation import CanonicalObservation
from policy_semantics.interfaces import ActionAdapter, ObservationAdapter
from policy_semantics.manifest import UNKNOWN, PolicyManifest
from policy_semantics.native_policy_action import NativePolicyAction

ADAPTER_NAME = "SmolVLALiberoActionAdapter"
ADAPTER_VERSION = "v0"

# robosuite/controllers/config/robots/default_panda.json (OSC_POSE, Panda).
# No longer consumed directly by decode() (see
# _decode_translation_rotation(), manifest-driven since this task's
# duplicate-scaling fix) -- kept as the cited, documented source for
# _SMOLVLA_LIBERO_MANIFEST's own native_action_clip_range/
# native_translation_scale_m/native_rotation_scale_rad values in
# policy_semantics/manifest.py, and reused by
# benchmark/diagnose_translation_rotation_scaling.py's own reporting.
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
    @staticmethod
    def _decode_gripper(raw_gripper: float, manifest: PolicyManifest) -> Optional[Tuple[float, bool]]:
        """Converts a checkpoint's own NATIVE-scale raw gripper value into
        CanonicalRobotCommand's fixed gripper_opening_01 scale (1.0=fully
        open, 0.0=fully closed), using manifest.native_gripper_range/
        native_gripper_min_means/native_gripper_max_means instead of a
        single hardcoded (-1, 1) formula -- see PolicyManifest's
        docstring and this task's chat report for the confirmed bug this
        replaces: a locally fine-tuned checkpoint whose own postprocessor
        produces a DIFFERENT native range (e.g. this project's own
        (0.0, 1.0) vs. LIBERO's (-1.0, 1.0)) decoded its entire output
        range to a single constant ("close") under the old fixed formula,
        regardless of what the model actually intended.

        Returns (gripper_opening_01, was_clipped), or None if the
        manifest doesn't declare a usable (valid, unambiguous) native
        range/polarity -- never guesses one -- or if raw_gripper itself
        is NaN/Inf. CompatibilityGate's gripper_native_range_known check
        (see compatibility_gate.py) is meant to keep this path from ever
        being reached with an unusable manifest in production, but this
        is checked again here regardless (defense in depth, same
        pattern as this method's postprocessor_used check above)."""
        if math.isnan(raw_gripper) or math.isinf(raw_gripper):
            return None

        native_range = manifest.native_gripper_range
        min_means = manifest.native_gripper_min_means
        max_means = manifest.native_gripper_max_means
        if (
            native_range is None
            or native_range[0] >= native_range[1]
            or min_means == UNKNOWN
            or max_means == UNKNOWN
            or min_means == max_means
        ):
            return None

        native_min, native_max = native_range
        clipped_value = _clip(raw_gripper, native_min, native_max)
        was_clipped = clipped_value != raw_gripper

        if min_means == "open" and max_means == "close":
            opening_01 = (native_max - clipped_value) / (native_max - native_min)
        else:  # the only other valid combination, given min_means != max_means above
            opening_01 = (clipped_value - native_min) / (native_max - native_min)

        return _clip(opening_01, 0.0, 1.0), was_clipped

    @staticmethod
    def _decode_translation_rotation(values_0_5: list, manifest: PolicyManifest) -> Optional[Tuple[tuple, tuple, bool]]:
        """Converts a checkpoint's own NATIVE-scale raw translation/
        rotation values (dims 0-5) into physical meters/radians, using
        manifest.native_translation_scale_m/native_rotation_scale_rad/
        native_action_clip_range instead of the hardcoded
        TRANSLATION_SCALE_M(0.05)/ROTATION_SCALE_RAD(0.5)/[-1,1] clip --
        see this task's chat report for the confirmed duplicate-scaling
        bug this replaces: a locally fine-tuned checkpoint whose own
        postprocessor already outputs real physical units (this
        project's own (1.0, 1.0) identity scale, no native clip) got a
        SECOND, LIBERO-specific 0.05m/0.5rad multiply applied on top,
        shrinking every commanded step by ~20x regardless of what the
        model actually intended.

        Returns (translation_m, rotation_axis_angle_rad, was_clipped),
        or None if the manifest doesn't declare a usable (valid) native
        scale/clip range -- never guesses one -- or any value is NaN/Inf.
        """
        if any(math.isnan(v) or math.isinf(v) for v in values_0_5):
            return None

        translation_scale = manifest.native_translation_scale_m
        rotation_scale = manifest.native_rotation_scale_rad
        clip_min, clip_max = manifest.native_action_clip_range
        if translation_scale is None or rotation_scale is None or clip_min >= clip_max:
            return None

        clipped = [_clip(v, clip_min, clip_max) for v in values_0_5]
        was_clipped = any(c != v for c, v in zip(clipped, values_0_5))

        translation_m = tuple(component * translation_scale for component in clipped[0:3])
        rotation_axis_angle_rad = tuple(component * rotation_scale for component in clipped[3:6])
        return translation_m, rotation_axis_angle_rad, was_clipped

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

        # Translation/rotation (dims 0-5): manifest-driven native scale/
        # clip -- see _decode_translation_rotation()'s docstring. LIBERO's
        # own manifest declares (0.05, 0.5, (-1,1)), reproducing this
        # adapter's original fixed behavior exactly (regression-tested);
        # a locally fine-tuned checkpoint whose postprocessor already
        # outputs real units declares (1.0, 1.0, (-inf,inf)) instead.
        translation_rotation_result = self._decode_translation_rotation(values[0:6], manifest)
        if translation_rotation_result is None:
            return None
        translation_m, rotation_axis_angle_rad, safety_clipped = translation_rotation_result
        clipped = [
            _clip(v, *manifest.native_action_clip_range) for v in values[0:6]
        ]

        raw_gripper = values[6]
        gripper_result = self._decode_gripper(raw_gripper, manifest)
        if gripper_result is None:
            return None
        gripper_opening_01, gripper_clipped = gripper_result
        safety_clipped = safety_clipped or gripper_clipped

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
                "native_action_values": clipped + [raw_gripper],
                "native_action_raw_values": values,
                "native_action_space": f"translation/rotation: clip={manifest.native_action_clip_range!r} -- "
                f"manifest.native_translation_scale_m={manifest.native_translation_scale_m!r}, "
                f"native_rotation_scale_rad={manifest.native_rotation_scale_rad!r}; "
                f"gripper: manifest.native_gripper_range={manifest.native_gripper_range!r}",
                "translation_scale_m": manifest.native_translation_scale_m,
                "rotation_scale_rad": manifest.native_rotation_scale_rad,
                "control_freq_hz": LIBERO_CONTROL_FREQ_HZ,
                "gripper_raw": raw_gripper,
                "gripper_convention": (
                    f"native_gripper_range={manifest.native_gripper_range!r}, "
                    f"min_means={manifest.native_gripper_min_means!r}, "
                    f"max_means={manifest.native_gripper_max_means!r}"
                ),
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
