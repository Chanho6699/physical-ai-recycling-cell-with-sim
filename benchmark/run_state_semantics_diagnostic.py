"""SmolVLA LIBERO 8D state semantics and distribution diagnostic (v0).

Follow-up to run_vla_action_direction_diagnostic.py and
run_counterfactual_direction_benchmark.py. Those isolated the executor and
adapter as innocent and found a fixed x/y directional bias plus an
instruction-wording effect. What's still unverified is whether the 8D
`observation.state` vector we send (from PyBulletPandaBackend.
get_libero_observation_state()) actually means what
HuggingFaceVLA/smolvla_libero was trained to expect. This script isolates
that into 6 candidate causes (see docstring section "CANDIDATE CAUSES"
below) using only READS of the checkpoint's own shipped files plus new,
non-production ablations.

Does not modify any production file, model checkpoint, or config. Every
ablation here is diagnostic-only: coordinate "hypothesis" transforms are
applied to a LOCAL COPY of the state dict built inside this script, never
inside policy_semantics/robot_sim/vla_server -- see
apply_coordinate_hypothesis()'s docstring for the explicit guarantee, and
benchmark/test_state_semantics_diagnostic.py's dedicated test for this.

=== INVESTIGATION FINDINGS (see final report for full detail/citations) ===

1. State field order -- CONFIRMED matching, two independent sources:
   - Our vla_server/model_loader.py's _SMOLVLA_LIBERO_STATE_FIELD_DIMS =
     (("ee_position", 3), ("ee_orientation_axis_angle", 3), ("gripper_qpos", 2))
   - The actual LIBERO->LeRobot dataset conversion formula (found via
     research, see huggingface/lerobot issue #940 discussion and LIBERO
     dataset documentation): observation.state =
     np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
   Same 3-field order, same dims (3, 3, 2). No reordering bug.

2. Quaternion convention -- CONFIRMED matching: robosuite's
   transform_utils.quat2axisangle()/quat2mat() docstrings state quat is
   (x, y, z, w) scalar-last. PyBullet's own convention (p.getQuaternionFromAxisAngle/
   p.getAxisAngleFromQuaternion, used throughout robot_sim/pybullet_panda_backend.py)
   is also (x, y, z, w) scalar-last. No component-order mismatch.

3. Axis-angle formula -- CONFIRMED matching: robosuite's quat2axisangle()
   computes axis*angle where angle = 2*acos(w) (clipping w to [-1, 1]
   first). This is the same standard formula PyBullet's
   getAxisAngleFromQuaternion() + our own axis*angle multiplication in
   get_libero_observation_state() implements. HOWEVER (see finding 6
   below) this representation is numerically unstable near angle=pi,
   which is exactly the operating regime both sides sit in.

4. Position/orientation frame -- robosuite's robot0_eef_pos/robot0_eef_quat
   sensors read MuJoCo's site_xpos/site_xmat, which are WORLD frame
   (confirmed: robosuite/robots/robot.py's eef_pos sensor is
   `self.sim.data.site_xpos[self.eef_site_id[arm]]`, and MuJoCo site_xpos
   is always world frame for a body, not base-relative, for the
   fixed-base Panda LIBERO uses -- base_to_eef_pos, a DIFFERENT sensor
   defined only in robosuite's mobile_robot.py, is not what LIBERO's
   fixed Panda tasks use). Our PyBullet backend already documents that
   its own base sits at world origin with identity orientation, so
   world == robot_base frame for us too -- same frame *convention*, but
   the absolute origin placement (where the table/workspace sits in that
   world frame) is env-specific and NOT expected to numerically match
   between LIBERO's original environment and our PyBullet scene. This
   matters for interpreting the position-channel (dims 0-2) z-scores
   below: a large z-score there is only weak evidence of a real bug,
   since a different absolute table/workspace placement alone would
   already produce one.

5. Gripper qpos meaning -- SUSPECTED mismatch, NOT fully confirmed
   (matches this task's requirement to distinguish confirmed vs.
   inferred). robosuite's raw gripper_qpos sensor
   (robosuite/robots/robot.py) is `[sim.data.qpos[x] for x in
   gripper_joint_pos_indexes]` -- i.e. two independent MuJoCo prismatic
   joint positions, each physically constrained to a non-negative range
   (same [0, 0.04]-style range as our own PyBullet
   panda_finger_joint1/2). Both should therefore be >= 0. But the
   checkpoint's own shipped observation.state.mean (see
   load_checkpoint_state_stats() below) has dims 6/7 = [+0.0269, -0.0272]
   -- essentially equal magnitude, OPPOSITE SIGN. A raw, always-non-negative
   joint value cannot have a negative training-set mean, so *something*
   between raw robosuite sensor output and the final stored
   observation.state negates one of the two gripper channels. This is a
   REAL, independently-corroborated anomaly -- huggingface/lerobot issue
   #940 raises the exact same observation ("the last two values appear
   to be negative of each other") and was closed "not planned" without
   ever confirming whether it's an intentional convention or a labeling
   bug. Our own get_libero_observation_state() returns
   [left_finger_qpos, right_finger_qpos], both from PyBullet's
   panda_finger_joint1/2, both non-negative -- i.e. we do NOT negate
   either channel. If the checkpoint's training convention did negate
   one, our gripper_qpos[1] channel is likely sign-flipped relative to
   what training data looked like.

6. Orientation operating point -- both regimes are near a PI-RADIAN
   rotation (checkpoint's mean axis-angle magnitude ~2.98 rad, dominant
   on the x-component index 3; OUR PyBullet reset state's axis-angle is
   ~3.1415 on the same x-component, see
   test_state_semantics_diagnostic.py's regression check) -- i.e. NOT a
   wildly different "pointing the wrong way" orientation regime, both
   represent a gripper-down-ish pose. BUT operating this close to pi
   radians is exactly where axis-angle representation is numerically
   unstable: a quaternion q and -q represent the identical physical
   rotation, but quat2axisangle(q) and quat2axisangle(-q) generally
   produce very different vectors except exactly at angle=pi (where they
   coincide up to an overall sign flip) -- see this module's
   detect_orientation_discontinuity() and the final report for measured
   evidence of whether this actually causes jumpy state values in
   practice, in this project's own PyBullet backend.

=== CANDIDATE CAUSES this script's ablations separate ===
  1. state field ORDER error               (Ablation A + a static regression check)
  2. coordinate frame/axis MEANING mismatch (Ablation C: sign-flip/swap hypotheses)
  3. orientation axis-angle CONVERSION bug  (detect_orientation_discontinuity())
  4. gripper qpos MEANING mismatch          (Ablation A's gripper-open/closed variants + finding 5 above)
  5. state values OUT OF DISTRIBUTION       (compute_zscore_report())
  6. model over-relies on state vs. image   (Ablation A vs. Ablation B comparison)

Run (needs a live server -- this script must run under the GPU venv,
e.g. .venv-vla, since load_checkpoint_state_stats() needs
safetensors/huggingface_hub which the plain PyBullet-only .venv does not
have; RealVLAPolicyClient/PyBulletPandaBackend work fine there too --
see run_vla_action_direction_diagnostic.py/run_counterfactual_direction_benchmark.py's
own module docstrings for the server startup commands):

  .venv-vla/bin/python -m benchmark.run_state_semantics_diagnostic \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS, _mean, _sign_match, _stdev
from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_VLA_CONFIG = "configs/vla_backend_smolvla_libero_config.json"
DEFAULT_CHECKPOINT_REPO_ID = "HuggingFaceVLA/smolvla_libero"
DEFAULT_STATS_FILENAME = "policy_preprocessor_step_5_normalizer_processor.safetensors"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]

STATE_DIM_NAMES = [
    "ee_position.x", "ee_position.y", "ee_position.z",
    "ee_orientation_axis_angle.x", "ee_orientation_axis_angle.y", "ee_orientation_axis_angle.z",
    "gripper_qpos.0", "gripper_qpos.1",
]

OOD_ZSCORE_THRESHOLD = 3.0
DISCONTINUITY_JUMP_THRESHOLD_RAD = 1.0  # see detect_orientation_discontinuity()
IMAGE_INSENSITIVITY_COSINE_STD_CEILING = 0.05  # see judge_image_vs_state_sensitivity()
HYPOTHESIS_SIGN_ACCURACY_SPIKE_MARGIN = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_REAL_VLA_CONFIG)
    parser.add_argument("--checkpoint-repo-id", type=str, default=DEFAULT_CHECKPOINT_REPO_ID)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    parser.add_argument("--num-distribution-samples", type=int, default=20)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--output-dir", type=str, default="results/state_semantics_diagnostic")
    return parser.parse_args()


# --- 1. Checkpoint stats (read-only; never modifies the checkpoint) ---


def load_checkpoint_state_stats(repo_id: str = DEFAULT_CHECKPOINT_REPO_ID) -> dict:
    """Reads observation.state.mean/.std straight out of the checkpoint's
    own shipped normalizer safetensors file (see this module's docstring,
    finding 1) -- the exact numbers HuggingFaceVLA/smolvla_libero's
    training data had for this channel, no guessing. Lazy-imports
    huggingface_hub/safetensors (only available in the GPU venv, e.g.
    .venv-vla) so importing this module elsewhere doesn't fail -- callers
    that don't need checkpoint stats (e.g. the ablation ends of this
    script, or tests using a fake policy) never hit this import."""
    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open

    stats_path = try_to_load_from_cache(repo_id=repo_id, filename=DEFAULT_STATS_FILENAME)
    if stats_path is None or not Path(stats_path).exists():
        raise FileNotFoundError(
            f"{DEFAULT_STATS_FILENAME} not found in the local HuggingFace cache for {repo_id!r} -- "
            "load the model once via vla_server/model_loader.py (or /load_model) so it's downloaded, "
            "then retry. This function only reads the existing cache; it never downloads anything itself."
        )

    with safe_open(stats_path, framework="pt") as handle:
        state_mean = handle.get_tensor("observation.state.mean").tolist()
        state_std = handle.get_tensor("observation.state.std").tolist()
        action_mean = handle.get_tensor("action.mean").tolist()
        action_std = handle.get_tensor("action.std").tolist()

    return {
        "repo_id": repo_id,
        "stats_file": str(stats_path),
        "observation_state_mean": state_mean,
        "observation_state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


# --- 2. Empirical distribution collection (pure PyBullet kinematics, no model calls) ---


def collect_empirical_state_distribution(num_samples: int) -> list:
    """Samples get_libero_observation_state() across num_samples distinct
    arm configurations (small varied joint-space moves from reset, real
    PyBullet forward kinematics each time -- not synthetic data) so the
    distribution reflects what this project's backend actually produces,
    not just a single frozen pose."""
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    samples = [backend.get_libero_observation_state()]

    # Small, varied moves (not just +x every time) so the sample spread
    # reflects genuine pose variation, same mechanism apply_command()
    # already uses in every other diagnostic in this project.
    from action_adapter.adapter_v0 import RobotCommand

    deltas = [
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0), (-0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.02, 0.0, 0.0, 0.0, 0.0), (0.0, -0.02, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -0.02, 0.05, 0.0, 0.0), (0.0, 0.0, -0.02, 0.0, 0.05, 0.0),
        (0.01, 0.01, -0.01, 0.0, 0.0, 0.05), (-0.01, -0.01, -0.01, -0.05, 0.0, 0.0),
    ]
    step = 0
    while len(samples) < num_samples:
        dx, dy, dz, drx, dry, drz = deltas[step % len(deltas)]
        command = RobotCommand(
            target_dx=dx, target_dy=dy, target_dz=dz,
            target_droll=drx, target_dpitch=dry, target_dyaw=drz,
            gripper_command="close" if step % 4 == 0 else "open",
        )
        backend.apply_command(command, steps=10)
        samples.append(backend.get_libero_observation_state())
        step += 1

    backend.shutdown()
    return samples[:num_samples]


def compute_zscore_report(samples: list, checkpoint_mean: list, checkpoint_std: list) -> dict:
    """Per-dimension min/max/mean/std of our OWN samples, the checkpoint's
    own mean/std for that dimension, and z = (our_sample_mean -
    checkpoint_mean) / checkpoint_std, plus what fraction of INDIVIDUAL
    samples (not just the mean) land beyond |z| > OOD_ZSCORE_THRESHOLD."""
    dims = len(STATE_DIM_NAMES)
    report = {}
    for dim_index in range(dims):
        values = [sample[dim_index] for sample in samples]
        our_mean = sum(values) / len(values)
        our_std = (sum((v - our_mean) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0
        ckpt_mean = checkpoint_mean[dim_index]
        ckpt_std = checkpoint_std[dim_index]
        z_of_mean = (our_mean - ckpt_mean) / ckpt_std if ckpt_std > 1e-9 else None
        per_sample_z = [((v - ckpt_mean) / ckpt_std) if ckpt_std > 1e-9 else None for v in values]
        out_of_distribution_fraction = (
            sum(1 for z in per_sample_z if z is not None and abs(z) > OOD_ZSCORE_THRESHOLD) / len(per_sample_z)
        )
        report[STATE_DIM_NAMES[dim_index]] = {
            "our_min": min(values), "our_max": max(values), "our_mean": our_mean, "our_std": our_std,
            "checkpoint_mean": ckpt_mean, "checkpoint_std": ckpt_std,
            "z_of_mean": z_of_mean,
            "out_of_distribution_fraction": out_of_distribution_fraction,
            "is_ood": (abs(z_of_mean) > OOD_ZSCORE_THRESHOLD) if z_of_mean is not None else None,
        }
    return report


def detect_orientation_discontinuity(samples: list) -> dict:
    """Flags a step-to-step JUMP in the 3D axis-angle vector (dims 3:6)
    bigger than DISCONTINUITY_JUMP_THRESHOLD_RAD -- evidence of the
    representational instability this module's docstring (finding 6)
    describes: near a pi-radian rotation, a tiny underlying quaternion
    sign change can produce a large apparent jump in axis-angle space
    even though the physical orientation barely changed."""
    jumps = []
    for i in range(1, len(samples)):
        previous_axis_angle = samples[i - 1][3:6]
        current_axis_angle = samples[i][3:6]
        jump = sum((current_axis_angle[k] - previous_axis_angle[k]) ** 2 for k in range(3)) ** 0.5
        jumps.append(jump)
    discontinuity_count = sum(1 for jump in jumps if jump > DISCONTINUITY_JUMP_THRESHOLD_RAD)
    return {
        "num_transitions": len(jumps),
        "max_jump_rad": max(jumps) if jumps else None,
        "mean_jump_rad": (sum(jumps) / len(jumps)) if jumps else None,
        "discontinuity_count": discontinuity_count,
        "discontinuity_detected": discontinuity_count > 0,
        "jump_threshold_rad": DISCONTINUITY_JUMP_THRESHOLD_RAD,
    }


# --- shared request-building helpers ---


def _predict_and_record(policy, image_role_pair, robot_state, instruction, object_position, bin_position, seed, step_index, strict, label):
    main_image, wrist_image = image_role_pair
    policy_input = PolicyInput(
        image=main_image,
        instruction=instruction,
        robot_state=robot_state,
        task_goal={},
        target_object_position=object_position,
        bin_position=bin_position,
        step_index=step_index,
        phase="move_to_object",
        images_by_role={"main": main_image, "wrist": wrist_image},
        seed=seed,
    )
    policy_output = policy.predict_action(policy_input)
    info = policy_output.info or {}

    compatibility_passed = (info.get("compatibility") or {}).get("passed")
    semantic_action_valid = bool(info.get("semantic_action_valid", True))
    degraded_input = bool(info.get("degraded_input", False))
    fallback_used = bool(info.get("fallback_used", False))

    if strict:
        violations = []
        if compatibility_passed is not True:
            violations.append(f"compatibility.passed={compatibility_passed!r}")
        if not semantic_action_valid:
            violations.append("semantic_action_valid=False")
        if degraded_input:
            violations.append("degraded_input=True")
        if fallback_used:
            violations.append("fallback_used=True")
        if violations:
            raise RuntimeError(f"--strict violated at {label} seed={seed}: {'; '.join(violations)}. info={info}")

    action_postprocess = info.get("action_postprocess") or {}
    canonical_after = action_postprocess.get("canonical_command") or {}
    metadata = canonical_after.get("metadata") or {}
    raw_model_action = metadata.get("raw_model_action")
    adapted_translation = canonical_after.get("translation_m")

    action_adapter = ActionAdapter()
    robot_command = action_adapter.convert(policy_output.action)
    commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]

    vector_to_object = [object_position[i] - robot_state["ee_position"][i] for i in range(3)]
    cosine_commanded = _cosine_similarity(commanded_translation, vector_to_object)
    sign_match_xyz = [_sign_match(commanded_translation[i], vector_to_object[i]) for i in range(3)]

    return {
        "label": label,
        "seed": seed,
        "instruction": instruction,
        "state_8d_before_normalization": list(robot_state["ee_position"]) + list(robot_state["ee_orientation_axis_angle"]) + list(robot_state["gripper_qpos"]),
        "object_position": object_position,
        "raw_model_action": raw_model_action,
        "adapted_translation": adapted_translation,
        "commanded_translation": commanded_translation,
        "cosine_commanded_vs_object": cosine_commanded,
        "sign_match_x": sign_match_xyz[0],
        "sign_match_y": sign_match_xyz[1],
        "sign_match_z": sign_match_xyz[2],
        "gripper_command": robot_command.gripper_command,
        "server_latency_ms": info.get("inference_latency_ms"),
        "main_image_hash": image_hash(main_image),
        "wrist_image_hash": image_hash(wrist_image),
        "compatibility_passed": compatibility_passed,
        "semantic_action_valid": semantic_action_valid,
        "degraded_input": degraded_input,
        "fallback_used": fallback_used,
    }


# --- 6A. Image-fixed / state-varied ---


def build_state_variants(base_robot_state: dict) -> dict:
    """Diagnostic-only state variants -- built as plain dicts here, never
    touching robot_sim/pybullet_panda_backend.py or
    vla_server/model_loader.py. "original" is the real, unmodified state;
    every other key is a synthetic counterfactual used ONLY by this
    ablation to see how much the model's output changes when just the
    state (not the image) is edited."""
    ee_position = list(base_robot_state["ee_position"])
    ee_orientation = list(base_robot_state["ee_orientation_axis_angle"])
    gripper_qpos = list(base_robot_state["gripper_qpos"])

    def with_state(position=None, orientation=None, gripper=None):
        return {
            **base_robot_state,
            "ee_position": position if position is not None else ee_position,
            "ee_orientation_axis_angle": orientation if orientation is not None else ee_orientation,
            "gripper_qpos": gripper if gripper is not None else gripper_qpos,
        }

    return {
        "original": with_state(),
        "x_mirrored": with_state(position=[-ee_position[0], ee_position[1], ee_position[2]]),
        "y_mirrored": with_state(position=[ee_position[0], -ee_position[1], ee_position[2]]),
        "xy_swapped": with_state(position=[ee_position[1], ee_position[0], ee_position[2]]),
        "position_zeroed": with_state(position=[0.0, 0.0, 0.0]),
        "orientation_zeroed": with_state(orientation=[0.0, 0.0, 0.0]),
        "gripper_open": with_state(gripper=[0.04, 0.04]),
        "gripper_closed": with_state(gripper=[0.0, 0.0]),
    }


def run_image_fixed_state_varied(policy, backend, instruction, object_position, bin_position, seeds, strict) -> list:
    robot_state, _state_8d, _object_position = build_robot_state(backend)
    main_image = backend.render_main_camera()
    wrist_image = backend.render_wrist_camera()
    variants = build_state_variants(robot_state)

    rows = []
    for variant_name, variant_state in variants.items():
        for seed in seeds:
            row = _predict_and_record(
                policy, (main_image, wrist_image), variant_state, instruction, object_position, bin_position,
                seed, 0, strict, f"ablationA__{variant_name}",
            )
            row["variant"] = variant_name
            rows.append(row)
    return rows


# --- 6B. State-fixed / image-varied ---


def run_state_fixed_image_varied(policy, instruction, positions, bin_position, seeds, object_type, strict) -> list:
    # A single, fixed state (a real pose, captured once) is reused for
    # every image variant below -- only the rendered main/wrist images
    # (and the object_position used purely for the cosine/sign-match
    # ground truth, NOT sent to the model) differ per position.
    reference_backend = PyBulletPandaBackend(gui=False)
    reference_backend.reset()
    fixed_robot_state, _state_8d, _fixed_object_position = build_robot_state(reference_backend)
    reference_backend.shutdown()

    rows = []
    for position_name, position in positions.items():
        image_backend = PyBulletPandaBackend(gui=False)
        image_backend.reset()
        image_backend.set_object_type(object_type)
        image_backend.set_object_position(list(position))
        main_image = image_backend.render_main_camera()
        wrist_image = image_backend.render_wrist_camera()
        image_backend.shutdown()

        for seed in seeds:
            row = _predict_and_record(
                policy, (main_image, wrist_image), fixed_robot_state, instruction, list(position), bin_position,
                seed, 0, strict, f"ablationB__{position_name}",
            )
            row["variant"] = position_name
            rows.append(row)
    return rows


# --- 6C. Diagnostic-only coordinate hypotheses ---


def apply_coordinate_hypothesis(robot_state: dict, hypothesis: str) -> dict:
    """Returns a NEW dict -- never mutates robot_state in place, and is
    only ever called from inside this diagnostic script's own
    run_coordinate_hypotheses() below. Nothing in policy_semantics/,
    vla_server/, or robot_sim/ calls this function or anything like it --
    see test_state_semantics_diagnostic.py's dedicated test asserting
    exactly that (grep-based, not just "trust me")."""
    x, y, z = robot_state["ee_position"]
    if hypothesis == "identity":
        new_position = [x, y, z]
    elif hypothesis == "x_sign_flip":
        new_position = [-x, y, z]
    elif hypothesis == "y_sign_flip":
        new_position = [x, -y, z]
    elif hypothesis == "xy_swap":
        new_position = [y, x, z]
    elif hypothesis == "xy_swap_sign":
        new_position = [-y, -x, z]
    else:
        raise ValueError(f"Unknown coordinate hypothesis: {hypothesis!r}")
    return {**robot_state, "ee_position": new_position}


COORDINATE_HYPOTHESES = ("identity", "x_sign_flip", "y_sign_flip", "xy_swap", "xy_swap_sign")


def run_coordinate_hypotheses(policy, positions, instruction, bin_position, seeds, object_type, strict) -> list:
    rows = []
    for position_name, position in positions.items():
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        backend.set_object_type(object_type)
        backend.set_object_position(list(position))
        robot_state, _state_8d, object_position = build_robot_state(backend)
        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()
        backend.shutdown()

        for hypothesis in COORDINATE_HYPOTHESES:
            hypothesis_state = apply_coordinate_hypothesis(robot_state, hypothesis)
            for seed in seeds:
                row = _predict_and_record(
                    policy, (main_image, wrist_image), hypothesis_state, instruction, object_position, bin_position,
                    seed, 0, strict, f"ablationC__{position_name}__{hypothesis}",
                )
                row["variant"] = f"{position_name}__{hypothesis}"
                row["position_name"] = position_name
                row["hypothesis"] = hypothesis
                rows.append(row)
    return rows


# --- 7. Automatic judgments ---


def judge_image_vs_state_sensitivity(rows_a: list, rows_b: list) -> dict:
    """Ablation A varies STATE with a fixed image -> spread in
    commanded_translation there measures STATE sensitivity. Ablation B
    varies the IMAGE with a fixed state -> spread there measures IMAGE
    (visual) sensitivity. Comparing the two spreads (using
    cosine_commanded_vs_object's std as a cheap 1-number summary of "how
    much did the output change") says which input the model is actually
    listening to."""
    def _cosine_std(rows):
        values = [row["cosine_commanded_vs_object"] for row in rows if row["cosine_commanded_vs_object"] is not None]
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    state_sensitivity_std = _cosine_std(rows_a)
    image_sensitivity_std = _cosine_std(rows_b)

    if state_sensitivity_std is None or image_sensitivity_std is None:
        return {"verdict": "unknown", "reason": "not enough rows in one of the two ablations"}

    image_insensitive = image_sensitivity_std < IMAGE_INSENSITIVITY_COSINE_STD_CEILING
    state_dominant = state_sensitivity_std > image_sensitivity_std * 2

    if image_insensitive and state_dominant:
        verdict = "state_overreliance"
        reason = (
            f"varying the image alone barely moves cosine_commanded (std={image_sensitivity_std:.3f}, below "
            f"{IMAGE_INSENSITIVITY_COSINE_STD_CEILING}), while varying state alone moves it much more "
            f"(std={state_sensitivity_std:.3f}) -- the model appears to weight state far more heavily than "
            "the visual input for this decision."
        )
    elif state_dominant:
        verdict = "state_dominant"
        reason = (
            f"state variation (std={state_sensitivity_std:.3f}) drives more output change than image variation "
            f"(std={image_sensitivity_std:.3f}), though the image is not completely ignored."
        )
    elif image_sensitivity_std > state_sensitivity_std * 2:
        verdict = "image_dominant"
        reason = (
            f"image variation (std={image_sensitivity_std:.3f}) drives more output change than state variation "
            f"(std={state_sensitivity_std:.3f}) -- the model reacts more to what it sees than to the reported state."
        )
    else:
        verdict = "balanced"
        reason = f"state (std={state_sensitivity_std:.3f}) and image (std={image_sensitivity_std:.3f}) sensitivities are comparable."

    return {
        "verdict": verdict, "reason": reason,
        "state_sensitivity_cosine_std": state_sensitivity_std, "image_sensitivity_cosine_std": image_sensitivity_std,
    }


def judge_hypothesis_sign_accuracy(rows_c: list) -> dict:
    """Per coordinate hypothesis, mean x+y sign accuracy across all
    positions/seeds -- if one hypothesis (other than "identity") spikes
    well above the rest, that specific coordinate transform is a
    plausible fix for the fixed-bias pattern found in the previous
    counterfactual benchmark."""
    by_hypothesis = {}
    for hypothesis in COORDINATE_HYPOTHESES:
        hypothesis_rows = [row for row in rows_c if row.get("hypothesis") == hypothesis]
        xy_matches = [
            row[f"sign_match_{axis}"] for row in hypothesis_rows for axis in ("x", "y") if row[f"sign_match_{axis}"] is not None
        ]
        accuracy = (sum(1 for match in xy_matches if match) / len(xy_matches)) if xy_matches else None
        by_hypothesis[hypothesis] = accuracy

    identity_accuracy = by_hypothesis.get("identity")
    if identity_accuracy is None:
        return {"suspected_hypothesis": None, "reason": "no identity-hypothesis rows to compare against", "by_hypothesis": by_hypothesis}

    best_hypothesis = max((h for h in by_hypothesis if by_hypothesis[h] is not None), key=lambda h: by_hypothesis[h], default=None)
    if best_hypothesis is None:
        return {"suspected_hypothesis": None, "reason": "no valid accuracy computed for any hypothesis", "by_hypothesis": by_hypothesis}

    spike = by_hypothesis[best_hypothesis] - identity_accuracy
    suspected = best_hypothesis != "identity" and spike > HYPOTHESIS_SIGN_ACCURACY_SPIKE_MARGIN
    return {
        "suspected_hypothesis": best_hypothesis if suspected else "identity",
        "reason": (
            f"'{best_hypothesis}' x+y sign accuracy ({by_hypothesis[best_hypothesis]:.3f}) is "
            f"{spike:.3f} above identity ({identity_accuracy:.3f}), above the {HYPOTHESIS_SIGN_ACCURACY_SPIKE_MARGIN} "
            "margin -- this coordinate remapping is a plausible fix (DIAGNOSTIC-ONLY finding; do not apply to "
            "production without further confirmation)."
            if suspected
            else f"no hypothesis beats identity by more than {HYPOTHESIS_SIGN_ACCURACY_SPIKE_MARGIN} "
            f"(identity={identity_accuracy:.3f}, best={best_hypothesis}:{by_hypothesis[best_hypothesis]:.3f}) -- "
            "no simple coordinate remap explains the bias."
        ),
        "by_hypothesis": by_hypothesis,
    }


def judge_ood_dimensions(zscore_report: dict) -> dict:
    ood_dims = [name for name, stats in zscore_report.items() if stats["is_ood"]]
    return {
        "ood_dimensions": ood_dims,
        "suspected": len(ood_dims) > 0,
        "reason": (
            f"dimensions {ood_dims} have |z_of_mean| > {OOD_ZSCORE_THRESHOLD} relative to the checkpoint's own "
            "training distribution -- out-of-distribution state input is a plausible contributor for these "
            "specific dimensions."
            if ood_dims
            else f"no dimension's mean exceeds |z| > {OOD_ZSCORE_THRESHOLD} relative to the checkpoint's training "
            "distribution."
        ),
    }


def summarize_root_cause(image_vs_state: dict, hypothesis_result: dict, ood_result: dict, discontinuity_result: dict) -> dict:
    candidates = []
    if image_vs_state.get("verdict") == "state_overreliance":
        candidates.append("model_over_relies_on_state")
    if hypothesis_result.get("suspected_hypothesis") not in (None, "identity"):
        candidates.append("coordinate_frame_axis_mismatch")
    if ood_result.get("suspected"):
        candidates.append("state_out_of_distribution")
    if discontinuity_result.get("discontinuity_detected"):
        candidates.append("orientation_axis_angle_instability")

    return {
        "candidates": candidates,
        "primary_candidate": candidates[0] if candidates else "unresolved_model_policy_quality",
        "reason": (
            f"{len(candidates)} candidate cause(s) flagged by the automatic judgments: {candidates}."
            if candidates
            else "none of the 4 measurable candidate causes (state overreliance, coordinate mismatch, OOD state, "
            "orientation instability) were flagged -- if the fixed bias persists, it is more likely the "
            "checkpoint's own learned policy quality on this task/scene, not a plumbing or representation bug."
        ),
    }


def print_report(zscore_report, discontinuity_result, image_vs_state, hypothesis_result, ood_result, root_cause) -> None:
    print("\n=== Per-dimension z-score report (vs. checkpoint training distribution) ===")
    for name, stats in zscore_report.items():
        z = stats["z_of_mean"]
        z_str = f"{z:+.2f}" if z is not None else "n/a"
        print(f"{name:<32} our_mean={stats['our_mean']:+.4f} ckpt_mean={stats['checkpoint_mean']:+.4f} z={z_str} OOD={stats['is_ood']}")

    print("\n=== Orientation discontinuity ===")
    for key, value in discontinuity_result.items():
        print(f"{key}: {value}")

    print("\n=== Image vs. state sensitivity ===")
    print(f"verdict: {image_vs_state['verdict']}")
    print(f"reason: {image_vs_state['reason']}")

    print("\n=== Coordinate hypothesis sign accuracy ===")
    print(f"suspected_hypothesis: {hypothesis_result['suspected_hypothesis']}")
    print(f"reason: {hypothesis_result['reason']}")
    print(f"by_hypothesis: {hypothesis_result['by_hypothesis']}")

    print("\n=== OOD dimensions ===")
    print(f"suspected: {ood_result['suspected']}")
    print(f"reason: {ood_result['reason']}")

    print("\n=== Root cause summary ===")
    print(f"primary_candidate: {root_cause['primary_candidate']}")
    print(f"reason: {root_cause['reason']}")


def run_all(args, policy=None) -> dict:
    owns_policy = policy is None
    if policy is None:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_stats = load_checkpoint_state_stats(args.checkpoint_repo_id)
    samples = collect_empirical_state_distribution(args.num_distribution_samples)
    zscore_report = compute_zscore_report(samples, checkpoint_stats["observation_state_mean"], checkpoint_stats["observation_state_std"])
    discontinuity_result = detect_orientation_discontinuity(samples)

    backend_a = PyBulletPandaBackend(gui=False)
    backend_a.reset()
    backend_a.set_object_type("plastic_bottle")
    default_object_position = DEFAULT_POSITIONS["center_right"]
    backend_a.set_object_position(list(default_object_position))
    rows_a = run_image_fixed_state_varied(policy, backend_a, args.instruction, default_object_position, DEFAULT_BIN_POSITION, args.seeds, args.strict)
    backend_a.shutdown()

    rows_b = run_state_fixed_image_varied(policy, args.instruction, DEFAULT_POSITIONS, DEFAULT_BIN_POSITION, args.seeds, "plastic_bottle", args.strict)

    hypothesis_positions = {"center_right": DEFAULT_POSITIONS["center_right"], "center_left": DEFAULT_POSITIONS["center_left"]}
    rows_c = run_coordinate_hypotheses(policy, hypothesis_positions, args.instruction, DEFAULT_BIN_POSITION, args.seeds, "plastic_bottle", args.strict)

    image_vs_state = judge_image_vs_state_sensitivity(rows_a, rows_b)
    hypothesis_result = judge_hypothesis_sign_accuracy(rows_c)
    ood_result = judge_ood_dimensions(zscore_report)
    root_cause = summarize_root_cause(image_vs_state, hypothesis_result, ood_result, discontinuity_result)

    result = {
        "checkpoint_stats": checkpoint_stats,
        "zscore_report": zscore_report,
        "discontinuity_result": discontinuity_result,
        "ablation_a_rows": rows_a,
        "ablation_b_rows": rows_b,
        "ablation_c_rows": rows_c,
        "judgments": {
            "image_vs_state_sensitivity": image_vs_state,
            "hypothesis_sign_accuracy": hypothesis_result,
            "ood_dimensions": ood_result,
            "root_cause": root_cause,
        },
    }

    log_path = output_dir / f"state_semantics_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as log_file:
        json.dump(result, log_file, ensure_ascii=False, indent=2)

    print_report(zscore_report, discontinuity_result, image_vs_state, hypothesis_result, ood_result, root_cause)
    print(f"\nFull result JSON: {log_path}")
    result["log_path"] = str(log_path)
    return result


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
