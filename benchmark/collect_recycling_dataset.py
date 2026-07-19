"""Recycling Cell LeRobot dataset collector (v0).

Runs DummyOpenVLAPolicy (the scripted, image-blind phase policy --
already proven able to reach task_status="success", see
policy/dummy_openvla_policy.py) across varied object positions and
instructions, and saves every SUCCESSFUL pick-and-place episode into a
real, official LeRobot v3.0 dataset via LeRobotDataset.create()/
add_frame()/save_episode() -- no hand-written parquet, no custom
schema. Failed episodes are discarded (clear_episode_buffer()), never
written to disk.

Also writes timing/frame_timing.jsonl next to the dataset root (one row
per SAVED frame): measured simulation_steps_elapsed/simulated_duration_s/
cumulative_simulated_time_s per frame (real p.stepSimulation() counts,
not estimates from command type -- see _FrameInstrumentation), since
apply_command()'s gripper redundant-actuation fix means physics duration
per frame is no longer constant (~40 steps on a "hold" frame, ~100 on a
gripper-transition frame -- see this task's chat report). This sidecar
is purely additive: it never touches the official data/meta/ LeRobot
schema, and (like collection_manifest.jsonl) only ever contains rows
from episodes that were actually save_episode()'d -- a failed episode's
timing rows are discarded together with its LeRobot buffer.

This is data collection ONLY: no fine-tuning is run here, no
checkpoint/production file is modified. See this task's final report
(chat) for how the produced dataset lines up with
HuggingFaceVLA/smolvla_libero's own training data format -- confirmed
directly against the real HuggingFaceVLA/libero dataset's
meta/info.json in an earlier turn (episode structure, 8D state,
two-camera 256x256 images, 7D action, all reused unchanged from this
project's already-verified production code):

  observation.images.image  <- PyBulletPandaBackend.render_main_camera()
  observation.images.image2 <- PyBulletPandaBackend.render_wrist_camera()
  observation.state         <- PyBulletPandaBackend.get_libero_observation_state() (8D,
                                gripper-channel-2 sign fix already applied in production)
  action                    <- DummyOpenVLAPolicy's own 7D
                                [dx,dy,dz,drx,dpitch,dyaw,gripper] output (the actual
                                command applied that step via action_adapter.adapter_v0.ActionAdapter)
  task                      <- the instruction string for this episode

Run:
  .venv-vla/bin/python -m benchmark.collect_recycling_dataset \\
    --episodes 10 --root datasets/recycling_lerobot_v0
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import robot_sim.pybullet_panda_backend as pybullet_panda_backend
from action_adapter.adapter_v0 import ActionAdapter
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO_ID = "local/recycling_cell_v0"
DEFAULT_ROOT = "datasets/recycling_lerobot_v0"
# Was 20 (v0). Measured real cadence (see timing/frame_timing.jsonl and
# this task's chat report) is ~6.0Hz on hold frames, ~2.4Hz on gripper-
# transition frames, ~5.5Hz blended over a typical episode -- 20 was
# never defensible (3.6x too fast even against the best case). 10
# matches HuggingFaceVLA/smolvla_libero's own declared training fps
# (meta/info.json's fps=10.0), which is the more useful nominal value
# for a fine-tuning pipeline (keeps this dataset's declared per-frame
# time scale consistent with what the checkpoint already learned from)
# and is within 2x of our real blended cadence, unlike 20's 3.6x-6x
# error. The exact, non-uniform real duration per frame remains fully
# recoverable from timing/frame_timing.jsonl regardless of what nominal
# fps is declared here -- this value is a compatibility choice for the
# official metadata, not a claim that cadence is actually uniform.
DEFAULT_FPS = 10
DEFAULT_OBJECT_TYPE = "plastic_bottle"
DEFAULT_MAX_STEPS_PER_EPISODE = 150
# Was 10 (v0). PyBulletPandaBackend.apply_command() used to always re-run
# a 60-step gripper actuation on every call (see this project's gripper
# redundant-actuation fix), and since gripper motor targets don't
# interrupt the arm's own in-flight position-control targets, the arm
# incidentally got 10+60=70 physics steps to converge every frame, not
# just the 10 explicitly requested. Now that redundant gripper actuation
# is skipped whenever the command doesn't change gripper state, "steady"
# frames (most of an episode) only get `steps_per_action` steps of
# convergence -- 10 was never actually enough on its own (measured
# mean state/action alignment error ~0.019m at steps=10 vs. ~0.0008m at
# steps=40); see this task's chat report. 40 is the smallest value
# measured to bring alignment error back under the 0.01m regression
# threshold with margin.
DEFAULT_STEPS_PER_ACTION = 40

# Same 4 object positions used throughout this project's counterfactual/
# offset diagnostics -- kept here as a small, self-contained literal
# (not imported from benchmark/run_counterfactual_direction_benchmark.py)
# so this collector has no dependency on the heavier demo-script import
# chain (onnxruntime etc.) and can run from a lean environment.
DEFAULT_POSITIONS = {
    "center_right": [0.42, 0.00, 0.05],
    "center_left": [0.27, 0.00, 0.05],
    "positive_y": [0.35, 0.18, 0.05],
    "negative_y": [0.35, -0.18, 0.05],
}

# LIBERO-style short, direct English phrasing (see the real
# HuggingFaceVLA/libero meta/tasks.parquet's actual instruction style,
# e.g. "put the bowl on the plate", "put the wine bottle on the rack" --
# confirmed in an earlier turn) plus this project's existing Korean
# instruction. DummyOpenVLAPolicy itself never reads this text (it's a
# scripted, image-blind policy -- see its module docstring); it's purely
# a dataset label here, matching how LIBERO's own "action" is what was
# actually executed, independent of whichever task string is attached.
DEFAULT_INSTRUCTIONS = {
    "ko_full": "플라스틱 병을 플라스틱 수거함에 넣어줘",
    "en_short": "Pick up the plastic bottle.",
    "en_full": "Pick up the plastic bottle and place it in the plastic bin.",
    "en_put": "Put the plastic bottle in the plastic bin.",
}

# Random per-attempt xy jitter added to whichever DEFAULT_POSITIONS anchor
# was picked (see resolve_position()) -- without this, every episode at
# "center_right" would be bit-for-bit the same object position, which
# would make position-distribution/split analysis meaningless.
JITTER_RADIUS_M = 0.03

# Train/validation/test split design (see this task's chat report, item
# 8, for the full rationale): validation withholds two anchor positions
# entirely (never seen with any seed during --split train collection);
# test is reserved for Real2Sim's real-camera-derived initial conditions
# later, not scripted-policy collection at all -- "positions" here is the
# only pool --split train/validation actually draw from.
SPLIT_POSITIONS = {
    "train": ["center_right", "center_left"],
    "validation": ["positive_y"],
    "test": ["negative_y"],  # held out from scripted collection entirely; reserved for Real2Sim test conditions.
}
SPLIT_SEED_RANGES = {
    "train": (0, 10_000),
    "validation": (10_000, 20_000),
    "test": (20_000, 30_000),
}

# Matches HuggingFaceVLA/smolvla_libero's own declared input_features
# exactly (confirmed against its cached config.json/meta/info.json in an
# earlier turn): two 256x256x3 uint8 images (dtype="image", i.e.
# use_videos=False -- embedded PNGs, same convention the real dataset
# itself uses), 8D float32 state, 7D float32 action.
FEATURES = {
    "observation.images.image": {
        "dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="Number of SUCCESSFUL episodes to save.")
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="Safety cap on total attempts regardless of success (default: --episodes * 5) -- "
        "so a policy/scene combination that can never succeed doesn't loop forever.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Must not already exist (LeRobotDataset.create()'s own requirement).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default=DEFAULT_OBJECT_TYPE)
    parser.add_argument(
        "--split", type=str, default="all", choices=["all", "train", "validation", "test"],
        help="Restrict which DEFAULT_POSITIONS anchors + seed range are used (see SPLIT_POSITIONS/"
        "SPLIT_SEED_RANGES). 'all' reproduces the original v0 behavior (all 4 anchors, no jitter).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base RNG seed for per-attempt position jitter (+/- JITTER_RADIUS_M in x/y). "
        "None (default) disables jitter entirely -- same fixed anchor positions as v0.",
    )
    parser.add_argument(
        "--diagnostic-force-failure", type=str, default=None,
        choices=["max_steps", "grasp_fail", "place_fail", "done_without_success"],
        help="Diagnostic only: force every attempt to fail via the named mode, to exercise the "
        "clear_episode_buffer() discard path on demand. Never used by default collection runs.",
    )
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def jitter_position(position, rng: Optional[random.Random]):
    """Adds independent uniform x/y jitter in [-JITTER_RADIUS_M, +JITTER_RADIUS_M]
    (z untouched -- objects always rest on the table at the same height).
    rng=None (the default whenever --seed is not passed) returns position
    unchanged, so existing callers/tests that don't care about jitter see
    the exact same fixed anchor positions as before this change."""
    if rng is None:
        return list(position)
    return [
        position[0] + rng.uniform(-JITTER_RADIUS_M, JITTER_RADIUS_M),
        position[1] + rng.uniform(-JITTER_RADIUS_M, JITTER_RADIUS_M),
        position[2],
    ]


class _FrameInstrumentation:
    """Collector-only physics-step instrumentation (no production file
    touched): counts real p.stepSimulation() calls per apply_command(),
    and flags whether THIS frame's apply_command() actually actuated the
    gripper -- by wrapping the specific backend instance's own
    open_gripper()/close_gripper() bound methods, not the class, so no
    other backend instance or production call path is affected. Used to
    populate timing/frame_timing.jsonl's simulation_steps_elapsed/
    gripper_transition fields with MEASURED values, not estimates from
    command type alone (a "close" command while already closed measures
    0 extra steps -- see the redundant-actuation fix)."""

    def __init__(self, backend: PyBulletPandaBackend):
        self.backend = backend
        self.step_count = 0
        self.gripper_actuated = False
        self._original_step = pybullet_panda_backend.p.stepSimulation
        self._original_open = backend.open_gripper
        self._original_close = backend.close_gripper

    def __enter__(self):
        def counting_step(*args, **kwargs):
            self.step_count += 1
            return self._original_step(*args, **kwargs)

        def counted_open(*args, **kwargs):
            self.gripper_actuated = True
            return self._original_open(*args, **kwargs)

        def counted_close(*args, **kwargs):
            self.gripper_actuated = True
            return self._original_close(*args, **kwargs)

        pybullet_panda_backend.p.stepSimulation = counting_step
        self.backend.open_gripper = counted_open
        self.backend.close_gripper = counted_close
        return self

    def __exit__(self, *exc_info):
        pybullet_panda_backend.p.stepSimulation = self._original_step
        self.backend.open_gripper = self._original_open
        self.backend.close_gripper = self._original_close

    def consume_frame(self, time_step: float):
        """Returns (simulation_steps_elapsed, simulated_duration_s,
        gripper_transition) for the frame just completed, and resets the
        per-frame counters for the next one."""
        steps = self.step_count
        gripper_transition = self.gripper_actuated
        self.step_count = 0
        self.gripper_actuated = False
        return steps, steps * time_step, gripper_transition


def run_one_episode(
    dataset,
    position,
    instruction,
    object_type,
    max_steps,
    steps_per_action,
    lie_object_position_offset=None,
    lie_bin_position_offset=None,
    force_done_after_step=None,
    instruction_name=None,
    seed=None,
    split=None,
    bin_position=None,
):
    """Runs one full scripted episode, add_frame()-ing every step into
    the dataset's CURRENT episode buffer -- but never calling
    save_episode() itself (the caller decides that based on whether the
    episode actually succeeded, per this task's explicit requirement
    that failed episodes are discarded, not saved).

    lie_object_position_offset / lie_bin_position_offset / force_done_after_step
    are diagnostic-only fault-injection hooks (all None -- the default --
    means zero behavior change from the original implementation). When
    set, they feed the POLICY a displaced target_object_position/
    bin_position (same "model sees an offset, ground truth/physics use
    the real value" pattern already used in
    benchmark/run_ee_position_offset_ab_experiment.py) or force
    policy_output.done True after a given step regardless of the
    underlying phase -- so a real, otherwise-successful run can be made
    to fail deterministically for testing clear_episode_buffer(), without
    touching DummyOpenVLAPolicy/PyBulletPandaBackend/ActionAdapter
    themselves.

    bin_position (default None, meaning "leave PyBulletPandaBackend's own
    reset() default bin pose untouched" -- byte-for-byte the same
    behavior every existing caller of this function already gets) lets a
    caller move the REAL, PHYSICAL bin before the episode starts (via
    backend.set_bin_position(), an existing production method -- see
    this task's chat report on why train80's single, never-varied bin
    position let the fine-tuned checkpoint learn a fixed release
    timing/trajectory instead of genuinely conditioning on the bin's
    visual position). DummyOpenVLAPolicy itself needs no change for
    this: it already reads bin_position out of PolicyInput every step
    (see policy/dummy_openvla_policy.py's move_above_bin phase), never a
    hardcoded coordinate -- only this collection entry point never had a
    way to tell the SIMULATOR to actually put the bin somewhere else.
    """
    backend = PyBulletPandaBackend(gui=False)
    state = backend.reset()
    backend.set_object_type(object_type)
    if bin_position is not None:
        state = backend.set_bin_position(list(bin_position))
    state = backend.set_object_position(list(position))
    bin_position = state["bin_position"]

    policy_target_object_position = list(position)
    if lie_object_position_offset is not None:
        policy_target_object_position = [
            position[i] + lie_object_position_offset[i] for i in range(3)
        ]
    policy_bin_position = bin_position
    if lie_bin_position_offset is not None:
        policy_bin_position = [bin_position[i] + lie_bin_position_offset[i] for i in range(3)]

    policy = DummyOpenVLAPolicy()
    policy.reset()
    action_adapter = ActionAdapter()

    success = False
    num_frames = 0
    final_status = state["task_status"]
    final_phase = policy.phase
    timing_rows = []
    cumulative_simulated_time_s = 0.0
    with _FrameInstrumentation(backend) as instrumentation:
        for step_index in range(max_steps):
            main_image = backend.render_main_camera()
            wrist_image = backend.render_wrist_camera()
            state_8d = backend.get_libero_observation_state()
            robot_state = backend.get_state()

            policy_input = PolicyInput(
                image=main_image,
                instruction=instruction,
                robot_state=robot_state,
                task_goal={},
                target_object_position=policy_target_object_position,
                bin_position=policy_bin_position,
                step_index=step_index,
                phase=policy.phase,
            )
            policy_output = policy.predict_action(policy_input)
            if force_done_after_step is not None and step_index >= force_done_after_step:
                policy_output.done = True
            robot_command = action_adapter.convert(policy_output.action)

            dataset.add_frame(
                {
                    "observation.images.image": main_image,
                    "observation.images.image2": wrist_image,
                    "observation.state": np.array(state_8d, dtype=np.float32),
                    "action": np.array(policy_output.action, dtype=np.float32),
                    "task": instruction,
                }
            )
            num_frames += 1
            # This frame's observation was captured at cumulative_simulated_time_s
            # (elapsed simulated time since episode start, measured -- not
            # frame_index/fps); simulation_steps_elapsed/simulated_duration_s
            # below describe the apply_command() call that takes the sim from
            # THIS frame's state to the next one, matching LeRobot's own
            # add_frame() timestamp convention ("when was this observation
            # captured") but with a real, non-uniform clock.
            frame_timestamp_s = cumulative_simulated_time_s

            robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
            simulation_steps_elapsed, simulated_duration_s, gripper_transition = instrumentation.consume_frame(
                backend.time_step
            )
            cumulative_simulated_time_s += simulated_duration_s

            timing_rows.append({
                "frame_index": step_index,
                "action_index": step_index,
                "simulation_steps_elapsed": simulation_steps_elapsed,
                "simulated_duration_s": simulated_duration_s,
                "cumulative_simulated_time_s": frame_timestamp_s,
                "gripper_transition": gripper_transition,
                "gripper_command": robot_command.gripper_command,
                "object_position": list(position),
                "instruction": instruction,
                "instruction_name": instruction_name,
                "seed": seed,
                "split": split,
            })

            final_status = robot_state_after["task_status"]
            final_phase = policy.phase
            if final_status == "success" or policy_output.done:
                success = final_status == "success"
                break

    backend.shutdown()
    return success, num_frames, final_status, final_phase, timing_rows


_FORCE_FAILURE_KWARGS = {
    # See run_one_episode()'s docstring. Offsets are large enough that the
    # lied-to target is well outside GRASP_THRESHOLD/PLACE_THRESHOLD of the
    # real object/bin, so the corresponding real-physics condition can
    # never be satisfied.
    "grasp_fail": {"lie_object_position_offset": [0.35, 0.0, 0.0]},
    "place_fail": {"lie_bin_position_offset": [0.35, 0.0, 0.0]},
    "done_without_success": {"force_done_after_step": 1},
}


def main() -> None:
    args = parse_args()
    max_attempts = args.max_attempts or args.episodes * 5

    root = resolve(args.root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=FEATURES,
        root=str(root),
        robot_type="franka_panda_pybullet",
        use_videos=False,
    )

    if args.split == "all":
        allowed_position_names = list(DEFAULT_POSITIONS.keys())
        seed_range = None
    else:
        allowed_position_names = SPLIT_POSITIONS[args.split]
        seed_range = SPLIT_SEED_RANGES[args.split]
    positions = [(name, DEFAULT_POSITIONS[name]) for name in allowed_position_names]
    instructions = list(DEFAULT_INSTRUCTIONS.items())

    max_steps = args.max_steps_per_episode
    fault_kwargs = {}
    if args.diagnostic_force_failure == "max_steps":
        max_steps = min(max_steps, 3)
    elif args.diagnostic_force_failure is not None:
        fault_kwargs = _FORCE_FAILURE_KWARGS[args.diagnostic_force_failure]

    manifest_path = root / "collection_manifest.jsonl"
    # Diagnostic sidecar, NOT part of the official LeRobot v3.0 layout --
    # a plain JSONL of per-attempt bookkeeping (position/instruction/seed/
    # success/failure reason) that the official format has no field for,
    # used only by benchmark/analyze_recycling_dataset.py's quality
    # report. Written next to (never inside) data/meta/, so it can't be
    # confused with or corrupt the official parquet files.
    #
    # timing/frame_timing.jsonl is a SEPARATE sidecar with a different
    # granularity/purpose: collection_manifest is one row per EPISODE
    # (attempt-level condition + outcome); frame_timing is one row per
    # FRAME (measured physics-step/duration timing), so downstream
    # analysis doesn't have to guess which rows belong together. Both are
    # written next to (never inside) data/meta/ -- the official LeRobot
    # schema is untouched by either.
    timing_dir = root / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    frame_timing_path = timing_dir / "frame_timing.jsonl"

    saved_episodes = 0
    success_count = 0
    attempt = 0

    print(f"=== Collecting {args.episodes} successful episodes -> {root} (split={args.split}) ===")
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
             open(frame_timing_path, "w", encoding="utf-8") as timing_file:
            while saved_episodes < args.episodes and attempt < max_attempts:
                position_name, anchor_position = positions[attempt % len(positions)]
                instruction_name, instruction = instructions[attempt % len(instructions)]

                episode_seed = None
                rng = None
                if args.seed is not None:
                    episode_seed = args.seed + attempt + (seed_range[0] if seed_range else 0)
                    rng = random.Random(episode_seed)
                position = jitter_position(anchor_position, rng)
                attempt += 1

                success, num_frames, final_status, final_phase, timing_rows = run_one_episode(
                    dataset, position, instruction, args.object_type,
                    max_steps, args.steps_per_action, **fault_kwargs,
                    instruction_name=instruction_name, seed=episode_seed, split=args.split,
                )

                if success:
                    dataset.save_episode()
                    # timing_rows is only persisted HERE, after save_episode()
                    # succeeds -- episode_index is only known now (LeRobot
                    # assigns it in save order, 0-based, matching saved_episodes
                    # before incrementing). A failed episode's timing_rows is
                    # simply never written -- garbage-collected along with the
                    # rest of that attempt's local state, exactly mirroring
                    # clear_episode_buffer()'s discard of the LeRobot buffer.
                    episode_index = saved_episodes
                    for row in timing_rows:
                        timing_file.write(json.dumps({"episode_index": episode_index, **row}) + "\n")
                    timing_file.flush()
                    saved_episodes += 1
                    success_count += 1
                else:
                    dataset.clear_episode_buffer()

                manifest_file.write(json.dumps({
                    "attempt": attempt,
                    "split": args.split,
                    "position_name": position_name,
                    "position": position,
                    "instruction_name": instruction_name,
                    "instruction": instruction,
                    "seed": episode_seed,
                    "success": success,
                    "final_status": final_status,
                    "final_phase": final_phase,
                    "num_frames": num_frames,
                    "saved": success,
                }) + "\n")
                manifest_file.flush()

                success_rate = success_count / attempt
                print(
                    f"[attempt {attempt:04d}] pos={position_name:<13} instr={instruction_name:<10} "
                    f"success={success} status={final_status:<10} frames={num_frames:3d} "
                    f"success_rate={success_rate:.2%} saved={saved_episodes}/{args.episodes}"
                )
    finally:
        dataset.finalize()

    print()
    print(
        f"=== Done: {saved_episodes} episodes saved, {attempt} attempts, "
        f"success_rate={(success_count / attempt) if attempt else 0:.2%} ==="
    )
    print(f"Dataset root: {root}")
    print(f"Manifest: {manifest_path}")
    print(f"Frame timing: {frame_timing_path}")
    if saved_episodes < args.episodes:
        print(f"WARNING: only {saved_episodes}/{args.episodes} requested episodes were saved (hit --max-attempts={max_attempts}).")


if __name__ == "__main__":
    main()
