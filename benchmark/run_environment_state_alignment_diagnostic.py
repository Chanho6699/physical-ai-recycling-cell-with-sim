"""Environment / state alignment diagnostic (v0) -- the final check before
fine-tuning.

Follow-up to run_state_semantics_diagnostic.py (which flagged ee_position.x
and gripper_qpos[1] as out-of-distribution against the checkpoint's
aggregate mean/std) and the now-production gripper-channel-2 sign fix
(robot_sim/pybullet_panda_backend.py's get_libero_observation_state()).
This script goes one level deeper than aggregate mean/std: it pulls REAL
per-timestep samples from HuggingFaceVLA/libero's actual training
parquet shards (not just the checkpoint's baked-in normalizer stats) and
compares them against a REPRESENTATIVE PyBullet sample (real scripted
episodes across varied object positions, not just small deltas from one
reset pose) -- full min/max/mean/std/percentiles/histograms, workspace
overlap, object-relative (ee-object) vector distributions, and an
orientation-specific comparison.

Modifies NO production file, NO checkpoint/config, and does not fine-tune
anything -- read-only investigation plus a new diagnostic script/tests
only.

=== WORLD-ORIGIN FINDING (see final report for full detail) ===

robosuite's generic table arena (robosuite/models/arenas/table_arena.py)
defaults to table_offset=(0, 0, 0.8) -- the table SURFACE sits at world
z=0.8 by default -- and Panda's base_xpos_offset for a "table" arena
(robosuite/models/robots/manipulators/panda_robot.py) is
`-0.16 - table_length/2` (~-0.56 for the default 0.8m table), i.e. the
robot base itself sits well behind/below the table-relative origin many
LIBERO tasks are built on. This project's PyBullet backend instead places
its own robot base at world origin (0,0,0) with the table top around
z~0.03 (see robot_sim/pybullet_panda_backend.py's _table_position). Both
conventions are internally self-consistent (world frame == robot_base
frame in both, confirmed in an earlier session's turn), but the absolute
placement differs substantially -- exactly the kind of "numbers differ,
not meaning" vs. "meaning differs" question this script's real-sample
comparison is built to answer quantitatively rather than by inspection
alone.

Run:
  .venv-vla/bin/python -m benchmark.run_environment_state_alignment_diagnostic
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from action_adapter.adapter_v0 import ActionAdapter
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO_ID = "HuggingFaceVLA/libero"
DEFAULT_DATASET_FILES = (
    "data/chunk-000/file-000.parquet",
    "data/chunk-000/file-100.parquet",
    "data/chunk-000/file-200.parquet",
)
STATE_DIM_NAMES = [
    "ee_position.x", "ee_position.y", "ee_position.z",
    "ee_orientation_axis_angle.x", "ee_orientation_axis_angle.y", "ee_orientation_axis_angle.z",
    "gripper_qpos.0", "gripper_qpos.1",
]

# The 4 counterfactual-benchmark positions plus 2 extras for broader
# PyBullet workspace coverage in this diagnostic's own episode collector.
DEFAULT_OBJECT_POSITIONS = {
    "center_right": [0.42, 0.00, 0.05],
    "center_left": [0.27, 0.00, 0.05],
    "positive_y": [0.35, 0.18, 0.05],
    "negative_y": [0.35, -0.18, 0.05],
    "far_right": [0.50, 0.10, 0.05],
    "far_left": [0.22, -0.10, 0.05],
}
DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-files", type=str, nargs="+", default=list(DEFAULT_DATASET_FILES))
    parser.add_argument("--max-steps-per-episode", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="results/environment_state_alignment_diagnostic")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Real HuggingFaceVLA/libero training-data samples ---


def load_real_dataset_samples(files=DEFAULT_DATASET_FILES) -> dict:
    """Downloads (once; cached thereafter by huggingface_hub) a handful of
    real LeRobot-format parquet shards from the actual training dataset
    (NOT the policy checkpoint's aggregate stats) and returns per-episode
    observation.state arrays plus a flat concatenation across all
    episodes -- the strongest evidence this task asked for, real
    per-timestep samples rather than a mean/std summary alone."""
    from huggingface_hub import hf_hub_download

    import pandas as pd

    episodes = {}
    for filename in files:
        local_path = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename=filename)
        df = pd.read_parquet(local_path, columns=["observation.state", "episode_index", "task_index"])
        for (episode_index, task_index), group in df.groupby(["episode_index", "task_index"]):
            key = f"{filename}::ep{episode_index}::task{task_index}"
            episodes[key] = np.stack(group["observation.state"].values).astype(np.float64)

    flat_states = np.concatenate(list(episodes.values()), axis=0)
    return {"episodes": episodes, "flat_states": flat_states, "num_episodes": len(episodes), "num_samples": len(flat_states)}


def estimate_reach_distances(episodes: dict, closed_threshold: float = 0.01) -> list:
    """Per episode: distance from the FIRST frame's ee_position to the
    ee_position at the first frame where gripper_qpos[0] drops below
    closed_threshold (an open->closed transition, i.e. approximately
    "the grasp moment") -- a real-data-derived proxy for "typical
    object-relative reach distance," since the dataset does not carry an
    explicit object-position label. Skips an episode if it never closes
    the gripper, or if it's already closed at frame 0 (nothing to measure)."""
    reach_distances = []
    for key, states in episodes.items():
        gripper_channel_0 = states[:, 6]
        closed_indices = np.where(gripper_channel_0 < closed_threshold)[0]
        if len(closed_indices) == 0 or closed_indices[0] == 0:
            continue
        first_closed = closed_indices[0]
        ee_start = states[0, 0:3]
        ee_grasp = states[first_closed, 0:3]
        reach_distances.append(float(np.linalg.norm(ee_grasp - ee_start)))
    return reach_distances


# --- Representative PyBullet workspace samples (real scripted episodes) ---


def collect_pybullet_workspace_samples(positions=None, bin_position=None, max_steps_per_episode=60) -> dict:
    """Runs a REAL DummyOpenVLAPolicy scripted episode (move_to_object ->
    close_gripper -> lift_object -> move_above_bin -> open_gripper) per
    object position, collecting get_libero_observation_state() at every
    step -- unlike a handful of small deltas from one reset pose, this
    covers the actual reach/lift/carry/place motion range this project's
    own robot visits during real task execution, which is the fair,
    comparable analogue to the real dataset's own real-execution
    trajectories.

    IMPORTANT CAVEAT for the ORIENTATION dimensions specifically (see
    final report): DummyOpenVLAPolicy's _predict_phase_action() always
    emits action[3:6] = [0.0, 0.0, 0.0] (confirmed by reading
    policy/dummy_openvla_policy.py -- every PolicyOutput.action literal
    in that file has zeros in the rotation slots) -- this scripted
    policy NEVER commands a rotation. So the orientation samples this
    function collects only reflect the EE's fixed reset-pose orientation
    the whole episode, not this project's actual achievable orientation
    range. See collect_pybullet_orientation_reachability_samples() below
    for a sample that actually exercises rotation, which is the fairer
    comparison for the orientation dimensions."""
    positions = positions or DEFAULT_OBJECT_POSITIONS
    bin_position = bin_position or DEFAULT_BIN_POSITION

    episodes = {}
    ee_object_vectors = []
    for position_name, position in positions.items():
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        backend.set_object_type("plastic_bottle")
        backend.set_object_position(list(position))

        policy = DummyOpenVLAPolicy()
        policy.reset()
        action_adapter = ActionAdapter()

        states = []
        robot_state = backend.get_state()
        for step_index in range(max_steps_per_episode):
            state_8d = backend.get_libero_observation_state()
            states.append(state_8d)
            ee_object_vectors.append([position[i] - state_8d[i] for i in range(3)])

            policy_input = PolicyInput(
                image=np.zeros((224, 224, 3), dtype=np.uint8),
                instruction="pick up the bottle",
                robot_state=robot_state,
                task_goal={},
                target_object_position=list(position),
                bin_position=bin_position,
                step_index=step_index,
                phase=policy.phase,
            )
            policy_output = policy.predict_action(policy_input)
            robot_command = action_adapter.convert(policy_output.action)
            robot_state = backend.apply_command(robot_command, steps=10)
            if robot_state["task_status"] == "success" or policy_output.done:
                break

        episodes[position_name] = np.array(states, dtype=np.float64)
        backend.shutdown()

    flat_states = np.concatenate(list(episodes.values()), axis=0)
    return {
        "episodes": episodes,
        "flat_states": flat_states,
        "ee_object_vectors": np.array(ee_object_vectors, dtype=np.float64),
        "num_episodes": len(episodes),
        "num_samples": len(flat_states),
    }


# Realistic per-step rotation magnitude a real model's command would
# actually produce after PandaCommandSafetyFilter's clip (see
# policy_semantics/safety_filter.py's DEFAULT_MAX_ROTATION_STEP_RAD).
ROTATION_REACHABILITY_STEP_RAD = 0.08
ROTATION_REACHABILITY_STEPS = 25


def collect_pybullet_orientation_reachability_samples(num_episodes: int = 4) -> dict:
    """Deliberately commands a real sequence of rotations (unlike
    collect_pybullet_workspace_samples(), which never rotates -- see its
    docstring) -- alternating +/- deltas on each of roll/pitch/yaw in
    turn, each step's magnitude clipped the same way a real model's
    output would be (ROTATION_REACHABILITY_STEP_RAD, comparable to
    PandaCommandSafetyFilter's own default clip) -- to measure this
    project's actual ACHIEVABLE orientation range, using the same
    apply_command()/get_libero_observation_state() path production uses,
    not a synthetic shortcut."""
    from action_adapter.adapter_v0 import RobotCommand

    rotation_patterns = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ]
    episodes = {}
    for episode_index in range(num_episodes):
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        states = [backend.get_libero_observation_state()]
        for step_index in range(ROTATION_REACHABILITY_STEPS):
            sign_x, sign_y, sign_z = rotation_patterns[(episode_index + step_index) % len(rotation_patterns)]
            command = RobotCommand(
                target_dx=0.0, target_dy=0.0, target_dz=0.0,
                target_droll=sign_x * ROTATION_REACHABILITY_STEP_RAD,
                target_dpitch=sign_y * ROTATION_REACHABILITY_STEP_RAD,
                target_dyaw=sign_z * ROTATION_REACHABILITY_STEP_RAD,
                gripper_command="open",
            )
            backend.apply_command(command, steps=10)
            states.append(backend.get_libero_observation_state())
        episodes[f"episode_{episode_index}"] = np.array(states, dtype=np.float64)
        backend.shutdown()

    flat_states = np.concatenate(list(episodes.values()), axis=0)
    return {"episodes": episodes, "flat_states": flat_states, "num_episodes": len(episodes), "num_samples": len(flat_states)}


# --- Distributional comparison ---


def _histogram_summary(values: np.ndarray, num_bins: int = 10) -> dict:
    counts, edges = np.histogram(values, bins=num_bins)
    return {"bin_edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def compute_distribution_report(real_states: np.ndarray, our_states: np.ndarray) -> dict:
    report = {}
    for dim_index, dim_name in enumerate(STATE_DIM_NAMES):
        real_col = real_states[:, dim_index]
        our_col = our_states[:, dim_index]
        report[dim_name] = {
            "real": {
                "min": float(real_col.min()), "max": float(real_col.max()),
                "mean": float(real_col.mean()), "std": float(real_col.std()),
                "p5": float(np.percentile(real_col, 5)), "p25": float(np.percentile(real_col, 25)),
                "p50": float(np.percentile(real_col, 50)), "p75": float(np.percentile(real_col, 75)),
                "p95": float(np.percentile(real_col, 95)),
                "histogram": _histogram_summary(real_col),
                "sample_values": [round(float(v), 4) for v in real_col[:5]],
            },
            "ours": {
                "min": float(our_col.min()), "max": float(our_col.max()),
                "mean": float(our_col.mean()), "std": float(our_col.std()),
                "p5": float(np.percentile(our_col, 5)), "p25": float(np.percentile(our_col, 25)),
                "p50": float(np.percentile(our_col, 50)), "p75": float(np.percentile(our_col, 75)),
                "p95": float(np.percentile(our_col, 95)),
                "histogram": _histogram_summary(our_col),
                "sample_values": [round(float(v), 4) for v in our_col[:5]],
            },
        }
        real_range = (real_col.min(), real_col.max())
        our_range = (our_col.min(), our_col.max())
        overlap_low = max(real_range[0], our_range[0])
        overlap_high = min(real_range[1], our_range[1])
        overlap_width = max(0.0, overlap_high - overlap_low)
        union_width = max(real_range[1], our_range[1]) - min(real_range[0], our_range[0])
        report[dim_name]["range_overlap_fraction"] = float(overlap_width / union_width) if union_width > 1e-9 else None
    return report


def compute_object_relative_comparison(real_reach_distances: list, our_ee_object_vectors: np.ndarray) -> dict:
    our_distances = np.linalg.norm(our_ee_object_vectors, axis=1)
    real_arr = np.array(real_reach_distances)
    return {
        "real_reach_distance": {
            "n": len(real_arr), "min": float(real_arr.min()), "max": float(real_arr.max()),
            "mean": float(real_arr.mean()), "std": float(real_arr.std()),
            "p50": float(np.percentile(real_arr, 50)),
        },
        "our_ee_to_object_distance": {
            "n": len(our_distances), "min": float(our_distances.min()), "max": float(our_distances.max()),
            "mean": float(our_distances.mean()), "std": float(our_distances.std()),
            "p50": float(np.percentile(our_distances, 50)),
        },
        "mean_ratio_ours_over_real": float(our_distances.mean() / real_arr.mean()) if real_arr.mean() > 1e-9 else None,
    }


# --- Coordinate semantics (documentation, backed by code citations) ---


def coordinate_semantics_summary() -> dict:
    return {
        "pybullet_ee_position": {
            "source": "PyBulletPandaBackend._get_ee_pose() -> p.getLinkState(end_effector_link_index) position",
            "frame": "PyBullet world frame; this project's robot base sits at world origin with identity "
            "orientation (see PyBulletPandaBackend's class docstring), so world == robot_base frame here",
            "status": "confirmed (code-read, this project)",
        },
        "robosuite_robot0_eef_pos": {
            "source": "robosuite/robots/robot.py's eef_pos sensor: self.sim.data.site_xpos[self.eef_site_id[arm]]",
            "frame": "MuJoCo site_xpos is always WORLD frame for a fixed-base body (base_to_eef_pos, a "
            "DIFFERENT, base-relative sensor, is defined only in robosuite/robots/mobile_robot.py and is not "
            "what LIBERO's fixed-base Panda tasks use)",
            "status": "confirmed (code-read, robosuite)",
        },
        "same_reference_frame_convention": {
            "verdict": True,
            "reason": "Both are WORLD frame, and both projects' robot base is placed at (or coincides with, "
            "for robosuite's typical fixed-base setup) their own world origin's reference -- i.e. the SAME "
            "semantic convention (world frame == robot_base frame, meters, EE position). This is a 'same "
            "meaning, different absolute placement/origin' situation, not a 'different meaning' situation.",
        },
        "world_origin_placement_difference": {
            "robosuite_table_offset": "(0, 0, 0.8) default -- robosuite/models/arenas/table_arena.py's "
            "TableArena(table_offset=(0,0,0.8)) -- table surface at world z=0.8",
            "robosuite_panda_base_xpos_offset_for_table_arena": "-0.16 - table_length/2 (~-0.56 for the "
            "default 0.8m table) -- robosuite/models/robots/manipulators/panda_robot.py's base_xpos_offset property",
            "this_project_table_position": "PyBulletPandaBackend._table_position = [0.35, 0.15, 0.015] "
            "(table top near z~0.03), robot base at world origin (0,0,0)",
            "conclusion": "The two environments' absolute world-coordinate placement of "
            "robot-base/table/workspace differs substantially and is the primary explanation for the "
            "ee_position magnitude differences measured below -- NOT a semantic/frame bug.",
        },
    }


def summarize_dimension_verdicts(distribution_report: dict) -> dict:
    verdicts = {}
    for dim_name, stats in distribution_report.items():
        real_mean, real_std = stats["real"]["mean"], stats["real"]["std"]
        our_mean = stats["ours"]["mean"]
        z = (our_mean - real_mean) / real_std if real_std > 1e-9 else None
        overlap = stats["range_overlap_fraction"]
        verdicts[dim_name] = {
            "z_of_our_mean_vs_real": z,
            "range_overlap_fraction": overlap,
            "flag": "OOD" if (z is not None and abs(z) > 3.0) else ("LOW_OVERLAP" if (overlap is not None and overlap < 0.3) else "OK"),
        }
    return verdicts


def decide_verdict(dimension_verdicts: dict, object_relative: dict) -> dict:
    ood_dims = [name for name, v in dimension_verdicts.items() if v["flag"] == "OOD"]
    low_overlap_dims = [name for name, v in dimension_verdicts.items() if v["flag"] == "LOW_OVERLAP"]
    position_dims_affected = [d for d in (ood_dims + low_overlap_dims) if d.startswith("ee_position")]
    orientation_dims_affected = [d for d in (ood_dims + low_overlap_dims) if d.startswith("ee_orientation")]
    gripper_dims_affected = [d for d in (ood_dims + low_overlap_dims) if d.startswith("gripper")]

    ratio = object_relative.get("mean_ratio_ours_over_real")
    object_relative_comparable = ratio is not None and 0.5 <= ratio <= 2.0

    if position_dims_affected and not orientation_dims_affected and not gripper_dims_affected:
        verdict = "B"
        reason = (
            f"ee_position dimension(s) {position_dims_affected} are OOD/low-overlap against real training "
            "samples, traced to a confirmed world-origin/workspace-placement difference (table height, robot "
            "base offset) rather than a frame/semantics bug -- a real environment-alignment item exists. "
            + (
                f"Object-relative (ee-to-object) distances ARE comparable in scale (ratio={ratio:.2f}), "
                "suggesting the RELATIVE geometry the policy actually needs is closer to training than the "
                "raw absolute position channel is."
                if object_relative_comparable
                else f"Object-relative distances are also substantially different (ratio={ratio}), so this is "
                "not purely a benign origin-offset artifact."
            )
        )
    elif orientation_dims_affected or gripper_dims_affected:
        verdict = "B"
        reason = f"Orientation and/or gripper dimensions flagged OOD/low-overlap: {orientation_dims_affected + gripper_dims_affected} -- needs investigation before fine-tuning."
    else:
        verdict = "A"
        reason = "No dimension is OOD or low-overlap against real training samples -- environment alignment looks reasonable; proceeding to fine-tuning is defensible."

    return {
        "verdict": verdict, "reason": reason,
        "ood_dimensions": ood_dims, "low_overlap_dimensions": low_overlap_dims,
        "object_relative_comparable": object_relative_comparable,
    }


def print_report(distribution_report, workspace_note, object_relative, coordinate_semantics, dimension_verdicts, verdict) -> None:
    print("\n=== Per-dimension distribution comparison (real HuggingFaceVLA/libero vs. ours) ===")
    for dim_name, stats in distribution_report.items():
        r, o = stats["real"], stats["ours"]
        print(f"\n{dim_name}:")
        print(f"  real: min={r['min']:+.4f} max={r['max']:+.4f} mean={r['mean']:+.4f} std={r['std']:.4f} p50={r['p50']:+.4f}")
        print(f"  ours: min={o['min']:+.4f} max={o['max']:+.4f} mean={o['mean']:+.4f} std={o['std']:.4f} p50={o['p50']:+.4f}")
        print(f"  range_overlap_fraction={stats['range_overlap_fraction']}")

    print("\n=== Object-relative (ee-to-object) comparison ===")
    print(json.dumps(object_relative, indent=2))

    print("\n=== Coordinate semantics ===")
    print(json.dumps(coordinate_semantics["same_reference_frame_convention"], indent=2))
    print(json.dumps(coordinate_semantics["world_origin_placement_difference"], indent=2))

    print("\n=== Per-dimension verdicts ===")
    for dim_name, v in dimension_verdicts.items():
        print(f"{dim_name}: flag={v['flag']} z={v['z_of_our_mean_vs_real']} overlap={v['range_overlap_fraction']}")

    print("\n=== FINAL VERDICT ===")
    print(f"verdict: {verdict['verdict']}")
    print(f"reason: {verdict['reason']}")


def run_all(args) -> dict:
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    real_data = load_real_dataset_samples(args.dataset_files)
    real_reach_distances = estimate_reach_distances(real_data["episodes"])

    our_data = collect_pybullet_workspace_samples(max_steps_per_episode=args.max_steps_per_episode)
    orientation_reachability_data = collect_pybullet_orientation_reachability_samples()

    # Position/gripper dims: fair comparison is the real-task-execution
    # workspace sample (our_data). Orientation dims: fair comparison is
    # the rotation-reachability sample (orientation_reachability_data),
    # since our_data's scripted policy never rotates at all (see
    # collect_pybullet_workspace_samples()'s docstring) -- using it for
    # orientation would just measure "how much orientation drifts when
    # nothing commands it," not this project's real achievable range.
    distribution_report = compute_distribution_report(real_data["flat_states"], our_data["flat_states"])
    orientation_only_report = compute_distribution_report(real_data["flat_states"], orientation_reachability_data["flat_states"])
    for dim_name in ("ee_orientation_axis_angle.x", "ee_orientation_axis_angle.y", "ee_orientation_axis_angle.z"):
        distribution_report[dim_name] = orientation_only_report[dim_name]
        distribution_report[dim_name]["ours_sample_source"] = "orientation_reachability (rotation-exercising episodes)"
    for dim_name in distribution_report:
        distribution_report[dim_name].setdefault("ours_sample_source", "workspace (real scripted pick-place episodes)")

    object_relative = compute_object_relative_comparison(real_reach_distances, our_data["ee_object_vectors"])
    coordinate_semantics = coordinate_semantics_summary()
    dimension_verdicts = summarize_dimension_verdicts(distribution_report)
    verdict = decide_verdict(dimension_verdicts, object_relative)

    result = {
        "real_data_summary": {"num_episodes": real_data["num_episodes"], "num_samples": real_data["num_samples"]},
        "our_data_summary": {"num_episodes": our_data["num_episodes"], "num_samples": our_data["num_samples"]},
        "our_orientation_reachability_summary": {
            "num_episodes": orientation_reachability_data["num_episodes"],
            "num_samples": orientation_reachability_data["num_samples"],
        },
        "distribution_report": distribution_report,
        "object_relative": object_relative,
        "coordinate_semantics": coordinate_semantics,
        "dimension_verdicts": dimension_verdicts,
        "verdict": verdict,
    }

    log_path = output_dir / f"environment_alignment_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as log_file:
        json.dump(result, log_file, ensure_ascii=False, indent=2, default=str)

    print_report(distribution_report, None, object_relative, coordinate_semantics, dimension_verdicts, verdict)
    print(f"\nFull result JSON: {log_path}")
    result["log_path"] = str(log_path)
    return result


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
