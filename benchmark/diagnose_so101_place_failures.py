"""SO-101 place_outside_target diagnosis (see this task's chat report,
"place_outside_target 원인 진단"). Does NOT change any waypoint,
transport delta, threshold, or gripper value -- run_pick_and_place_episode()
is called with its existing, unmodified default behavior (no
extra_settle_steps needed here; this script instruments via the
already-supported `on_step` callback only, purely to observe joint
tracking error, without altering what gets applied to the backend).

For each seed, records the full transport/place coordinate chain
(initial object position -> after grasp -> after lift -> before
transport -> intended transport target -> after transport -> intended
place-descend target -> after place-descend -> before release -> after
release -> final settled position) and decomposes the final xy error
into:
  - how much of it was already present BEFORE release (targeting/IK
    error accumulated through transport + place_descend), vs
  - how much was added by settle-phase drift AFTER release.

Also reports, per phase, the EE target-vs-actual error (already
computed by so101_scripted_expert.move_to_target()'s own "error" key),
the grasped-object-to-EE relative drift (already computed as
"max_relative_drift_m"), and a NEW (diagnostic-only, local to this
script) per-phase joint-target-vs-actual-joint-state tracking error
computed via the existing on_step hook -- no change to
so101_scripted_expert.py's control flow.

Reuses benchmark.evaluate_so101_expert_small_randomization's own
sample_object_position().

Run:
  .venv-vla/bin/python -m benchmark.diagnose_so101_place_failures
"""

import argparse
import json
import math
from pathlib import Path

from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    TRANSPORT_DELTA_XY,
    sample_object_position,
)
from benchmark.so101_scripted_expert import (
    PHASE_LIFT,
    PHASE_PLACE_DESCEND,
    PHASE_TRANSPORT,
    TARGET_XY_ERROR_PASS_M,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_place_diagnosis.json"

FAILED_SEEDS = [0, 6]
COMPARISON_SUCCESS_SEEDS = [2, 3, 4]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def xy_distance(a: list, b: list) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class JointTrackingTracker:
    """Diagnostic-only, local to this script -- uses the ALREADY
    supported on_step(phase, arm_joint_targets, gripper_target) hook
    (so101_scripted_expert.py is not modified for this). For each
    phase, compares the joint target recorded on one step against the
    ACTUAL joint state observed at the START of the next step in that
    same phase (i.e. after settle_steps physics steps have run) -- a
    per-step joint tracking error, not fabricated, not fed back into
    control."""

    def __init__(self, backend: So101PyBulletBackend):
        self.backend = backend
        self._last_target_by_phase = {}
        self.errors_by_phase = {}

    def on_step(self, phase: str, arm_joint_targets: list, gripper_target_normalized: float) -> None:
        current = self.backend.get_joint_positions()
        if phase in self._last_target_by_phase:
            prev_target = self._last_target_by_phase[phase]
            error = max(abs(c - t) for c, t in zip(current, prev_target))
            self.errors_by_phase.setdefault(phase, []).append(error)
        self._last_target_by_phase[phase] = list(arm_joint_targets)

    def summary(self, phase: str) -> dict:
        errors = self.errors_by_phase.get(phase, [])
        return {
            "max_m": max(errors) if errors else None,
            "mean_m": (sum(errors) / len(errors)) if errors else None,
            "num_samples": len(errors),
        }


def diagnose_seed(seed: int) -> dict:
    object_position = sample_object_position(seed, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
    backend = So101PyBulletBackend(gui=False, object_position=object_position)
    try:
        backend.reset()
        initial_object_position, _ = backend.get_object_pose()
        target_zone_center_xy = backend.get_scene_state()["target_zone_center_xy"]

        tracker = JointTrackingTracker(backend)
        try:
            result = run_pick_and_place_episode(backend, TRANSPORT_DELTA_XY, on_step=tracker.on_step)
        except So101ExpertError as exc:
            return {
                "seed": seed, "object_position": initial_object_position, "pre_release_failure": True,
                "failure_reason": exc.failure_reason, "failure_phase": exc.phase,
            }

        lift = result["lift"]
        transport = result["transport"]
        place_descend = result["place_descend"]

        object_after_grasp = lift["object_start_position"]
        object_after_lift = transport["object_start_position"]
        ee_before_transport = lift["final_ee_position"]
        intended_transport_target = transport["target"]
        ee_after_transport = transport["final_ee_position"]
        object_after_transport = place_descend["object_start_position"]
        intended_place_descend_target = place_descend["target"]
        ee_after_place_descend = place_descend["final_ee_position"]
        object_before_release = result["object_release_position"]
        object_after_release = result["object_position_immediately_after_release"]
        final_object_position = result["object_final_position"]

        # transport_delta_xy is applied as: intended_transport_target =
        # ee_before_transport + transport_delta_xy (see
        # so101_scripted_expert.py's own run_pick_and_place_episode()) --
        # verified numerically below, not assumed.
        transport_delta_applied = [
            intended_transport_target[0] - ee_before_transport[0],
            intended_transport_target[1] - ee_before_transport[1],
        ]

        # Structural check: target_zone_center_xy = initial_object_position_xy
        # + target_zone_offset_xy, and (by this project's own design note)
        # target_zone_offset_xy == transport_delta_xy == [0.05, 0.05]. So IF
        # the EE tracked the object with zero error through
        # pre_grasp/approach/grasp/lift, intended_transport_target would
        # automatically equal target_zone_center_xy regardless of object
        # start position -- this quantifies how much that self-compensating
        # structure actually held, given REAL (imperfect) tracking.
        intended_transport_target_vs_target_zone_center_error_m = xy_distance(intended_transport_target, target_zone_center_xy)

        release_xy_error_from_target_m = xy_distance(object_before_release, target_zone_center_xy)
        post_release_drift_xy_m = xy_distance(final_object_position, object_after_release)
        final_xy_error_m = result["object_target_xy_error_m"]

        return {
            "seed": seed,
            "pre_release_failure": False,
            "place_success": result["place_success"],
            "failure_reason": result["failure_reason"],
            "coordinates": {
                "initial_object_position": initial_object_position,
                "object_after_grasp": object_after_grasp,
                "object_after_lift": object_after_lift,
                "ee_before_transport": ee_before_transport,
                "intended_transport_target": intended_transport_target,
                "ee_after_transport": ee_after_transport,
                "object_after_transport": object_after_transport,
                "intended_place_descend_target": intended_place_descend_target,
                "ee_after_place_descend": ee_after_place_descend,
                "object_before_release": object_before_release,
                "object_after_release": object_after_release,
                "final_object_position": final_object_position,
                "target_zone_center_xy": target_zone_center_xy,
            },
            "transport_delta_xy_configured": TRANSPORT_DELTA_XY,
            "transport_delta_xy_actually_applied": transport_delta_applied,
            "errors": {
                "lift_ee_target_vs_actual_m": lift["error"],
                "transport_ee_target_vs_actual_m": transport["error"],
                "place_descend_ee_target_vs_actual_m": place_descend["error"],
                "lift_grasped_object_relative_drift_m": lift["max_relative_drift_m"],
                "transport_grasped_object_relative_drift_m": transport["max_relative_drift_m"],
                "place_descend_grasped_object_relative_drift_m": place_descend["max_relative_drift_m"],
                "joint_tracking_error_lift": tracker.summary(PHASE_LIFT),
                "joint_tracking_error_transport": tracker.summary(PHASE_TRANSPORT),
                "joint_tracking_error_place_descend": tracker.summary(PHASE_PLACE_DESCEND),
                "intended_transport_target_vs_target_zone_center_error_m": intended_transport_target_vs_target_zone_center_error_m,
                "release_xy_error_from_target_m": release_xy_error_from_target_m,
                "post_release_drift_xy_m": post_release_drift_xy_m,
                "final_xy_error_m": final_xy_error_m,
            },
        }
    finally:
        backend.close()


def summarize(diagnoses: list) -> dict:
    with_data = [d for d in diagnoses if not d.get("pre_release_failure")]

    def avg(values):
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else None

    per_seed = {}
    for d in with_data:
        e = d["errors"]
        per_seed[str(d["seed"])] = {
            "final_xy_error_m": e["final_xy_error_m"],
            "release_xy_error_from_target_m": e["release_xy_error_from_target_m"],
            "post_release_drift_xy_m": e["post_release_drift_xy_m"],
            "transport_ee_target_vs_actual_m": e["transport_ee_target_vs_actual_m"],
            "place_descend_ee_target_vs_actual_m": e["place_descend_ee_target_vs_actual_m"],
            "intended_transport_target_vs_target_zone_center_error_m": e["intended_transport_target_vs_target_zone_center_error_m"],
            "error_present_before_release": e["release_xy_error_from_target_m"] > TARGET_XY_ERROR_PASS_M,
        }

    return {
        "per_seed_error_breakdown": per_seed,
        "avg_release_xy_error_from_target_m_failed": avg(
            [d["errors"]["release_xy_error_from_target_m"] for d in with_data if d["seed"] in FAILED_SEEDS]
        ),
        "avg_release_xy_error_from_target_m_success": avg(
            [d["errors"]["release_xy_error_from_target_m"] for d in with_data if d["seed"] in COMPARISON_SUCCESS_SEEDS]
        ),
        "avg_post_release_drift_xy_m_failed": avg(
            [d["errors"]["post_release_drift_xy_m"] for d in with_data if d["seed"] in FAILED_SEEDS]
        ),
        "avg_post_release_drift_xy_m_success": avg(
            [d["errors"]["post_release_drift_xy_m"] for d in with_data if d["seed"] in COMPARISON_SUCCESS_SEEDS]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    seeds = FAILED_SEEDS + COMPARISON_SUCCESS_SEEDS
    diagnoses = []
    for seed in seeds:
        d = diagnose_seed(seed)
        diagnoses.append(d)
        if d.get("pre_release_failure"):
            print(f"[seed {seed}] PRE-RELEASE FAILURE ({d['failure_reason']} @ {d['failure_phase']})")
        else:
            e = d["errors"]
            print(
                f"[seed {seed}] place_success={d['place_success']} final_xy_err={e['final_xy_error_m']:.4f} "
                f"release_xy_err={e['release_xy_error_from_target_m']:.4f} post_release_drift={e['post_release_drift_xy_m']:.4f} "
                f"transport_ee_err={e['transport_ee_target_vs_actual_m']:.4f} place_descend_ee_err={e['place_descend_ee_target_vs_actual_m']:.4f}"
            )

    summary = summarize(diagnoses)

    output = {
        "config": {"failed_seeds": FAILED_SEEDS, "comparison_success_seeds": COMPARISON_SUCCESS_SEEDS, "transport_delta_xy": TRANSPORT_DELTA_XY},
        "diagnoses": diagnoses,
        "summary": summary,
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== place_outside_target diagnosis summary ===")
    print(f"per_seed_error_breakdown: {json.dumps(summary['per_seed_error_breakdown'], indent=2, default=str)}")
    print(f"avg_release_xy_error_from_target_m (failed seeds 0,6): {summary['avg_release_xy_error_from_target_m_failed']}")
    print(f"avg_release_xy_error_from_target_m (success seeds 2,3,4): {summary['avg_release_xy_error_from_target_m_success']}")
    print(f"avg_post_release_drift_xy_m (failed seeds 0,6): {summary['avg_post_release_drift_xy_m_failed']}")
    print(f"avg_post_release_drift_xy_m (success seeds 2,3,4): {summary['avg_post_release_drift_xy_m_success']}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
