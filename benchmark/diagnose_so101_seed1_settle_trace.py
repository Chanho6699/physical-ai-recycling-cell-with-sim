"""SO-101 seed-1 precise settle diagnosis (see this task's chat report,
"seed 1 정밀 진단"). Under the OLD fixed-360-step single-instant
check, seed 1 was classified settle_failed; a subsequent diagnostic
extension (extra_settle_steps, now superseded) showed a large,
isolated angular-velocity spike at step 1080 (0.09 -> 2.81 rad/s) while
object height/position barely moved -- this script records EVERY
physics step of seed 1's release-settle phase (via
run_pick_and_place_episode()'s new `record_settle_trace=True` option,
which now ALSO records PyBullet contact-point count and max contact
normal force per step -- a pure addition, not previously available) to
determine whether that spike was:

  A. a REAL physical rocking/contact event (position/orientation
     genuinely changing, contact forces genuinely shifting), or
  B. a single-frame NUMERICAL artifact (position/orientation
     essentially unchanged despite the reported velocity spike).

Does NOT special-case or filter out any spike in the judgment logic
itself -- run_pick_and_place_episode()'s own continuous-stability
requirement (120 consecutive passing steps) is used AS-IS; this script
only OBSERVES whether that requirement naturally absorbs/tolerates or
rejects the spike, per this task's explicit instruction not to add an
ad-hoc exception rule.

Run:
  .venv-vla/bin/python -m benchmark.diagnose_so101_seed1_settle_trace
"""

import argparse
import json
from pathlib import Path

from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    TRANSPORT_DELTA_XY,
    sample_object_position,
)
from benchmark.so101_scripted_expert import So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_seed1_settle_trace.json"
SEED = 1


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    object_position = sample_object_position(SEED, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
    backend = So101PyBulletBackend(gui=False, object_position=object_position)
    try:
        backend.reset()
        settled_object_position, _ = backend.get_object_pose()

        try:
            result = run_pick_and_place_episode(backend, TRANSPORT_DELTA_XY, record_settle_trace=True)
        except So101ExpertError as exc:
            print(f"seed {SEED} raised a pre-settle So101ExpertError: {exc.failure_reason} @ {exc.phase} -- no settle trace to analyze")
            return

        trace = result["settle_trace"]

        # Find the largest single-step-to-next-step jump in angular
        # speed anywhere in the trace, and inspect position/orientation
        # around it.
        biggest_jump = None
        for i in range(1, len(trace)):
            jump = trace[i]["angular_speed_radps"] - trace[i - 1]["angular_speed_radps"]
            if biggest_jump is None or jump > biggest_jump["jump"]:
                biggest_jump = {"jump": jump, "index": i, "before": trace[i - 1], "after": trace[i]}

        output = {
            "seed": SEED,
            "object_position": settled_object_position,
            "settle_success": result["settle_success"],
            "settle_steps_used": result["settle_steps_used"],
            "settle_timeout": result["settle_timeout"],
            "max_consecutive_stable_steps": result["max_consecutive_stable_steps"],
            "final_consecutive_stable_steps": result["final_consecutive_stable_steps"],
            "continuous_stable_steps_required": result["continuous_stable_steps_required"],
            "place_success": result["place_success"],
            "failure_reason": result["failure_reason"],
            "final_xy_error": result["final_xy_error"],
            "biggest_angular_speed_jump": biggest_jump,
            "full_trace": trace,
        }
    finally:
        backend.close()

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"=== seed {SEED} precise settle trace ===")
    print(f"settle_success: {output['settle_success']}  settle_timeout: {output['settle_timeout']}")
    print(f"settle_steps_used: {output['settle_steps_used']}  max_consecutive_stable_steps: {output['max_consecutive_stable_steps']}")
    print(f"place_success: {output['place_success']}  failure_reason: {output['failure_reason']}  final_xy_error: {output['final_xy_error']}")

    if biggest_jump:
        b, a = biggest_jump["before"], biggest_jump["after"]
        pos_delta = sum((a["object_position"][i] - b["object_position"][i]) ** 2 for i in range(3)) ** 0.5
        print(f"\nBiggest single-step angular speed jump: step {b['step']}->{a['step']}, "
              f"ang {b['angular_speed_radps']:.4f} -> {a['angular_speed_radps']:.4f} rad/s")
        print(f"  position delta over that single step: {pos_delta:.6f} m")
        print(f"  contact_count before/after: {b['contact_count']} / {a['contact_count']}")
        print(f"  max_contact_normal_force before/after: {b['max_contact_normal_force']:.4f} / {a['max_contact_normal_force']:.4f}")
        classification = "A_real_physical_event" if pos_delta > 0.001 else "B_single_frame_numerical_spike"
        print(f"  classification: {classification} (position moved {'>' if pos_delta>0.001 else '<='} 0.001m over that one step)")

    print(f"\nFull per-step trace ({len(output['full_trace'])} entries): {output_path}")


if __name__ == "__main__":
    main()
