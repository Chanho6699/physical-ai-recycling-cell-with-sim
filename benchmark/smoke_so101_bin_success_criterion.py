"""SO-101 bin production place_success criterion smoke test (see this
task's chat report, "use_bin=True일 때 실제 open-top bin 수거 결과를
기준으로 production place_success를 계산"). Does NOT touch expert
waypoints/clearances/bin geometry/settle thresholds, does NOT collect
a dataset, does NOT touch robot_sim/pybullet_panda_backend.py, does
NOT run the 20-seed benchmark.

Covers (see this task's own section 7):
  A. normal success -- a real use_bin=True episode with the production
     bin scene, place_success=True, failure_reason=None, every
     bin_success_debug condition True
  B. object outside bin -- controlled evaluate_bin_place_success()
     input (inside_inner_xy=False) -> place_success=False,
     failure_reason="object_outside_bin"
  C. object above rim -- controlled input
     (object_center_below_rim=False) -> failure_reason="object_not_below_rim"
  D. not settled -- controlled input (settle_success=False) ->
     failure_reason="settle_failed"
  E. flat compatibility -- a real use_bin=False episode still uses the
     EXISTING target_xy_error-based judgment, bin_success_debug is None

B/C/D deliberately call benchmark.so101_scripted_expert.evaluate_bin_place_success()
directly with a controlled dict (see this task's own "controlled
input으로 테스트할 수 있게 한다") rather than contriving a live physics
setup to force those specific outcomes -- the pure function's own
logic is exactly what a live episode would have called with real
inputs, so this is a faithful, non-fragile test of the same decision
logic.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_bin_success_criterion
"""

import json
from pathlib import Path

from benchmark.so101_scripted_expert import (
    FAILURE_OBJECT_NOT_BELOW_RIM,
    FAILURE_OBJECT_OUTSIDE_BIN,
    FAILURE_SETTLE_FAILED,
    evaluate_bin_place_success,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "bin_success_criterion_smoke.json"

# A fully-passing baseline dict -- each test below flips exactly ONE
# field away from this to isolate that single condition's effect (see
# this task's own priority-order requirement).
PASSING_BASELINE = {
    "layout_validation_passed": True, "object_separated": True, "inside_inner_xy": True,
    "object_center_below_rim": True, "object_top_below_rim": True, "settle_success": True,
    "manipulation_steps_completed": True, "place_waypoint_reached": True,
    "object_final_xyz": [0.42, 0.10, 0.074], "object_final_aabb": {"min": [0.40, 0.08, 0.054], "max": [0.44, 0.12, 0.094]},
    "bin_inner_bounds": {"x_min": 0.35, "x_max": 0.49, "y_min": 0.03, "y_max": 0.17}, "rim_z": 0.134,
    "center_rim_delta": -0.06, "top_rim_delta": -0.04, "failed_conditions": [],
}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def test_normal_success() -> dict:
    backend = So101PyBulletBackend(gui=False, use_bin=True)
    try:
        backend.reset()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        result = run_pick_and_place_episode(backend, transport_delta_xy)
        debug = result["bin_success_debug"]
        all_conditions_true = debug is not None and all(
            debug[k] for k in (
                "layout_validation_passed", "object_separated", "inside_inner_xy",
                "object_center_below_rim", "settle_success", "manipulation_steps_completed", "place_waypoint_reached",
            )
        )
        return {
            "place_success": result["place_success"], "failure_reason": result["failure_reason"],
            "bin_success_debug": debug, "all_conditions_true": all_conditions_true,
            "test_pass": result["place_success"] is True and result["failure_reason"] is None and all_conditions_true,
        }
    finally:
        backend.close()


def test_object_outside_bin() -> dict:
    controlled = {**PASSING_BASELINE, "inside_inner_xy": False, "failed_conditions": ["inside_inner_xy"]}
    place_success, failure_reason, failure_phase = evaluate_bin_place_success(controlled)
    return {
        "place_success": place_success, "failure_reason": failure_reason, "failure_phase": failure_phase,
        "test_pass": place_success is False and failure_reason == FAILURE_OBJECT_OUTSIDE_BIN,
    }


def test_object_above_rim() -> dict:
    controlled = {**PASSING_BASELINE, "object_center_below_rim": False, "failed_conditions": ["object_center_below_rim"]}
    place_success, failure_reason, failure_phase = evaluate_bin_place_success(controlled)
    return {
        "place_success": place_success, "failure_reason": failure_reason, "failure_phase": failure_phase,
        "test_pass": place_success is False and failure_reason == FAILURE_OBJECT_NOT_BELOW_RIM,
    }


def test_not_settled() -> dict:
    controlled = {**PASSING_BASELINE, "settle_success": False, "failed_conditions": ["settle_success"]}
    place_success, failure_reason, failure_phase = evaluate_bin_place_success(controlled)
    return {
        "place_success": place_success, "failure_reason": failure_reason, "failure_phase": failure_phase,
        "test_pass": place_success is False and failure_reason == FAILURE_SETTLE_FAILED,
    }


def test_flat_compatibility() -> dict:
    backend = So101PyBulletBackend(gui=False)  # use_bin defaults to False
    try:
        backend.reset()
        result = run_pick_and_place_episode(backend, [0.05, 0.05])
        return {
            "place_success": result["place_success"], "failure_reason": result["failure_reason"],
            "bin_success_debug_is_none": result["bin_success_debug"] is None,
            "object_target_xy_error_m": result["object_target_xy_error_m"],
            "test_pass": result["place_success"] is True and result["failure_reason"] is None and result["bin_success_debug"] is None,
        }
    finally:
        backend.close()


def main() -> None:
    crashed = False
    crash_reason = None
    results = {}
    try:
        results["A_normal_success"] = test_normal_success()
        results["B_object_outside_bin"] = test_object_outside_bin()
        results["C_object_above_rim"] = test_object_above_rim()
        results["D_not_settled"] = test_not_settled()
        results["E_flat_compatibility"] = test_flat_compatibility()
    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"

    overall_pass = not crashed and all(r.get("test_pass", False) for r in results.values())

    output = {"crashed": crashed, "crash_reason": crash_reason, "overall_pass": overall_pass, "results": results}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 bin production place_success criterion smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        for name, r in results.items():
            print(f"[{name}] test_pass={r['test_pass']}  place_success={r['place_success']}  failure_reason={r.get('failure_reason')}")
        print()
        print("A_normal_success bin_success_debug:", results["A_normal_success"]["bin_success_debug"])
        print("E_flat_compatibility object_target_xy_error_m:", results["E_flat_compatibility"]["object_target_xy_error_m"])
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {resolve(str(OUTPUT_PATH))}")


if __name__ == "__main__":
    main()
