"""SO-101 bin scene-layout validation smoke test (see this task's chat
report, "bin scene layout 문제를 production 수준 규칙으로 정리"). Does
NOT touch success judgment/failure_reason/settle logic, does NOT run
the 20-seed benchmark, does NOT collect a dataset, does NOT touch
robot_sim/pybullet_panda_backend.py.

Covers (see this task's own section 8):
  A. valid bin scene  -- use_bin=True, production bin default offset,
     validate_initial_scene_layout() passes, no object-bin overlap,
     object/table/bin bounds all sane
  B. invalid overlap   -- an offset too small for the bin's own size
     must raise InvalidSceneLayoutError BEFORE any expert phase runs,
     with the overlapping wall/AABB info in the error
  C. flat backward compatibility -- use_bin=False keeps the ORIGINAL
     flat default (target_zone_offset_xy=[0.05, 0.05],
     surface_footprint_xy=[0.15, 0.15]) untouched; the bin-specific
     defaults never leak into a flat scene
  D. explicit override -- a user-supplied target_zone_offset_xy takes
     priority over the bin default and still validates cleanly

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_bin_scene_layout
"""

import json
import math
from pathlib import Path

from robot_sim.so101_pybullet_backend import (
    DEFAULT_BIN_SURFACE_FOOTPRINT_XY,
    DEFAULT_BIN_TARGET_ZONE_OFFSET_XY,
    DEFAULT_OBJECT_POSITION,
    DEFAULT_SCENE_CONFIG,
    InvalidSceneLayoutError,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "bin_scene_layout_smoke.json"

# Deliberately too small to clear the bin's own outer half-width
# (0.074m) + the object's own half-extent (0.02m) = 0.094m minimum on
# at least one axis -- matches the flat marker's OLD default, which is
# exactly the overlap this whole task exists to catch.
INVALID_OFFSET_TOO_SMALL = [0.05, 0.05]

# A genuinely valid, USER-CHOSEN alternative to the production bin
# default -- chosen independently from DEFAULT_BIN_TARGET_ZONE_OFFSET_XY
# so this test actually exercises "a DIFFERENT valid value the user
# picked", not just the production default under another name. Must
# satisfy BOTH: (a) clear the object-bin overlap on the y axis alone
# (offset_y > bin_outer_half(0.074) + object_half(0.02) = 0.094), and
# (b) keep the bin's own outer footprint inside the bin-default table
# (offset_y + 0.074 <= table_half(0.19), i.e. offset_y <= 0.116) --
# 0.105 sits comfortably inside that (0.094, 0.116] window.
USER_OVERRIDE_OFFSET = [0.02, 0.105]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def test_valid_bin_scene() -> dict:
    backend = So101PyBulletBackend(gui=False, use_bin=True)
    try:
        backend.reset()
        scene = backend.get_scene_state()
        return {
            "resolved_offset": scene["target_zone_offset_xy"],
            "layout_validation_passed": scene["layout_validation_passed"],
            "layout_validation_failures": scene["layout_validation_failures"],
            "object_aabb_initial": scene["object_aabb_initial"],
            "bin_outer_bounds": scene["bin_outer_bounds"],
            "table_surface_bounds": scene["table_surface_bounds"],
            "test_pass": scene["layout_validation_passed"] is True and len(scene["layout_validation_failures"]) == 0,
        }
    finally:
        backend.close()


def test_invalid_overlap() -> dict:
    backend = So101PyBulletBackend(gui=False, use_bin=True, scene_config={"target_zone_offset_xy": INVALID_OFFSET_TOO_SMALL})
    raised = False
    failure_type = None
    error_message = None
    error_has_aabb_info = False
    try:
        backend.reset()
    except InvalidSceneLayoutError as exc:
        raised = True
        failure_type = exc.failure_type
        error_message = str(exc)
        error_has_aabb_info = "object_aabb" in error_message and "target_zone_offset_xy" in error_message
    finally:
        backend.close()

    return {
        "raised_pass": raised, "failure_type": failure_type, "error_message": error_message,
        "error_has_aabb_info_pass": error_has_aabb_info,
        "test_pass": raised and error_has_aabb_info and failure_type in ("object_bin_overlap", "unexpected_object_bin_contact"),
    }


def test_flat_backward_compatibility() -> dict:
    backend = So101PyBulletBackend(gui=False)  # use_bin defaults to False
    try:
        backend.reset()
        scene = backend.get_scene_state()
        offset_unchanged = scene["target_zone_offset_xy"] == DEFAULT_SCENE_CONFIG["target_zone_offset_xy"]
        footprint_unchanged = backend.scene_config["surface_footprint_xy"] == DEFAULT_SCENE_CONFIG["surface_footprint_xy"]
        object_position = scene["object_position"]
        object_position_matches_historical = all(
            math.isclose(object_position[i], DEFAULT_OBJECT_POSITION[i], abs_tol=1e-3) for i in range(3)
        )
        auto_validation_not_run = scene["layout_validation_passed"] is None

        # validate_initial_scene_layout() is still SAFE to call manually
        # for a flat scene (see this task's own "flat scene에 영향을
        # 주지 않음") -- object/table checks (B, D) still apply and
        # should trivially pass; bin-specific checks (A, C) have
        # nothing to check.
        manual_validation = backend.validate_initial_scene_layout()

        return {
            "offset_unchanged_pass": offset_unchanged, "resolved_offset": scene["target_zone_offset_xy"],
            "footprint_unchanged_pass": footprint_unchanged, "resolved_footprint": backend.scene_config["surface_footprint_xy"],
            "object_position": object_position, "object_position_matches_historical_pass": object_position_matches_historical,
            "auto_validation_not_run_pass": auto_validation_not_run,
            "manual_validation_passed": manual_validation["passed"],
            "test_pass": (
                offset_unchanged and footprint_unchanged and object_position_matches_historical
                and auto_validation_not_run and manual_validation["passed"]
            ),
        }
    finally:
        backend.close()


def test_explicit_override() -> dict:
    backend = So101PyBulletBackend(gui=False, use_bin=True, scene_config={"target_zone_offset_xy": USER_OVERRIDE_OFFSET})
    try:
        backend.reset()
        scene = backend.get_scene_state()
        user_value_honored = scene["target_zone_offset_xy"] == USER_OVERRIDE_OFFSET
        not_silently_replaced_by_bin_default = scene["target_zone_offset_xy"] != DEFAULT_BIN_TARGET_ZONE_OFFSET_XY
        return {
            "resolved_offset": scene["target_zone_offset_xy"], "user_value_honored_pass": user_value_honored,
            "not_silently_replaced_by_bin_default_pass": not_silently_replaced_by_bin_default,
            "layout_validation_passed": scene["layout_validation_passed"],
            "test_pass": user_value_honored and not_silently_replaced_by_bin_default and scene["layout_validation_passed"] is True,
        }
    finally:
        backend.close()


def main() -> None:
    crashed = False
    crash_reason = None
    results = {}
    try:
        results["A_valid_bin_scene"] = test_valid_bin_scene()
        results["B_invalid_overlap"] = test_invalid_overlap()
        results["C_flat_backward_compatibility"] = test_flat_backward_compatibility()
        results["D_explicit_override"] = test_explicit_override()
    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"

    overall_pass = not crashed and all(r.get("test_pass", False) for r in results.values())

    output = {
        "crashed": crashed, "crash_reason": crash_reason, "overall_pass": overall_pass,
        "config": {
            "default_bin_target_zone_offset_xy": DEFAULT_BIN_TARGET_ZONE_OFFSET_XY,
            "default_bin_surface_footprint_xy": DEFAULT_BIN_SURFACE_FOOTPRINT_XY,
            "flat_default_target_zone_offset_xy": DEFAULT_SCENE_CONFIG["target_zone_offset_xy"],
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 bin scene-layout validation smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        for name, r in results.items():
            print(f"[{name}] test_pass={r['test_pass']}")
        print()
        print("A_valid_bin_scene:", results["A_valid_bin_scene"])
        print()
        print("B_invalid_overlap:", {k: v for k, v in results["B_invalid_overlap"].items() if k != "error_message"})
        print("  error_message:", results["B_invalid_overlap"]["error_message"][:300] if results["B_invalid_overlap"]["error_message"] else None)
        print()
        print("C_flat_backward_compatibility:", results["C_flat_backward_compatibility"])
        print()
        print("D_explicit_override:", results["D_explicit_override"])
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {resolve(str(OUTPUT_PATH))}")


if __name__ == "__main__":
    main()
