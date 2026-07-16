"""ArUco Real2Sim + SmolVLA distributed closed-loop integration tests (v0).

Covers this task's 7 required scenarios. Reuses existing, already-tested
components unmodified wherever possible:

  - real2sim.aruco_table_mapper.ArUcoTableMapper (unmodified) against the
    project's real configs/real2sim_aruco_table_calibration.json
  - robot_sim.pybullet_panda_backend.PyBulletPandaBackend (unmodified)
  - benchmark.run_full_recycling_cell_demo.build_real2sim_sync_info() /
    run_dummy_openvla_policy() (this task's new/modified code)
  - the FakeServerPolicy/FakeRecorder/make_args/make_task_goal test
    doubles already built for the SmolVLA closed-loop tests

Tests 1-2 use a synthetic ArUco frame (real cv2.aruco marker images
pasted onto a blank canvas at known pixel positions, detected with the
project's real ArucoDetector -- not mocked) so the homography math can
be checked against a hand-computable expected answer without needing a
real camera or a saved photo. Test 3 exercises the real ONNX YOLO
detector on a frame with nothing recognizable. Tests 4-5 use the real
PyBullet backend. Tests 6-7 are regression re-runs of prior suites.

Run: python -m benchmark.test_real2sim_aruco_smolvla_integration
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from benchmark.run_full_recycling_cell_demo import build_real2sim_sync_info
from benchmark.test_full_demo_closed_loop import FakeRecorder, make_args, make_task_goal
from perception.detection_types import Detection
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.aruco_table_mapper import ArUcoTableMapper
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

import benchmark.run_full_recycling_cell_demo as demo
from benchmark.test_language_eval_and_guards import FixedActionPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARUCO_CALIBRATION_PATH = PROJECT_ROOT / "configs" / "real2sim_aruco_table_calibration.json"

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


# Pixel layout for the synthetic frame -- a clean rectangle so the
# expected mapped sim position is hand-computable: the average of the
# four calibrated corner sim_xy values (see configs/
# real2sim_aruco_table_calibration.json), since these four pixel
# corners also form a rectangle and the bbox center below sits at its
# exact centroid.
CANVAS_SIZE = (480, 640)  # (height, width)
MARKER_PIXEL_CENTERS = {
    0: (120, 380),  # front_left  -> sim [0.55, -0.25]
    1: (520, 380),  # front_right -> sim [0.55,  0.25]
    2: (520, 120),  # back_right  -> sim [0.25,  0.25]
    3: (120, 120),  # back_left   -> sim [0.25, -0.25]
}
MARKER_SIZE_PX = 80
EXPECTED_MAPPED_XY = (0.4, 0.0)  # average of the 4 calibrated corners above


def build_synthetic_aruco_frame(marker_ids) -> np.ndarray:
    """A blank light-gray canvas with real ArUco marker images (via
    cv2.aruco.generateImageMarker, the same call
    benchmark/generate_aruco_markers.py uses) pasted at
    MARKER_PIXEL_CENTERS for each id in `marker_ids` -- omitting an id
    is how test 2 simulates "that marker wasn't seen this frame"."""
    height, width = CANVAS_SIZE
    canvas = np.full((height, width, 3), 220, dtype=np.uint8)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    half = MARKER_SIZE_PX // 2
    for marker_id in marker_ids:
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE_PX)
        center_x, center_y = MARKER_PIXEL_CENTERS[marker_id]
        marker_rgb = np.repeat(marker_image[:, :, None], 3, axis=2)
        canvas[center_y - half : center_y + half, center_x - half : center_x + half] = marker_rgb
    return canvas


def make_bbox_detection(label: str = "bottle", confidence: float = 0.9) -> Detection:
    # bbox centered at (320, 250) -- the centroid of the 4 marker-center
    # pixels above: x-centers (120,520,520,120) average to 320,
    # y-centers (380,380,120,120) average to 250 (NOT 240 -- the
    # rectangle's height spans 380 down to 120, whose midpoint is 250).
    return Detection(label=label, confidence=confidence, bbox_xyxy=[300.0, 230.0, 340.0, 270.0])


def main() -> None:
    print("=== 1. synthetic-frame ArUco mapping regression (all 4 markers) ===")
    frame_all = build_synthetic_aruco_frame([0, 1, 2, 3])
    mapper = ArUcoTableMapper(ARUCO_CALIBRATION_PATH)
    detected = mapper.detect_markers(frame_all)
    check("all 4 required marker ids detected", sorted(detected.keys()) == [0, 1, 2, 3], f"detected={sorted(detected.keys())}")

    detection = make_bbox_detection()
    mapped_position, mapping_debug = mapper.map_detection(detection, frame_all)
    check("homography_valid is True", mapping_debug["homography_valid"] is True)
    check("mapped_position is not None", mapped_position is not None)
    if mapped_position is not None:
        check(
            "mapped_position matches hand-computed centroid within 1cm",
            abs(mapped_position[0] - EXPECTED_MAPPED_XY[0]) < 0.01
            and abs(mapped_position[1] - EXPECTED_MAPPED_XY[1]) < 0.01,
            f"mapped_position={mapped_position} expected~={EXPECTED_MAPPED_XY}",
        )
    check("out_of_bounds is False", mapping_debug["out_of_bounds"] is False)
    print()

    print("=== 2. missing marker (id 3) -> strict failure (homography_valid=False, sim_position=None) ===")
    frame_missing = build_synthetic_aruco_frame([0, 1, 2])  # marker 3 not placed
    detected_missing = mapper.detect_markers(frame_missing)
    check("marker 3 is genuinely absent from detection", 3 not in detected_missing, f"detected={sorted(detected_missing.keys())}")
    mapped_missing, debug_missing = mapper.map_detection(detection, frame_missing)
    check("homography_valid is False when a required marker is missing", debug_missing["homography_valid"] is False)
    check("mapped position is None (strict failure, not a guess)", mapped_missing is None)
    check("missing_marker id 3 surfaced in the debug error message", "3" in debug_missing.get("error", ""))
    print()

    print("=== 3. detection failure -> strict failure (no bottle in frame) ===")
    model_path = PROJECT_ROOT / "weights" / "yolo26n.onnx"
    if model_path.exists():
        blank_frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # nothing recognizable
        detector = ONNXYOLODetector(model_path=str(model_path), confidence_threshold=0.5)
        detections = detector.detect(blank_frame)
        check("a blank frame yields no (high-confidence) detections", len(detections) == 0, f"detections={detections}")
    else:
        check("weights/yolo26n.onnx present (skipping real-detector check)", False, "model file not found -- see report")
    # Regression proof that main()'s existing hard-fail branch for this
    # case is still present and unmodified (this task never touched it).
    source = (PROJECT_ROOT / "benchmark" / "run_full_recycling_cell_demo.py").read_text(encoding="utf-8")
    check(
        "main() still hard-fails on empty detections before any real2sim mapping",
        "if not detections:" in source and "No detections found in the image" in source,
    )
    print()

    print("=== 4. successful mapping -> PyBullet object position matches, sync_locked=True ===")
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    state_after_mapping = backend.set_object_position(list(mapped_position))
    check(
        "PyBullet object position equals the ArUco-mapped position",
        all(abs(state_after_mapping["object_position"][i] - mapped_position[i]) < 1e-9 for i in range(3)),
        f"pybullet={state_after_mapping['object_position']} mapped={mapped_position}",
    )

    sync_info = build_real2sim_sync_info(
        real2sim_mode="aruco",
        task_frame_hash="deadbeef0000",
        external_camera_frame_path=None,
        mapping_debug=mapping_debug,
        marker_detections=detected,
        detection=detection,
        sim_position=list(mapped_position),
        pybullet_object_position=state_after_mapping["object_position"],
    )
    check("sync_info.position_match is True", sync_info["position_match"] is True)
    check("sync_info.sync_locked is True", sync_info["sync_locked"] is True)
    check("sync_info.detected_marker_ids reflects all 4 markers", sorted(sync_info["detected_marker_ids"]) == [0, 1, 2, 3])
    check("sync_info.marker_corners_px has 4 entries", len(sync_info["marker_corners_px"]) == 4)
    print()

    print("=== 4b. mismatched position -> sync_locked=False (defensive check itself works) ===")
    bad_sync_info = build_real2sim_sync_info(
        real2sim_mode="aruco",
        task_frame_hash="deadbeef0000",
        external_camera_frame_path=None,
        mapping_debug=mapping_debug,
        marker_detections=detected,
        detection=detection,
        sim_position=[0.4, 0.0, 0.05],
        pybullet_object_position=[0.5, 0.1, 0.05],  # deliberately different
    )
    check("mismatched position -> sync_locked False", bad_sync_info["sync_locked"] is False)
    print()

    print("=== 5. VLA control loop does not overwrite the real2sim-mapped object position ===")
    fake_policy = FixedActionPolicy(translation=(0.01, 0.0, 0.0), gripper=-1.0)
    recorder = FakeRecorder()
    args = make_args(max_policy_steps=5, real_vla_observation_mode="pybullet")
    task_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    original_create_policy_backend = demo.create_policy_backend
    demo.create_policy_backend = lambda _args: fake_policy
    try:
        final_state, steps_run = demo.run_dummy_openvla_policy(
            args,
            backend,
            safety_gate=None,
            recorder=recorder,
            task_frame=task_frame,
            task_goal=make_task_goal(),
            sim_position=list(mapped_position),
            bin_position=[0.3, 0.35, 0.05],
            real2sim_sync_info=sync_info,
        )
    finally:
        demo.create_policy_backend = original_create_policy_backend

    check(
        "object position after the VLA loop still matches the real2sim mapping "
        "(the loop never called set_object_position again)",
        # Tolerance is a few mm, not a hard 0 -- PyBullet's own gravity/
        # contact solver lets an unheld box settle a hair on the table
        # over `steps_per_action` physics ticks each step (real, expected
        # physics, same reason state["object_position"] isn't bit-exact
        # to what set_object_position() originally wrote); a real
        # sync-lock violation (the external camera pipeline re-deriving
        # and re-applying a position) would show up as a jump of
        # centimeters, not sub-mm settling.
        all(abs(final_state["object_position"][i] - mapped_position[i]) < 5e-3 for i in range(3)),
        f"final={final_state['object_position']} mapped={mapped_position}",
    )
    check("held_object is False (nothing physically moved the object either)", final_state["held_object"] is False)

    step_logs = [s["extra"]["real_vla_step_log"] for s in recorder.steps if s.get("extra") and "real_vla_step_log" in s["extra"]]
    check("real2sim_sync attached to every VLA step log", all("real2sim_sync" in log for log in step_logs))
    check(
        "real2sim_sync.sync_locked stayed True on every step (never re-derived/overwritten mid-episode)",
        all(log["real2sim_sync"]["sync_locked"] is True for log in step_logs),
    )
    check(
        "real2sim_sync.mapped_position identical across every step",
        all(log["real2sim_sync"]["mapped_position"] == sync_info["mapped_position"] for log in step_logs),
    )
    check("degraded_input=False on every step (FixedActionPolicy's canned clean response)", all(not log["degraded_input"] for log in step_logs))
    check("fallback_used=False on every step", all(not log["fallback_used"] for log in step_logs))
    check("semantic_action_valid=True on every step", all(log["semantic_action_valid"] for log in step_logs))
    backend.shutdown()
    print()

    print("=== 6. regression: existing strict SmolVLA closed-loop suite ===")
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.test_full_demo_closed_loop"], capture_output=True, text=True, timeout=300
    )
    passed = "ALL CHECKS PASSED" in result.stdout
    check("benchmark.test_full_demo_closed_loop -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=== 7. regression: existing ROI real2sim mapping path (--real2sim-mode roi, the default) ===")
    from real2sim.calibrated_image_to_sim_mapper import CalibratedImageToSimMapper

    roi_mapper = CalibratedImageToSimMapper.from_config_file(PROJECT_ROOT / "configs" / "real2sim_webcam_calibration.json")
    roi_position, roi_debug = roi_mapper.map_bbox_to_sim(detection.bbox_xyxy, 640, 480)
    check("ROI mapper (unmodified this task) still produces a mapped position", roi_position is not None, f"roi_position={roi_position}")
    check("ROI mapping_mode unchanged", roi_debug.get("mapping_mode") == "roi_linear_table_plane", f"got={roi_debug.get('mapping_mode')}")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
