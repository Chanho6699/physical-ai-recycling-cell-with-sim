"""Perception-to-action episode metadata schema (v0).

A single place that defines the nested metadata attached to each
recorded episode -- external perception, Real2Sim mapping, wrist camera
refinement, robot execution, and final result -- so
run_full_recycling_cell_demo.py (writer), inspect_recorded_episode.py
(reader), and TrajectoryRecorder/LeRobotDatasetExporter (which just
carries whatever dict it's given through unchanged) all agree on the
same shape without redefining it three times.

Not a formal jsonschema/pydantic validator -- these are just builder
functions returning plain dicts, and a thin read/write pair for the
standalone metadata.json file saved alongside episode.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def build_task_goal_section(task_goal) -> dict:
    return {
        "instruction": task_goal.instruction,
        "action": task_goal.action,
        "target_object": task_goal.target_object,
        "target_bin": task_goal.target_bin,
    }


def build_input_source_section(args, saved_webcam_frame_path: Optional[str]) -> dict:
    return {
        "image_source": args.image_source,
        "camera_url_used": bool(args.image_source == "webcam" and args.camera_url),
        "saved_webcam_frame": saved_webcam_frame_path,
    }


def build_detections_section(detections) -> list:
    return [
        {"label": detection.label, "confidence": detection.confidence, "bbox_xyxy": detection.bbox_xyxy}
        for detection in detections
    ]


def build_selected_target_section(detection, sim_object_type: str) -> dict:
    return {
        "label": detection.label,
        "mapped_object_type": sim_object_type,
        "confidence": detection.confidence,
        "bbox_xyxy": detection.bbox_xyxy,
    }


def build_real2sim_section(mode: str, mapping_debug: dict) -> dict:
    if mode == "aruco":
        return {
            "mode": mode,
            "homography_valid": mapping_debug.get("homography_valid"),
            "detected_marker_ids": mapping_debug.get("detected_marker_ids"),
            "mapped_position_raw": mapping_debug.get("mapped_position_raw"),
            "mapped_position": mapping_debug.get("mapped_position"),
            "out_of_bounds": mapping_debug.get("out_of_bounds"),
            "out_of_bounds_policy": mapping_debug.get("out_of_bounds_policy"),
        }
    return {
        "mode": mode,
        "mapped_position": mapping_debug.get("mapped_position"),
        "normalized_center": mapping_debug.get("normalized_center"),
        "clamped": mapping_debug.get("clamped"),
    }


def build_wrist_camera_section(wrist_camera_mode: str, refinement_debug: Optional[dict]) -> dict:
    if wrist_camera_mode == "off":
        return {"mode": "off"}
    if refinement_debug is None:
        # observe mode, or refine mode that never actually fired
        # (arm never got within --refine-distance-threshold).
        return {"mode": wrist_camera_mode, "refinement_attempted": False}

    return {
        "mode": wrist_camera_mode,
        "refinement_policy": refinement_debug.get("refinement_policy"),
        "refinement_attempted": True,
        "refinement_applied": refinement_debug.get("refinement_applied"),
        "coarse_target_position": refinement_debug.get("coarse_target_position"),
        "wrist_estimated_position": refinement_debug.get("wrist_estimated_position"),
        "refined_target_position": refinement_debug.get("refined_target_position"),
        "wrist_refinement_delta_xy": refinement_debug.get("xy_delta_from_coarse"),
        "object_visible": refinement_debug.get("object_visible"),
        "object_pixel_count": refinement_debug.get("object_pixel_count"),
        "fallback_reason": refinement_debug.get("fallback_reason"),
    }


def build_robot_section(policy_name: str, policy_steps: int, final_state: dict) -> dict:
    return {
        "simulator": final_state.get("simulator", "pybullet_panda"),
        "policy": policy_name,
        "policy_backend": final_state.get("policy_backend", "local-dummy"),
        "policy_server_url": final_state.get("policy_server_url"),
        "avg_inference_latency_ms": final_state.get("avg_inference_latency_ms"),
        "policy_steps": policy_steps,
        "final_status": final_state.get("task_status"),
        "last_event": final_state.get("last_event"),
    }


def build_result_section(final_state: dict, bin_position: list, success: bool) -> dict:
    return {
        "success": success,
        "held_object": final_state.get("held_object"),
        "final_object_position": final_state.get("object_position"),
        "bin_position": bin_position,
    }


def build_policy_observation_section(final_state: dict) -> dict:
    """VLA-readiness summary: how many control-loop steps actually fed a
    wrist-camera frame into PolicyInput.image (used_wrist_observation_steps)
    vs. how many of those got a frame saved to disk
    (recorded_wrist_observation_steps, gated by --policy-observation-save-interval)."""
    return {
        "policy_observation_source": final_state.get("policy_observation_source", "none"),
        "used_wrist_observation_steps": final_state.get("used_wrist_observation_steps", 0),
        "recorded_wrist_observation_steps": final_state.get("recorded_wrist_observation_steps", 0),
    }


def build_safety_section(safety_mode: str, mock_hand_intrusion: bool, final_state: dict) -> dict:
    """Safety Pause/Resume summary: mode is outside the VLA policy
    entirely -- the policy proposes actions every step regardless, and
    this section (plus the per-step safety_pause/safety_still_paused/
    safety_resume events) records when the Safety Gate paused their
    application and for how long. hand_safety_source distinguishes v0
    mock-timed intrusion from v1's real external-camera hand detector
    (see safety/external_camera_hand_monitor.py); both drive the exact
    same pause/resume state machine."""
    return {
        "mode": safety_mode,
        "mock_hand_intrusion": mock_hand_intrusion,
        "hand_safety_source": final_state.get("hand_safety_source", "none"),
        "hand_detector_backend": final_state.get("hand_detector_backend"),
        "pause_count": final_state.get("safety_pause_count", 0),
        "resume_count": final_state.get("safety_resume_count", 0),
        "paused_steps": final_state.get("paused_steps", 0),
        "hand_intrusion_events": final_state.get("hand_intrusion_events", 0),
        "final_safety_state": final_state.get("final_safety_state", "running"),
    }


def build_episode_metadata(
    episode_id: str,
    task_goal,
    input_source: dict,
    detections: list,
    selected_target: dict,
    real2sim: dict,
    wrist_camera: dict,
    robot: dict,
    result: dict,
    policy_observation: Optional[dict] = None,
    safety: Optional[dict] = None,
    episode_tag: Optional[str] = None,
) -> dict:
    return {
        "episode_id": episode_id,
        "created_at": datetime.now().isoformat(),
        "episode_tag": episode_tag,
        "task_goal": build_task_goal_section(task_goal),
        "input_source": input_source,
        "detections": detections,
        "selected_target": selected_target,
        "real2sim": real2sim,
        "wrist_camera": wrist_camera,
        "policy_observation": policy_observation or {"policy_observation_source": "none"},
        "safety": safety or {"mode": "off"},
        "robot": robot,
        "result": result,
    }


def write_episode_metadata_file(episode_dir: Union[str, Path], metadata: dict) -> str:
    from data_collection.trajectory_recorder import to_jsonable

    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(to_jsonable(metadata), metadata_file, ensure_ascii=False, indent=2)
    return str(metadata_path)


def load_episode_metadata(episode_dir: Union[str, Path]) -> Optional[dict]:
    metadata_path = Path(episode_dir) / "metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)
