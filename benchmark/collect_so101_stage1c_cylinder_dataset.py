"""Stage 1C upright-cylinder dataset collector (see this task's chat
report, "Stage 1C의 정식 데이터 생성"). Collects 100 NEW episodes (70
train / 15 validation / 15 held-out test) of an upright cylinder --
radius=0.02m, height=0.04m, identical footprint scale to the already-
validated cube/box (this task's own Expert V2.1 feasibility check:
125/125 = 100% legacy AND physical success).

Uses Expert V2.1 (benchmark.so101_expert_v2_size_aware's own
run_pick_and_place_episode_v2_1()), NEVER V1
(benchmark.so101_scripted_expert.run_pick_and_place_episode) and NEVER
the orientation-aware V2 (benchmark.so101_expert_v2_orientation) -- per
this task's own explicit instruction and absolute principle 2 (neither
V1 nor orientation-V2 files are imported for control flow, only for
their OWN already-validated helper pieces: make_frame_recorder(),
write_phase_id_mapping(), verify_dataset(), compute_bin_success_debug(),
evaluate_bin_place_success() -- all read-only reuse, not modification).

A NEW, additive collect_episode_v2_1() lives in THIS file (not added to
benchmark/collect_so101_bin_dataset.py's own collect_episode(), which
hardcodes V1's run_pick_and_place_episode()) -- per this task's own
section-3 guidance: "기존 결과가 바뀔 위험이 있다면 additive 신규
스크립트를 사용하라".

Position groups (5, this task's own required set) map onto
benchmark.collect_so101_stage1b_box_dataset's own already-validated
REGION_DEFS (REUSED, not redefined) -- train spreads across multiple
sub-regions per group for broader coverage; validation/test each pin
ONE representative sub-region per group (the SAME ones this task's own
Expert-only cylinder evaluation already used: center, existing_x_min,
bridge_plus_x, corner_pn, x_min_corridor) for direct before/after
comparability. See this file's own POSITION_GROUP_PLAN for the exact,
reported adjustment.

Seed blocks (train=20000s, validation=21000s, test=22000s) are new and
disjoint from EVERY seed block used anywhere else in this project
(original dataset 0-199; Stage 1A 5000s/6000s/7000s; Stage 1B box
15000s/16000s/17000s/18000s; zero-shot eval policy-noise bases
500000/600000/700000/800000 -- a different seed namespace entirely).

Run:
  .venv-vla/bin/python -m benchmark.collect_so101_stage1c_cylinder_dataset
"""

import datetime
import json
import math
from collections import Counter
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import pybullet as p

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_MODE_ANCHOR_OFFSET_XY, FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY
from benchmark.collect_so101_bin_dataset import DEFAULT_INSTRUCTION
from benchmark.collect_so101_episode import make_frame_recorder, verify_dataset, write_phase_id_mapping
from benchmark.collect_so101_stage1b_box_dataset import REGION_DEFS
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_dataset_schema import SO101_FEATURES, SO101_ROBOT_TYPE
from benchmark.so101_expert_v2_size_aware import ObjectMetadata, run_pick_and_place_episode_v2_1
from benchmark.so101_scripted_expert import (
    So101ExpertError,
    compute_bin_success_debug,
    evaluate_bin_place_success,
)
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, DEFAULT_SCENE_CONFIG, InvalidSceneLayoutError, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = "datasets/so101_bin_stage1c_cylinder_100"
REPO_ID = "local/so101_bin_stage1c_cylinder_100"
FPS = 10

CYLINDER_RADIUS_M = 0.02
CYLINDER_HEIGHT_M = 0.04
OBJECT_TYPE = "upright_cylinder_v1"
OBJECT_SHAPE = "cylinder"

# Deterministic yaw assignment (this task's own section 4: "scene 다양성과
# 대칭성 검증을 위해... deterministic하게 배정할 수 있다") -- cycles
# through the 5 values by episode order within each region; recorded as
# metadata ONLY (never a claimed capability -- see this file's own
# position_manifest "yaw" field and docs/experiments summary's explicit
# "yaw별 성능을 별도 능력으로 주장하지 않는다").
YAW_CYCLE_DEG = [0, 45, 90, 135, 180]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# Position-group -> sub-region plan (this task's own required 5 groups,
# reusing REGION_DEFS's own x_range/y_range verbatim -- SEEDS are new,
# region GEOMETRY is not redefined). Train spreads across every
# sub-region in a group for broader coverage; validation/test pin ONE
# representative sub-region per group (matching
# evaluate_so101_expert_v2_cylinder.py's own group->region mapping) for
# direct before/after comparability. See this task's own final report,
# "위치 그룹 분포" for the reported rationale.
POSITION_GROUP_PLAN = {
    "train": {
        "center": {"center": (10, list(range(20000, 20015)))},
        "interior": {
            "existing_x_min": (4, list(range(20100, 20110))), "existing_x_max": (4, list(range(20110, 20120))),
            "existing_y_min": (4, list(range(20120, 20130))), "existing_y_max": (3, list(range(20130, 20140))),
        },
        "edge": {
            "bridge_plus_x": (5, list(range(20200, 20210))), "bridge_minus_y": (5, list(range(20210, 20220))),
            "bridge_plus_y": (5, list(range(20220, 20230))),
        },
        "corner": {
            "corner_pp": (5, list(range(20300, 20310))), "corner_pn": (5, list(range(20310, 20320))),
            "corner_np": (5, list(range(20320, 20330))), "corner_nn": (5, list(range(20330, 20340))),
        },
        "x_min_corridor": {"x_min_corridor": (10, list(range(20400, 20415)))},
    },
    "validation": {
        "center": {"center": (3, list(range(21000, 21006)))},
        "interior": {"existing_x_min": (3, list(range(21020, 21026)))},
        "edge": {"bridge_plus_x": (3, list(range(21040, 21046)))},
        "corner": {"corner_pn": (3, list(range(21060, 21066)))},
        "x_min_corridor": {"x_min_corridor": (3, list(range(21080, 21086)))},
    },
    "test": {
        "center": {"center": (3, list(range(22000, 22006)))},
        "interior": {"existing_x_min": (3, list(range(22020, 22026)))},
        "edge": {"bridge_plus_x": (3, list(range(22040, 22046)))},
        "corner": {"corner_pn": (3, list(range(22060, 22066)))},
        "x_min_corridor": {"x_min_corridor": (3, list(range(22080, 22086)))},
    },
}


def collect_episode_v2_1(
    dataset, seed: int, task: str, episode_index_counter: dict, yaw_deg: float,
    x_range: tuple, y_range: tuple, bin_center_override_xy: list, scene_config: dict,
) -> dict:
    """Additive V2.1-driven counterpart of
    benchmark.collect_so101_bin_dataset.collect_episode() -- SAME
    save/discard contract (place_success required, buffer cleared on
    any failure, episode_index only assigned on save), but drives
    Expert V2.1 instead of V1. Reuses make_frame_recorder() UNCHANGED
    (it only needs an on_step(phase, arm_joint_targets,
    gripper_target_normalized) callback, which run_pick_and_place_episode_v2_1()
    already provides identically to V1's own run_pick_and_place_episode())."""
    yaw_rad = math.radians(yaw_deg)
    sampled_object_position = sample_object_position(seed, x_range, y_range)

    backend = So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=bin_center_override_xy, scene_config=scene_config, object_yaw_rad=yaw_rad,
    )

    try:
        try:
            backend.reset()
        except InvalidSceneLayoutError as exc:
            return {
                "seed": seed, "saved": False, "failure_reason": f"scene_invalid:{exc.failure_type}",
                "sampled_object_position": sampled_object_position, "frame_count": 0,
            }

        scene = backend.get_scene_state()
        if not scene["layout_validation_passed"]:
            return {
                "seed": seed, "saved": False, "failure_reason": "scene_invalid",
                "sampled_object_position": sampled_object_position, "frame_count": 0,
            }

        dynamics_info = p.getDynamicsInfo(backend.object_id, -1, physicsClientId=backend.client_id)
        object_mass, lateral_friction, restitution = dynamics_info[0], dynamics_info[1], dynamics_info[5]
        rolling_friction, spinning_friction = dynamics_info[6], dynamics_info[7]

        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        on_step, frame_counter = make_frame_recorder(dataset, backend, task)
        metadata = ObjectMetadata(
            shape="cylinder", position=list(sampled_object_position), height_m=CYLINDER_HEIGHT_M,
            radius_m=CYLINDER_RADIUS_M, mass_kg=object_mass, friction=lateral_friction,
        )

        failure_phase = None
        try:
            result = run_pick_and_place_episode_v2_1(backend, metadata, transport_delta_xy, on_step=on_step)
            bin_debug = result["bin_place_result"]["debug"]
            final_object_position = backend.get_object_position()
            bin_place_debug_for_success = {
                "rise_reached": bin_debug["rise_reached"], "pre_place_reached": bin_debug["pre_place_reached"],
                "descend_reached": bin_debug["descend_reached"], "retreat_reached": bin_debug["retreat_reached"],
                "object_separated_during_wait": result["bin_place_result"]["object_separated_during_wait"],
            }
            bin_success_debug = compute_bin_success_debug(
                backend, bin_place_debug_for_success, result["bin_place_result"]["release_constraint_removed"],
                final_object_position, True, scene["layout_validation_passed"],
            )
            legacy_success, failure_reason, failure_phase = evaluate_bin_place_success(bin_success_debug)
        except So101ExpertError as exc:
            legacy_success, failure_reason, failure_phase = False, exc.failure_reason, exc.phase

        if not legacy_success:
            dataset.clear_episode_buffer()
            return {
                "seed": seed, "saved": False, "failure_reason": failure_reason, "failure_phase": failure_phase,
                "sampled_object_position": sampled_object_position, "frame_count": frame_counter["count"],
                "object_mass": object_mass, "lateral_friction": lateral_friction,
                "rolling_friction": rolling_friction, "spinning_friction": spinning_friction, "restitution": restitution,
            }

        dataset.save_episode()
        episode_index = episode_index_counter["count"]
        episode_index_counter["count"] += 1
        return {
            "seed": seed, "saved": True, "failure_reason": None, "failure_phase": None,
            "episode_index": episode_index, "sampled_object_position": sampled_object_position,
            "frame_count": frame_counter["count"],
            "object_mass": object_mass, "lateral_friction": lateral_friction,
            "rolling_friction": rolling_friction, "spinning_friction": spinning_friction, "restitution": restitution,
        }
    finally:
        backend.close()


def main() -> None:
    root = resolve(DATASET_ROOT)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")

    for split, groups in POSITION_GROUP_PLAN.items():
        target_sum = sum(target for regions in groups.values() for target, _ in regions.values())
        expected = {"train": 70, "validation": 15, "test": 15}[split]
        assert target_sum == expected, f"{split} targets sum to {target_sum}, expected {expected}"

    all_seed_pools = []
    for split, groups in POSITION_GROUP_PLAN.items():
        pool = set()
        for regions in groups.values():
            for _, seeds in regions.values():
                pool.update(seeds)
        all_seed_pools.append((split, pool))
    for i in range(len(all_seed_pools)):
        for j in range(i + 1, len(all_seed_pools)):
            overlap = all_seed_pools[i][1] & all_seed_pools[j][1]
            assert not overlap, f"seed overlap between {all_seed_pools[i][0]} and {all_seed_pools[j][0]}: {overlap}"

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, features=SO101_FEATURES, root=str(root), robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )

    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {
        "surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
        "object_shape": "cylinder", "object_radius": CYLINDER_RADIUS_M, "object_height": CYLINDER_HEIGHT_M,
    }

    episode_index_counter = {"count": 0}
    position_manifest = []
    attempts_log = []
    yaw_cursor = {"i": 0}

    def next_yaw_deg():
        yaw = YAW_CYCLE_DEG[yaw_cursor["i"] % len(YAW_CYCLE_DEG)]
        yaw_cursor["i"] += 1
        return yaw

    try:
        for split in ("train", "validation", "test"):
            for position_group, regions in POSITION_GROUP_PLAN[split].items():
                for region_name, (target, seed_candidates) in regions.items():
                    region = REGION_DEFS[region_name]
                    saved_count = 0
                    for seed in seed_candidates:
                        if saved_count >= target:
                            break
                        yaw_deg = next_yaw_deg()
                        outcome = collect_episode_v2_1(
                            dataset, seed, DEFAULT_INSTRUCTION, episode_index_counter, yaw_deg,
                            region["x_range"], region["y_range"], fixed_bin_center_xy, scene_config_override,
                        )
                        attempts_log.append({
                            "seed": seed, "split": split, "position_group": position_group, "region_name": region_name,
                            "yaw_deg": yaw_deg, "saved": outcome["saved"], "failure_reason": outcome["failure_reason"],
                            "failure_phase": outcome.get("failure_phase"),
                        })
                        print(f"[{split}/{position_group}/{region_name}] seed={seed} yaw={yaw_deg}deg saved={outcome['saved']} "
                              f"failure_reason={outcome['failure_reason']} ({saved_count + int(outcome['saved'])}/{target})")

                        if outcome["saved"]:
                            saved_count += 1
                            sampled = outcome["sampled_object_position"]
                            x_offset = sampled[0] - DEFAULT_OBJECT_POSITION[0]
                            y_offset = sampled[1] - DEFAULT_OBJECT_POSITION[1]
                            position_manifest.append({
                                "episode_id": outcome["episode_index"], "split": split, "seed": seed,
                                "expert_version": "v2.1", "strategy": "size_aware",
                                "object_shape": OBJECT_SHAPE, "object_radius": CYLINDER_RADIUS_M,
                                "object_diameter": 2.0 * CYLINDER_RADIUS_M, "object_height": CYLINDER_HEIGHT_M,
                                "object_position": list(sampled), "object_yaw": math.radians(yaw_deg),
                                "position_group": position_group, "region_name": region_name,
                                "object_mass": outcome["object_mass"], "lateral_friction": outcome["lateral_friction"],
                                "rolling_friction": outcome["rolling_friction"], "spinning_friction": outcome["spinning_friction"],
                                "restitution": outcome["restitution"],
                                "constraint_based_grasp": True, "contact_physics_verified": False,
                                "legacy_success": True, "constraint_based_success": True,
                                "failure_phase": None, "failure_reason": None, "discarded": False,
                                "x_offset": x_offset, "y_offset": y_offset,
                            })
                    if saved_count < target:
                        raise RuntimeError(
                            f"Stage 1C cylinder collection shortfall: split={split!r} group={position_group!r} "
                            f"region={region_name!r} saved only {saved_count}/{target} from {len(seed_candidates)} seed candidates."
                        )
    finally:
        dataset.finalize()

    if episode_index_counter["count"] > 0:
        write_phase_id_mapping(root)

    manifest_path = root / "stage1c_position_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in position_manifest:
            f.write(json.dumps(record, default=str) + "\n")

    attempts_log_path = root / "stage1c_collection_attempts_log.jsonl"
    with open(attempts_log_path, "w", encoding="utf-8") as f:
        for record in attempts_log:
            f.write(json.dumps(record, default=str) + "\n")

    verification = verify_dataset(root)

    summary = {
        "dataset_name": root.name, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "object_type": OBJECT_TYPE, "cylinder_radius_m": CYLINDER_RADIUS_M, "cylinder_height_m": CYLINDER_HEIGHT_M,
        "total_episodes_saved": episode_index_counter["count"],
        "split_counts": {split: sum(1 for r in position_manifest if r["split"] == split) for split in ("train", "validation", "test")},
        "position_group_counts": {g: sum(1 for r in position_manifest if r["position_group"] == g) for g in POSITION_GROUP_PLAN["train"]},
        "yaw_counts": {},
        "total_attempts": len(attempts_log), "total_discarded": sum(1 for a in attempts_log if not a["saved"]),
        "failure_reason_counts": {}, "verify_dataset_result": verification, "dataset_root": str(root),
    }
    summary["yaw_counts"] = dict(Counter(r["object_yaw"] for r in position_manifest))
    summary["failure_reason_counts"] = dict(Counter(a["failure_reason"] for a in attempts_log if not a["saved"]))

    with open(root / "collection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"total_episodes_saved: {summary['total_episodes_saved']}/100")
    print(f"split_counts: {summary['split_counts']}")
    print(f"position_group_counts: {summary['position_group_counts']}")
    print(f"total_discarded: {summary['total_discarded']}")
    print(f"failure_reason_counts: {summary['failure_reason_counts']}")
    print(f"verify_dataset: state_has_nan_or_inf={verification.get('state_has_nan_or_inf')} "
          f"action_has_nan_or_inf={verification.get('action_has_nan_or_inf')}")
    print(f"Dataset root: {root}")


if __name__ == "__main__":
    main()
