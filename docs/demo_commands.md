# Demo Commands

이 프로젝트의 주요 데모 실행 명령을 모아 정리합니다. 모든 명령은 프로젝트 루트에서 가상환경을 활성화한 뒤 실행합니다.

```bash
source .venv/bin/activate
```

GUI 창을 확인하고 싶으면 `--headless`를 빼고 `--gui`(기본값)로 실행하면 됩니다. 자동화/CI 환경에서는 `--headless`를 권장합니다.

## 1. Full demo

전체 파이프라인(TaskGoal → 인식 → Real2Sim → policy → SafetyGate → 기록)을 한 번에 실행합니다.

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --headless
```

기대 결과: `final_status=success`, `last_event=object_placed_in_bin`, `PASS`.

## 2. Full demo with recording

동일한 흐름을 실행하면서 raw episode(JSON + frame PNG)를 기록합니다.

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --record --record-images \
  --headless
```

기대 결과: `final_status=success`, `PASS`, `recorded_episode: datasets/raw_episodes/episode_...`.

## 3. Safety block demo

mock safety monitor로 hazard를 강제 발생시켜, 로봇이 첫 스텝에서 차단되는지 확인합니다.

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --safety-monitor mock \
  --simulate-hazard \
  --headless
```

기대 결과: `final_status=blocked_by_safety`, `PASS`.

실제 ONNX YOLO 모델로 이미지 기반 hazard를 점검하려면 `--safety-monitor onnx`를 사용합니다(사람이 포함된 이미지가 필요합니다).

## 4. Scripted policy demo

모델 없이 결정론적 4단계 IK 이동만으로 pick-and-place를 수행합니다.

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy scripted \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --headless
```

기대 결과: `policy_steps=4`, `final_status=success`, `PASS`.

## 5. Dummy OpenVLA policy control loop

`run_full_recycling_cell_demo.py`가 감싸기 전, `DummyOpenVLAPolicy`의 online control loop만 단독으로 검증하는 원래 데모입니다.

```bash
python -m benchmark.run_dummy_openvla_policy_control_demo \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --headless
```

`--carry-height`(기본 0.18), `--grasp-z-offset`(기본 0.015), `--max-step-size`, `--position-tolerance` 등으로 phase 동작을 조정할 수 있습니다.

## 6. Dataset export

기록된 raw episode들을 LeRobot 스타일 JSONL dataset으로 변환합니다.

```bash
python -m benchmark.export_lerobot_dataset_demo \
  --raw-episodes-dir datasets/raw_episodes \
  --output-dir datasets/lerobot_recycling_cell_v0
```

정확한 인자 이름은 `python -m benchmark.export_lerobot_dataset_demo --help`로 확인하세요.

## 7. Dataset replay validation

변환된 dataset의 action을 PyBullet Panda에서 다시 재생해서, 기록된 action이 동일한 결과를 재현하는지 검증합니다.

```bash
python -m benchmark.replay_lerobot_dataset_demo \
  --dataset-dir datasets/lerobot_recycling_cell_v0 \
  --headless
```

기대 결과: `replay_success_rate=1.00` (success episode만 모은 dataset 기준).

## 8. OpenVLA action adapter replay

dataset action을 OpenVLA 스타일 7-DoF action으로 변환한 뒤, 같은 방식으로 replay 검증합니다.

```bash
python -m benchmark.replay_openvla_action_adapter_demo \
  --dataset-dir datasets/lerobot_recycling_cell_v0 \
  --headless
```

기대 결과: `replay_success_rate=1.00`.

## 9. Real2Sim mapping calibration probe

PyBullet을 켜지 않고 detection → Real2Sim mapping만 빠르게 확인합니다. `configs/real2sim_webcam_calibration.json`의 `image_roi`/`axis_mapping`/`sim_workspace`를 튜닝할 때, 매번 pick-and-place 전체를 실행하지 않고 이 probe로 `mapped_position`이 원하는 대로 나오는지 먼저 확인하는 용도입니다.

```bash
python -m benchmark.probe_real2sim_mapping \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --save-debug-image
```

`--image-path`, `--camera-index`, `--real2sim-calibration`(기본 `configs/real2sim_webcam_calibration.json`)도 `run_full_recycling_cell_demo.py`와 동일하게 지원합니다. 출력의 `=== Real2Sim Mapping Debug ===` 블록(`bbox_center`, `bbox_size`, `normalized_center`, `clamped`, `mapped_position` 등)으로 같은 물체를 카메라에서 가깝게/멀게/좌우로 옮겼을 때 `mapped_position`이 실제로 달라지는지 비교하세요. `--save-debug-image`를 주면 detection bbox뿐 아니라 calibration ROI 사각형도 함께 그려서 저장합니다.

## 10. ArUco marker 생성

테이블 작업영역 네 모서리에 붙일 marker 4장을 PNG로 생성합니다. 인쇄해서 실제 테이블에 붙인 뒤 아래 ArUco probe/full demo를 사용하세요.

```bash
python -m benchmark.generate_aruco_markers \
  --dictionary DICT_4X4_50 \
  --marker-ids 0 1 2 3 \
  --marker-size-px 600 \
  --output-dir results/aruco_markers
```

`cv2.aruco`가 설치되어 있지 않으면 traceback 없이 안내 메시지와 함께 `FAIL`을 출력합니다.

## 11. ArUco table-plane mapping probe

PyBullet 없이 detection + marker 검출 + homography mapping만 확인합니다. marker 4개(0~3)가 화면에 모두 보이는 상태에서 실행하세요.

```bash
python -m benchmark.probe_aruco_real2sim_mapping \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --save-debug-image
```

`--aruco-calibration`(기본 `configs/real2sim_aruco_table_calibration.json`)로 marker id/`sim_xy`/dictionary를 바꿀 수 있습니다. marker가 4개 미만 감지되면 `ArUco mapping failed: required markers [...], detected [...]`를 출력하고 traceback 없이 `FAIL`로 끝납니다.

## 12. Full demo에서 ArUco mapping 사용

`--real2sim-mode aruco`를 주면 `run_full_recycling_cell_demo.py`가 ROI 매핑 대신 ArUco homography 매핑을 사용합니다(marker가 4개 모두 보여야 PyBullet 단계까지 진행됩니다).

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --real2sim-mode aruco \
  --aruco-calibration configs/real2sim_aruco_table_calibration.json \
  --save-webcam-frame \
  --save-debug-image \
  --headless
```

marker가 부족하면 PyBulletPandaBackend를 아예 생성하지 않고 `FAIL`로 끝납니다. `--real2sim-mode roi`(기본값)로 기존 ROI 매핑 방식을 그대로 쓸 수 있습니다.

## 13. Wrist camera probe

외부 카메라/YOLO/ArUco 없이 PyBullet 안에서만, Panda 손목에 붙인 가상 카메라(`PyBulletWristCamera`)가 지정한 위치의 물체를 실제로 볼 수 있는지 확인합니다.

```bash
python -m benchmark.probe_pybullet_wrist_camera \
  --object-position 0.40 -0.10 0.05 \
  --headless \
  --save-images
```

기대 결과: `object_visible: True`, `position_error_xy <= 0.05`, `PASS`. `--gui`로 실행하면 GUI 창에서 직접 확인할 수 있습니다. `--wrist-camera-config`로 `configs/wrist_camera_config.json`을 교체할 수 있습니다.

## 14. Full demo에서 wrist camera observe 모드 사용

`--wrist-camera-mode observe`를 주면 Real2Sim mapping으로 정한 위치 위쪽으로 end-effector를 한 번 이동시켜 wrist camera로 관찰하고, 추정 위치를 실제 위치와 비교해서 출력합니다. **관찰만 하고 policy의 grasp target은 바꾸지 않으며, 이후 pick-and-place는 기존과 동일하게 수행됩니다.**

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --real2sim-mode aruco \
  --aruco-calibration configs/real2sim_aruco_table_calibration.json \
  --wrist-camera-mode observe \
  --save-wrist-camera-images \
  --save-webcam-frame \
  --save-debug-image \
  --headless
```

`--wrist-camera-mode off`(기본값)는 기존 동작 그대로입니다. wrist camera 이미지/depth/segmentation/debug JSON은 `results/wrist_camera/`에 저장됩니다.

## 15. Wrist camera grasp refinement 확인

PyBullet 안에서만, 일부러 offset을 준 "coarse" target을 wrist camera 추정값으로 실제 물체 위치 쪽으로 보정하는지 확인합니다.

```bash
python -m benchmark.probe_wrist_grasp_refinement \
  --object-position 0.40 -0.10 0.05 \
  --coarse-offset 0.04 -0.03 \
  --policy blend \
  --headless \
  --save-images
```

기대 결과: `refinement_applied: True`, `error_after_xy < error_before_xy`, `PASS`. `--policy override`(x/y 완전 대체)와 `--policy none`(관찰만, 항상 fallback)도 지원합니다.

## 16. Full demo에서 wrist camera refine 모드 사용

`--wrist-camera-mode refine`을 주면(현재 `--policy dummy-openvla`에서만 연결됨) `move_to_object` phase가 grasp 직전(`--refine-distance-threshold` 이내)에 도달했을 때 딱 한 번 wrist camera로 물체 위치를 다시 추정하고, 신뢰 조건을 만족하면 `--wrist-refinement-policy`(기본 `blend`)로 grasp target의 x/y를 보정합니다.

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --real2sim-mode aruco \
  --aruco-calibration configs/real2sim_aruco_table_calibration.json \
  --confidence-threshold 0.10 \
  --wrist-camera-mode refine \
  --wrist-refinement-policy blend \
  --save-wrist-camera-images \
  --save-webcam-frame \
  --save-debug-image \
  --gui \
  --policy-step-delay 0.08
```

기대 결과: `=== Wrist Camera Grasp Refinement ===` 출력, `refinement_applied: True`(신뢰 조건을 만족할 때), `final_status: success`, **PASS**. 최종 요약에도 `wrist_refinement_applied`/`wrist_refinement_delta_xy`가 함께 출력됩니다. 신뢰 조건(안 보임/픽셀 부족/보정폭 초과)을 만족하지 못하면 원래 coarse target 그대로 pick-and-place를 진행합니다(fallback).

## 17. Full demo + episode recording (perception-to-action 전체 기록)

`--record`에 `--record-images`를 더하면, 로봇 trajectory뿐 아니라 detection/Real2Sim mapping/wrist camera refinement 정보까지 episode 폴더 하나에 함께 저장됩니다(`metadata.json` + `debug/*.json` + `frames/*`).

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-source webcam \
  --camera-url http://172.17.32.1:5050/video \
  --real2sim-mode aruco \
  --aruco-calibration configs/real2sim_aruco_table_calibration.json \
  --confidence-threshold 0.10 \
  --wrist-camera-mode refine \
  --wrist-refinement-policy blend \
  --save-wrist-camera-images \
  --save-webcam-frame \
  --save-debug-image \
  --record \
  --record-images \
  --gui \
  --policy-step-delay 0.08
```

기대 결과: 기존과 동일하게 `final_status: success`, `PASS`이면서 `recorded_episode: datasets/raw_episodes/episode_...`가 함께 출력됩니다. `--no-record-perception-metadata`로 metadata.json 저장만 끌 수 있고(로봇 trajectory 기록 자체는 `--record`만으로 계속 동작), `--episode-tag`로 임의의 태그 문자열을 metadata에 남길 수 있습니다.

## 18. 저장된 episode 요약 확인

```bash
python -m benchmark.inspect_recorded_episode \
  --episode-dir datasets/raw_episodes/episode_YYYYMMDD_HHMMSS_xxxxxx
```

기대 결과: instruction, real2sim mode/mapped position, wrist refinement 적용 여부와 몇 번째 step에서 일어났는지, policy_steps, final_status를 요약 출력하고 성공 episode면 `PASS`.
