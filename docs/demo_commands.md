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
