# Dataset Pipeline

이 문서는 로봇 실행 기록이 raw episode에서 시작해서 dataset export, replay 검증, OpenVLA 스타일 action 변환까지 어떻게 이어지는지 정리합니다.

```text
raw episode (TrajectoryRecorder)
        ↓
LeRobot 스타일 JSONL (LeRobotDatasetExporter)
        ↓
replay validator (PyBullet Panda에서 재실행)
        ↓
OpenVLA 스타일 7-DoF action vector (OpenVLAActionAdapter)
        ↓
future: 실제 OpenVLA / LeRobot 학습
```

## 1. raw episode (TrajectoryRecorder)

`data_collection/trajectory_recorder.py`의 `TrajectoryRecorder`가 데모 실행 중 매 step마다 `phase`, `action_name`, `robot_state`, `action`, `safety`, (옵션) `image`, (옵션) `extra`를 기록합니다.

- 저장 위치: `datasets/raw_episodes/episode_<timestamp>_<id>/`
- 내용: `episode.json`(전체 step 기록) + `frames/*.png`(`--record-images`일 때)
- numpy 값(`np.float64`, `np.ndarray` 등)은 `to_jsonable()` 헬퍼로 JSON 직렬화 가능한 native 타입으로 변환합니다.

### perception-to-action 전체 기록 (v0)

`run_full_recycling_cell_demo.py --record`는 이제 로봇 trajectory뿐 아니라 **외부 카메라 관찰 → YOLO detection → Real2Sim(ROI/ArUco) mapping → wrist camera refinement → robot 실행 → 최종 결과**로 이어지는 전체 perception-to-action 체인을 함께 기록합니다.

```text
datasets/raw_episodes/episode_<timestamp>_<id>/
  episode.json          # 기존 그대로: step별 robot_state/action/safety(+ extra 이벤트)
  metadata.json          # 신규: 아래 스키마 전체 (episode_id/task_goal/input_source/
                          #        detections/selected_target/real2sim/wrist_camera/robot/result)
  frames/
    frame_000000.png ...     # 기존: step별 프레임(--record-images)
    external_webcam_frame.jpg     # 신규(있으면): --save-webcam-frame으로 저장된 원본 웹캠 프레임 복사본
    external_detection_debug.jpg  # 신규(있으면): --save-debug-image로 저장된 detection/mapping debug 이미지 복사본
  debug/
    aruco_mapping_debug.json      # 신규: ArUcoTableMapper.map_detection()의 debug dict 전체
                                   #   (roi 모드면 real2sim_mapping_debug.json)
    wrist_refinement_debug.json   # 신규(refine 모드에서 refinement가 실제로 시도됐으면):
                                   #   refine_target_with_wrist_camera()의 debug dict 전체
```

`data_collection/perception_episode_schema.py`가 `metadata.json`의 스키마를 정의합니다(`build_episode_metadata()` 등). 같은 dict를 `TrajectoryRecorder.update_metadata()`로 `episode.json`의 기존 `"metadata"` 필드에도 그대로 병합하므로, exporter가 이미 하던 대로 `metadata`를 그대로 복사하기만 해도 `real2sim`/`wrist_camera`/`selected_target`/`result`가 함께 넘어갑니다(아래 2번 참고). `wrist_camera` refinement가 실제로 일어난 step에는 `record_step(..., extra={"event_type": "wrist_refinement", ...})`로 표시되어, `episode.json`의 `steps` 리스트만 보고도 정확히 몇 번째 step에서 보정이 일어났는지 알 수 있습니다 (`benchmark/inspect_recorded_episode.py`가 이 필드로 `wrist_refinement_step_index`를 찾아 보여줍니다).

**아직 official LeRobot 포맷은 아닙니다** -- 여전히 이 프로젝트 고유의 JSONL/PNG 기반 raw 포맷이고, 이미지/비디오/parquet 공식 변환은 future work로 남아 있습니다 (5번 참고).

## 2. LeRobot 스타일 JSONL (LeRobotDatasetExporter)

`data_collection/lerobot_dataset_exporter.py`의 `LeRobotDatasetExporter`가 raw episode를 다음 구조로 변환합니다.

```text
datasets/<dataset_name>/
  meta/info.json
  meta/episodes.jsonl      # 각 episode의 instruction/task_goal/metadata/성공 여부
  data/episodes.jsonl      # step별 image/state/instruction/action
  videos_or_frames/        # (옵션) frame 이미지
```

**중요한 점: 이것은 LeRobot 공식 parquet/video 포맷이 아닙니다.** 구조와 필드 이름을 LeRobot 스타일로 맞춘 로컬 JSONL 포맷이며, 실제 LeRobot 라이브러리로 바로 로드할 수 있는 형식은 아직 아닙니다.

exporter 코드 자체는 이번 단계에서 수정하지 않았습니다 -- raw episode의 `metadata` 필드를 그대로 복사하는 기존 동작만으로, 새로 추가된 `real2sim`/`wrist_camera`/`selected_target`/`result` 섹션이 `meta/episodes.jsonl`의 각 항목에 자동으로 포함됩니다(1번의 `update_metadata()` 참고).

### action 정의: `action_t = state[t+1] - state[t]`

각 step의 action은 "다음 step에서 실제로 무엇이 바뀌었는가"를 그대로 기록한 delta입니다.

```text
delta_ee_position[t] = end_effector_position[t+1] - end_effector_position[t]
gripper_action[t]    = "close" | "open" | "hold"   (gripper_width[t+1] - gripper_width[t] 기준)
```

이렇게 정의한 이유는, policy가 예측한 "의도된 action"이 아니라 **실제로 시뮬레이션에서 일어난 물리적 결과**를 기록해서, 나중에 이 action을 그대로 재생했을 때 원래 episode와 동일한 결과가 나오는지 검증할 수 있게 하기 위해서입니다. 즉 action label의 정답 여부를 실행 결과로 직접 검증할 수 있는 구조를 우선했습니다.

## 3. Replay Validator

`benchmark/replay_lerobot_dataset_demo.py`가 dataset의 각 episode를 읽어서, 기록된 `delta_ee_position`/`gripper_action`을 `ActionAdapter`를 거쳐 `PyBulletPandaBackend`에 순서대로 적용합니다. 재생 결과의 `task_status`가 원래 episode의 `expected_status`와 같으면 그 episode는 replay에 성공한 것으로 판정합니다.

- success episode만 모은 dataset(`datasets/lerobot_recycling_cell_v0`)에서 `replay_success_rate=1.00`을 확인했습니다.
- `benchmark/replay_openvla_action_adapter_demo.py`는 같은 replay 구조를 사용하되, dataset action을 먼저 `OpenVLAActionAdapter.dataset_action_to_openvla_action()`으로 7-DoF vector로 변환한 뒤 `ActionAdapter`에 넣는 경로까지 검증합니다.

### 사례: gripper action threshold 버그

초기 exporter는 `GRIPPER_ACTION_THRESHOLD = 1e-4`로 gripper_width의 변화량을 open/close/hold로 분류했습니다. 그런데 실제 물리 시뮬레이션에서는 gripper가 멈춰 있어도 미세한 진동(최대 ~1e-3 수준)이 있었고, 이 임계값이 그 노이즈보다 작아서 노이즈를 실제 open/close 이벤트로 잘못 분류했습니다. 그 결과 재생 시 물체를 옮기는 도중에 gripper가 잘못 열리는 것으로 기록되어, replay에서 물체가 중간에 떨어지고(`released`) `replay_success_rate`가 낮게 나왔습니다.

이 문제는 replay validator로 **기록된 action을 실제로 재실행해봤기 때문에** 발견할 수 있었습니다. label만 눈으로 보는 것으로는 드러나지 않는 문제였습니다. 노이즈 수준(~1e-3)과 실제 전환 폭(~0.038)을 확인한 뒤 `GRIPPER_ACTION_THRESHOLD = 0.005`로 올려서 해결했고, 재검증에서 `replay_success_rate=1.00`을 확인했습니다.

## 4. OpenVLA 스타일 7-DoF action

`policy/openvla_action_adapter.py`의 `OpenVLAActionAdapter`가 dataset action을 다음 형태로 변환합니다.

```text
{"delta_ee_position": [dx, dy, dz], "gripper_action": "hold"|"close"|"open"}
        ↓
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

- `droll`/`dpitch`/`dyaw`는 v0에서 항상 `0.0`입니다(기록된 dataset action에 회전 delta가 없기 때문).
- `gripper_action`이 `"hold"`이면 직전에 사용한 gripper 값을 그대로 유지합니다(상태를 가진 adapter).
- 이 vector는 `ActionAdapter.convert()`를 그대로 통과하므로, 실제 OpenVLA 모델이 이 형태의 7-DoF action을 출력한다고 가정했을 때 이후 실행 경로(ActionAdapter → RobotCommand → PyBulletPandaBackend)가 문제없이 동작하는지 미리 검증하는 용도입니다.

## 5. Episode inspector

`benchmark/inspect_recorded_episode.py --episode-dir <경로>`가 `metadata.json`(없으면 `episode.json`의 `metadata` 필드로 fallback)을 읽어서 instruction, real2sim mode/mapped position, wrist camera refinement 적용 여부와 몇 번째 step에서 일어났는지, policy_steps, final_status를 한 화면에 요약합니다. 전체 JSON을 열어보지 않고도 이 episode에서 무슨 일이 있었는지 빠르게 확인하는 용도입니다.

## 6. 앞으로 (future)

- LeRobot 공식 parquet/video 포맷으로 변환하는 exporter 추가
- HuggingFace Hub 업로드 (아직 하지 않음)
- 실제 OpenVLA fine-tuning에 이 dataset을 사용하는 것 (아직 하지 않음)
- 회전(δroll/δpitch/δyaw)이 실제로 의미를 갖는 조작(물체 방향 조정 등)이 추가되면 그에 맞춰 action 정의 확장
- 지금은 `frames/`에 wrist camera rgb/depth/segmentation 이미지 자체를 복사하지 않고 `results/wrist_camera/`의 경로만 step 이벤트에 남깁니다 -- episode 폴더를 완전히 독립적으로 만들려면 이 파일들도 함께 복사하는 것이 다음 개선 지점입니다.
