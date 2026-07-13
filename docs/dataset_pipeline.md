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
                          #        detections/selected_target/real2sim/wrist_camera/
                          #        policy_observation/robot/result)
  frames/
    frame_000000.png ...     # 기존: step별 프레임(--record-images)
    external_webcam_frame.jpg     # 신규(있으면): --save-webcam-frame으로 저장된 원본 웹캠 프레임 복사본
    external_detection_debug.jpg  # 신규(있으면): --save-debug-image로 저장된 detection/mapping debug 이미지 복사본
    wrist_policy_step_000005.png  # 신규(있으면): --policy-observation-source wrist + --record-images일 때
                                   #   --policy-observation-save-interval(기본 5) step마다 하나씩
  debug/
    aruco_mapping_debug.json      # 신규: ArUcoTableMapper.map_detection()의 debug dict 전체
                                   #   (roi 모드면 real2sim_mapping_debug.json)
    wrist_refinement_debug.json   # 신규(refine 모드에서 refinement가 실제로 시도됐으면):
                                   #   refine_target_with_wrist_camera()의 debug dict 전체
```

`data_collection/perception_episode_schema.py`가 `metadata.json`의 스키마를 정의합니다(`build_episode_metadata()` 등). 같은 dict를 `TrajectoryRecorder.update_metadata()`로 `episode.json`의 기존 `"metadata"` 필드에도 그대로 병합하므로, exporter가 이미 하던 대로 `metadata`를 그대로 복사하기만 해도 `real2sim`/`wrist_camera`/`policy_observation`/`selected_target`/`result`가 함께 넘어갑니다(아래 2번 참고). `wrist_camera` refinement가 실제로 일어난 step에는 `record_step(..., extra={"event_type": "wrist_refinement", ...})`로 표시되어, `episode.json`의 `steps` 리스트만 보고도 정확히 몇 번째 step에서 보정이 일어났는지 알 수 있습니다 (`benchmark/inspect_recorded_episode.py`가 이 필드로 `wrist_refinement_step_index`를 찾아 보여줍니다).

### VLA-ready per-step observation (`--policy-observation-source wrist`)

`--record-policy-observations`가 켜져 있으면, 매 step `episode.json`의 그 step에 `policy_output`(`action`/`policy_backend`/`inference_latency_ms`/`used_image_input`/`observation_source`)이 `extra`로 기록되고, `--policy-observation-source wrist`가 실제로 관찰을 넣은 step에는 `policy_input`(`image_source`/`image_shape`/`has_image`)과 `wrist_observation`(`object_visible`/`object_pixel_count`/`object_bbox_px`/`estimated_world_position`)도 함께 남습니다. `metadata.json`의 `policy_observation` 섹션에는 episode 전체를 통틀어 `used_wrist_observation_steps`(실제로 wrist 이미지를 policy input으로 사용한 step 수)와 `recorded_wrist_observation_steps`(그중 이미지 파일까지 저장된 step 수)가 요약됩니다.

### policy backend (`--policy-backend local-dummy | fastapi-dummy`)

`metadata.json`의 `robot` 섹션에 `policy_backend`(`local-dummy`/`fastapi-dummy`), `policy_server_url`(fastapi-dummy일 때만), `avg_inference_latency_ms`(episode 전체 step의 평균 왕복 지연시간, fastapi-dummy일 때만 값이 채워짐)가 추가됐습니다. `local-dummy`는 in-process deterministic placeholder(`DummyOpenVLAPolicy`를 직접 호출)이고, `fastapi-dummy`는 external inference-server-style placeholder(`openvla_server_dummy/dummy_server.py`에 HTTP로 같은 `DummyOpenVLAPolicy`를 호스팅)입니다 -- 둘 다 같은 VLA 스타일 `PolicyInput`/`PolicyOutput` 개념을 공유하므로, 나중에 실제 OpenVLA가 `fastapi-dummy`의 서버 구현부만 대체하면 이 데이터 스키마와 control loop는 그대로 재사용됩니다. 지금의 fastapi dummy server는 image content로 learned visual reasoning을 하지 않고, `inference_latency_ms`만 기록해서 나중에 real model과 비교할 수 있는 자리를 마련해 둔 상태입니다.

`DummyOpenVLAPolicy`는 이 이미지 내용을 실제로 해석하지 않는 placeholder이지만, 이 구조 덕분에 나중에 real OpenVLA를 같은 `PolicyInput`/`PolicyOutput` 인터페이스로 꽂아 넣어도 기록되는 데이터 형태는 그대로입니다.

### Safety Pause/Resume (`--safety-mode pause-resume`)

Safety Pause/Resume은 VLA policy 바깥에서 동작합니다 -- policy는 매 step 계속 action을 제안하려 하지만(정확히는, pause 중에는 policy가 아예 호출되지도 않습니다), Safety Gate가 그 action을 로봇에 적용할지 말지를 결정합니다. `episode.json`의 `steps` 리스트에는 hazard가 감지/해소될 때마다 `extra`로 이벤트가 남습니다.

```json
{"step_index": 11, "event_type": "safety_pause", "reason": "mock_hand_intrusion",
 "safety_mode": "pause-resume", "robot_action_applied": false, "hand_detected": true}
{"step_index": 12, "event_type": "safety_still_paused", "reason": "mock_hand_intrusion",
 "safety_mode": "pause-resume", "robot_action_applied": false, "hand_detected": true}
{"step_index": 23, "event_type": "safety_resume", "reason": "hand_cleared",
 "stable_clear_steps": 3, "robot_action_applied": false, "hand_detected": false}
```

`metadata.json`의 `safety` 섹션(`build_safety_section()`)에는 episode 전체 요약이 남습니다:

```json
{"safety": {"mode": "pause-resume", "hand_safety_source": "external-camera",
 "hand_detector_backend": "mediapipe", "pause_count": 1, "resume_count": 1,
 "paused_steps": 8, "hand_intrusion_events": 1, "final_safety_state": "running"}}
```

`--hand-safety-source`가 실제 hand intrusion 신호의 출처를 결정합니다: `mock`(v0, `--mock-hand-start-step`/`--mock-hand-end-step` 구간을 손이 있는 것처럼 흉내)과 `external-camera`(v1, MediaPipe HandLandmarker로 실제 외부 카메라 frame에서 손을 검출)가 정확히 같은 pause/resume state machine과 episode/metadata 스키마를 공유합니다 -- `reason` 필드만 `"mock_hand_intrusion"` 대 `"hand_in_workspace"`로 다릅니다. `hand_intrusion_events`는 raw hand-in-workspace 감지 횟수(현재는 pause_count와 동일하게 집계), `hand_detector_backend`는 `external-camera`일 때만 `"mediapipe"`로 채워집니다. `--safety-mode off`(기본값)/`block`은 이 기능 이전과 동일하게 동작하고, `pause_count`/`resume_count`/`paused_steps`/`hand_intrusion_events`는 모두 0으로 남습니다.

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

`benchmark/inspect_recorded_episode.py --episode-dir <경로>`가 `metadata.json`(없으면 `episode.json`의 `metadata` 필드로 fallback)을 읽어서 instruction, real2sim mode/mapped position, wrist camera refinement 적용 여부와 몇 번째 step에서 일어났는지, `safety_mode`가 `off`가 아니면 `hand_safety_source`(+`external-camera`면 `hand_detector_backend`, 아니면 `mock_hand_intrusion`)/`safety_pause_count`/`safety_resume_count`/`paused_steps`/`hand_intrusion_events`/`final_safety_state`, policy_backend(및 `fastapi-dummy`면 `avg_inference_latency_ms`), policy_steps, final_status를 한 화면에 요약합니다. 전체 JSON을 열어보지 않고도 이 episode에서 무슨 일이 있었는지 빠르게 확인하는 용도입니다.

## 6. 앞으로 (future)

- LeRobot 공식 parquet/video 포맷으로 변환하는 exporter 추가
- HuggingFace Hub 업로드 (아직 하지 않음)
- Safety Pause/Resume: 외부 카메라 hand intrusion detector(v1, `--hand-safety-source external-camera`)에 이어 wrist camera 기반 hand safety, 외부+wrist 카메라 fusion, 실제 하드웨어 ROS2 e-stop 브릿지
- 실제 OpenVLA fine-tuning에 이 dataset을 사용하는 것 (아직 하지 않음)
- 회전(δroll/δpitch/δyaw)이 실제로 의미를 갖는 조작(물체 방향 조정 등)이 추가되면 그에 맞춰 action 정의 확장
- 지금은 `frames/`에 wrist camera rgb/depth/segmentation 이미지 자체를 복사하지 않고 `results/wrist_camera/`의 경로만 step 이벤트에 남깁니다 -- episode 폴더를 완전히 독립적으로 만들려면 이 파일들도 함께 복사하는 것이 다음 개선 지점입니다.
