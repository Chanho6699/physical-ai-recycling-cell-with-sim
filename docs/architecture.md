# Architecture

이 문서는 `run_full_recycling_cell_demo.py` 기준 **현재** 파이프라인을 정리합니다. 프로젝트 초기 단계(더미 sphere backend, FastAPI dummy OpenVLA server 중심 구조)의 설계 기록은 [00_project_goal.md](00_project_goal.md), [03_architecture.md](03_architecture.md)에 남아 있습니다. 이 문서는 그 이후 TaskGoal / Real2Sim / Franka Panda / SafetyGate / dataset pipeline / DummyOpenVLAPolicy가 추가된 현재 상태를 설명합니다.

## 전체 파이프라인

```text
Instruction + Image
        ↓
TaskGoal Parser
        ↓
YOLO Detector
        ↓
Target Selector
        ↓
Real2Sim Mapper
        ↓
Policy
(scripted / dummy-openvla / future OpenVLA)
        ↓
7-DoF Action
        ↓
ActionAdapter
        ↓
RobotCommand
        ↓
SafetyGate
        ↓
PyBullet Panda Backend
        ↓
TrajectoryRecorder
        ↓
Dataset Export / Replay Validation
```

## Hardware-Portable Backend Abstraction (v0)

**이 프로젝트는 PyBullet 데모가 아니라, 실제 하드웨어 로봇팔에 이식했을 때 바로 기능할 수 있는 Physical AI / VLA-ready 소프트웨어 뇌를 만드는 것이 목표입니다.** 그래서 control loop(`run_full_recycling_cell_demo.py`)는 특정 구현체가 아니라 네 개의 인터페이스에 의존하도록 정리되어 있습니다.

| 인터페이스 | 위치 | 현재 구현체 |
|---|---|---|
| `RobotBackend` | `robot_core/robot_backend.py` | `PyBulletPandaBackend` |
| `CameraBackend` | `vision/camera_backend.py` | `WebcamCameraBackend`, `StaticImageCameraBackend`, `PyBulletWristCameraBackend` |
| `PolicyBackend` (= `BasePolicy`) | `policy/policy_backend.py`, `policy/base_policy.py` | `DummyOpenVLAPolicy`, `FastAPIVLAPolicyClient` |
| `SafetySupervisor` | `safety/safety_supervisor.py` | pause/resume state machine (mock 또는 external-camera hand intrusion 신호 기반) |

`run_full_recycling_cell_demo.py`의 `create_robot_backend(args)` / `create_policy_backend(args)` / `create_external_camera_backend(args)` / `create_hand_safety_monitor(args, ...)` 네 helper 함수가 각 인터페이스의 실제 구현체를 결정하는 유일한 지점입니다 -- control loop 나머지 코드는 이 함수들이 반환한 객체의 인터페이스 메서드만 호출합니다. 하드웨어로 옮길 때는 이 네 함수의 반환값만 바꾸면 됩니다.

`RobotBackend`는 실제 하드웨어 이식 시 무엇을 구현해야 하는지 보여주는 두 개의 skeleton도 함께 정의합니다 (아직 실제 제어 코드는 없고 모든 메서드가 `NotImplementedError`를 던집니다):

- `robot_core/real_robot_backend.py`의 `RealRobotBackend` -- vendor SDK/컨트롤러 API로 이어질 자리.
- `robot_core/ros2_robot_backend.py`의 `ROS2RobotBackend` -- **ROS2RobotBackend is a hardware-portability skeleton, not a working ROS2 controller yet.** `rclpy`는 `__init__` 안에서만 lazy import되므로, ROS2가 설치되어 있지 않아도 이 프로젝트의 나머지 부분은 전혀 영향받지 않습니다. 예상 연결점: `/joint_states`, `/joint_trajectory_controller/follow_joint_trajectory`, `/gripper_controller`, `/tf`, `/camera/color/image_raw`, `/safety_state`, MoveIt2 planning service/action.

`vision/camera_backend.py`도 같은 방식으로 `ROS2CameraBackend` skeleton을 포함합니다(역시 lazy `rclpy` import, 아직 미구현). v0에서는 `run_full_recycling_cell_demo.py`의 기존 프레임 로딩 경로(`--image-source image/webcam`) 자체를 강제로 `CameraBackend`로 바꾸지 않았습니다 -- 이미 검증된 성공 경로를 건드리지 않기 위한 선택이며, `create_external_camera_backend()`는 별도로 동작이 확인된 독립 adapter로 존재합니다.

자세한 이식 체크리스트(캘리브레이션, 교체 매핑 표)는 [docs/hardware_portability.md](hardware_portability.md)를 참고하세요.

## 구성 요소

### TaskGoal Parser

`llm_agent/rule_based_parser.py`의 `RuleBasedTaskGoalParser`가 한국어 명령을 `TaskGoal`(`action`, `target_object`, `target_bin`, `instruction`, `constraints`)로 변환합니다. 실제 LLM을 사용하지 않는 rule-based 구현이며, 나중에 LLM Agent로 교체할 수 있도록 인터페이스만 고정해 두었습니다.

### YOLO Detector

`perception/onnx_yolo_detector.py`의 `ONNXYOLODetector`가 ONNX Runtime으로 이미지에서 객체를 탐지해 `Detection`(label, confidence, bbox_xyxy) 리스트를 반환합니다.

### Target Selector

`real2sim/recyclable_object_mapper.py`의 `RecyclableObjectMapper.select_recyclable_by_target()`이 탐지 결과 중 `TaskGoal.target_object`와 일치하는 대상을 고릅니다.

### Real2Sim Mapper

세 가지 매퍼가 있고, `run_full_recycling_cell_demo.py`/`probe_*_real2sim_mapping.py`는 `--real2sim-mode {roi, aruco}`로 뒤의 둘을 선택합니다.

- `real2sim/image_to_sim_mapper.py`의 `ImageToSimMapper` (v0): bbox 중심 픽셀 좌표를 image_x→sim_x, image_y→sim_y로 그대로 늘려 붙이는 단순 선형 매핑. `run_task_goal_real2sim_panda_demo.py` 계열 등 이전 데모들이 계속 사용합니다.
- `real2sim/calibrated_image_to_sim_mapper.py`의 `CalibratedImageToSimMapper` (v1, `--real2sim-mode roi`, 기본값): ROI(`image_roi`), 축 매핑(`axis_mapping`), ROI-clamp 여부(`clamp_to_roi`)를 `configs/real2sim_webcam_calibration.json`으로 명시적으로 분리한 calibrated 버전. depth 카메라나 depth estimation 없이, "카메라가 고정되어 있고 물체는 테이블 평면 위에 있다"는 가정만으로 물체가 카메라에서 가까워지거나 멀어질 때(대개 image y가 변함)를 Panda의 forward/depth 축(`sim_x`)에, 좌우 이동(image x)을 `sim_y`에 반영합니다. ROI 밖 bbox는 `clamp_to_roi=true`면 가장 가까운 ROI 경계로 clamp합니다. `map_bbox_to_sim()`은 debug dict를 함께 반환해서 `print_mapping_debug()`/`draw_roi_rectangle()`로 원인 분석이 가능합니다. 카메라를 조금이라도 움직이면 `image_roi`를 사람이 다시 수동으로 맞춰야 하는 baseline입니다.
- `real2sim/aruco_table_mapper.py`의 `ArUcoTableMapper` (v0, `--real2sim-mode aruco`): 테이블 작업영역 네 모서리에 붙인 ArUco marker 4개(`configs/real2sim_aruco_table_calibration.json`의 `required_marker_ids`, 각 marker의 알려진 `sim_xy`)를 매 프레임 `cv2.aruco.ArucoDetector`로 탐지하고, marker 중심 픽셀 좌표 ↔ 알려진 sim_xy 4쌍으로 `cv2.findHomography()`를 계산해서 그 프레임의 다른 임의 픽셀(선택된 bottle bbox 중심)을 sim 평면 좌표로 변환합니다. ROI 매핑과 달리 호모그래피를 매번 새로 계산하므로, 카메라가 살짝 움직이거나 각도가 바뀌어도 marker 4개가 보이는 한 재보정 없이 그대로 동작합니다. marker가 4개 미만 감지되면 `map_detection()`이 `(None, debug)`를 반환하고 `debug["error"]`에 원인(`required_marker_ids` vs `detected_marker_ids`)을 담아서, 호출부가 traceback 없이 `FAIL` 처리하도록 합니다. `draw_aruco_debug_image()`가 marker 외곽선/ID/테이블 폴리곤/bbox/mapped position을 함께 그립니다. `benchmark/generate_aruco_markers.py`로 marker PNG(`DICT_4X4_50`, id 0~3)를 생성해 인쇄합니다.

### Policy

세 가지 policy가 동일한 이후 단계(ActionAdapter → SafetyGate → PyBulletPandaBackend)를 공유합니다.

- **scripted** (`benchmark/run_full_recycling_cell_demo.py::run_scripted_policy`): `move_end_effector_to(object) → close_gripper → move_end_effector_to(bin 위) → open_gripper` 4단계를 한 번의 큰 IK 이동으로 수행하는 결정론적 시퀀스.
- **dummy-openvla** (`policy/dummy_openvla_policy.py::DummyOpenVLAPolicy`): 모델 없이 phase state machine으로 동작하는 scripted oracle policy. `BasePolicy` 인터페이스(`reset()`, `predict_action(PolicyInput) -> PolicyOutput`)를 구현합니다.
  - phase: `move_to_object → close_gripper → lift_object → move_above_bin → open_gripper → done`
  - `move_to_object`/`close_gripper` 이후 곧바로 대각선으로 bin까지 이동하면 낮은 높이에서 테이블/bin과 충돌하거나 IK가 불안정해지는 문제가 있어서, `lift_object`(수직 상승) → `move_above_bin`(수평 이동 후 수직 하강) 두 단계로 나누어 해결했습니다.
  - phase 전환은 내부 거리 계산뿐 아니라 backend state(`held_object`, `task_status`)도 함께 확인합니다.
- **future** (미구현): 위와 동일한 `BasePolicy` 인터페이스를 구현하는 실제 policy 어댑터. 아래 "다음 단계" 참고.

### 7-DoF Action / ActionAdapter / RobotCommand

Policy는 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` 형태의 7-DoF action을 출력합니다. `action_adapter/adapter_v0.py`의 `ActionAdapter.convert()`가 이를 `RobotCommand`(`target_dx/dy/dz/droll/dpitch/dyaw`, `gripper_command`)로 변환합니다. 이 파일은 dataset replay 검증에도 그대로 재사용되기 때문에 이후 단계에서 수정하지 않는 것을 원칙으로 하고 있습니다.

### SafetyGate

`safety/safety_gate.py`의 `SafetyGate`가 `RobotCommand`를 실제로 적용하기 전에 카메라 프레임을 보고 hazard 여부를 확인합니다.

```text
frame → SafetyMonitor.check(frame) → SafetyDecision(emergency_stop, reason, detections)
→ allowed=False면 명령 차단 (task_status=blocked_by_safety)
→ allowed=True면 PyBulletPandaBackend.apply_command() 실행
```

`--safety-monitor none/mock/onnx` 세 가지 모드를 지원합니다(`none`은 점검 자체를 건너뜀). 별도 데모(`run_task_goal_real2sim_panda_interrupt_demo.py`)에서는 이동 중간에도 주기적으로 점검하는 Layer 2(interruptible motion)까지 구현했습니다.

### Safety Pause/Resume (v0, mock-timed)

위의 `--safety-monitor` 경로는 hazard를 만나면 episode를 `blocked_by_safety`로 **종료**시킵니다. `--safety-mode pause-resume`는 이와 별개의, 종료 대신 **일시정지 후 재개**하는 경로입니다. 중요한 설계 원칙은: **Safety Pause/Resume은 VLA policy 바깥에 있다.** `DummyOpenVLAPolicy`(혹은 실제 VLA)는 "action을 제안"할 뿐이고, 그 action을 실제로 로봇에 적용할지는 전적으로 Safety Gate가 결정합니다. v0에서는 hand/person detector가 없으므로, `MockHandIntrusionMonitor`(`safety/mock_hand_intrusion_monitor.py`)가 `--mock-hand-start-step`/`--mock-hand-end-step` 구간 동안 손이 있는 것처럼 `emergency_stop=True`를 흉내 냅니다.

```text
매 control loop step:
1. robot_state 읽기
2. (필요하면) wrist observation 읽기
3. safety check: hand_intrusion_gate.check(frame, "safety_check")
4. 위험하면(hand_detected=True): policy를 아예 호출하지 않고, action도 적용하지 않고
   safety_pause/safety_still_paused 이벤트만 기록한 뒤 다음 step으로 continue
5. 안전하면: policy action 생성 -> ActionAdapter -> apply_command()
```

상태 머신은 `running -> paused_by_safety -> (resuming) -> running`으로 움직입니다. 손이 사라져도 곧바로 재개하지 않고, `--safety-resume-stable-steps`(기본 3)번 연속으로 `hand_detected=False`가 나와야 `safety_resume` 이벤트와 함께 다음 step부터 정상 control loop를 이어갑니다 -- 재개된 step은 policy phase/target을 그대로 유지하므로 episode를 처음부터 다시 실행하지 않습니다. pause/resume이 몇 번 일어나든 최종적으로 물체를 놓으면 `final_status=success`이며, pause 자체는 절대로 episode를 실패시키지 않습니다. wrist camera grasp refinement도 `hand_detected=True`이거나 아직 재개 대기(`resuming`) 중일 때는 적용하지 않습니다 -- 로봇이 멈춰 있는 동안 grasp target이 계속 바뀌지 않도록, refinement는 `safety_supervisor.is_running()`일 때만 시도되고, 재개 이후 새 observation으로 다시 시도할 수 있습니다.

이 상태 머신 자체는 `safety/safety_supervisor.py`의 `SafetySupervisor`가 소유합니다(`step(hand_detected, reason, still_paused_reason) -> Optional[dict]`) -- 원래 `run_full_recycling_cell_demo.py`의 control loop 안에 그대로 inline되어 있던 것을 동작 변경 없이 그대로 옮겼습니다. `SafetySupervisor`는 hand/hazard *탐지* 자체는 하지 않습니다(그건 여전히 `MockHandIntrusionMonitor`/`ExternalCameraHandSafetyMonitor`를 감싼 `SafetyGate`의 몫입니다) -- control loop가 `hand_detected` boolean을 넘겨주면 그것만 보고 pause/resume 여부와 이벤트를 결정합니다. 즉 "Policy proposes action. SafetySupervisor decides whether the action can be applied. RobotBackend executes only safe commands."

`--safety-mode`는 세 가지 값을 가집니다: `off`(기본값, 이 기능 이전과 완전히 동일한 동작), `block`(기존 `--safety-monitor mock/onnx` + `--simulate-hazard` 차단 경로를 그대로 유지, 회귀 없음), `pause-resume`(위에서 설명한 새 상태 머신). `--safety-monitor`/`--simulate-hazard`와 `--safety-mode`/`--mock-hand-intrusion`은 서로 독립적인 축이라 동시에 켤 수도 있습니다.

### External Camera Hand Safety Monitor (v1, real hand/arm intrusion)

v0의 `MockHandIntrusionMonitor`는 시간(step index)만 보고 손을 흉내 냈습니다. v1은 실제 외부 카메라 frame에서 손/팔을 검출합니다: `safety/external_camera_hand_monitor.py`의 `ExternalCameraHandSafetyMonitor`가 MediaPipe HandLandmarker(Tasks API, `weights/hand_landmarker.task` 사전학습 모델 -- 커스텀 학습 아님)로 손 landmark를 찾고, 그 landmark가 **ArUco 마커 4개로 정의된 작업공간 polygon** 내부에 들어오는지 `cv2.pointPolygonTest`로 판정합니다. 분리수거 작업의 핵심은 사람 전체가 보이는지가 아니라 테이블 위로 손/팔 일부가 들어오는지이므로, person detector가 아니라 **hand/arm intrusion detector** 중심으로 설계했습니다. 손이 화면에 보여도 작업공간 polygon 밖이면 pause하지 않습니다(`hand_detected=True, hand_in_workspace=False`).

`ExternalCameraHandSafetyMonitor`는 `MockHandIntrusionMonitor`와 정확히 같은 duck-typed 인터페이스(`set_step()`, `check(frame) -> SafetyDecision`)를 구현하므로, `run_full_recycling_cell_demo.py`의 pause/resume control loop 코드는 전혀 바뀌지 않고 `--hand-safety-source mock` -> `external-camera`로 SafetyMonitor 구현체만 바뀝니다:

```text
--safety-mode pause-resume --hand-safety-source mock --mock-hand-intrusion   (v0, 시간 기반)
--safety-mode pause-resume --hand-safety-source external-camera             (v1, 실제 손 검출)
```

작업공간 polygon은 ArUco Real2Sim mapping이 이미 검출한 marker 4개의 pixel 중심(`marker_centers_px`, `required_marker_ids` 순서 = front_left/front_right/back_right/back_left)을 그대로 재사용합니다(`build_hand_safety_workspace_polygon()`) -- hand safety monitor가 자체적으로 marker를 다시 검출하지 않습니다. ArUco 마커를 못 찾으면 `configs/hand_safety_config.json`의 `roi` 블록으로 fallback하고, 그것도 없으면 `workspace_valid=False`로 기록되어 절대 pause하지 않습니다(오탐으로 인한 불필요한 정지보다 "판정 불가능은 안전 쪽으로 무시"를 선택). 외부 카메라는 `--image-source webcam`일 때 매 control loop step마다 새 frame을 읽어(`WebcamSource`를 별도로 열어 둔 채 유지) 실제 실시간 감시를 흉내 냅니다; `--image-source image`(정적 이미지)에서는 라이브 재촬영이 불가능하므로 최초 프레임을 계속 재사용합니다.

`SafetyDecision`에 `severity`(`"high"`/`"none"`) 필드가 추가되어 hand intrusion 같은 고심각도 이벤트를 표시할 수 있습니다. `--save-hand-safety-debug-images`를 주면 매 step `results/safety_hand_debug/hand_safety_step_<step>.png`에 workspace polygon/손 landmark/bbox/침입 지점/`SAFE`|`HAND_IN_WORKSPACE` 상태 텍스트를 그린 디버그 이미지를 저장합니다.

향후 방향: wrist camera 기반 hand safety(그리퍼에 더 가까운 손 침입 감지), 외부+wrist 카메라 fusion, 실제 하드웨어에서 이 SafetyDecision을 ROS2 e-stop/hardware stop 브릿지로 전달하는 것.

### PyBullet Panda Backend

`robot_sim/pybullet_panda_backend.py`의 `PyBulletPandaBackend`가 실제 Franka Panda URDF를 로드해서 IK 기반 end-effector 이동, gripper open/close, 물체 grasp/place, `task_status` 관리를 담당합니다. 두 인터페이스를 동시에 구현합니다: 기존 `SimulatorBackend`(`reset`, `apply_command`, `get_state`, `close`, 이전 단계의 단순 sphere backend `robot_sim/pybullet_backend.py`와 같은 방식으로 다룰 수 있음)와 하드웨어 이식을 겨냥한 `RobotBackend`(`move_end_effector_to`/`open_gripper`/`close_gripper`/`shutdown` 포함 -- 위 "Hardware-Portable Backend Abstraction" 참고). `shutdown()`은 기존 `close()`의 얇은 alias일 뿐이라 `close()`를 쓰는 기존 코드는 전혀 바뀌지 않습니다.

### TrajectoryRecorder / Dataset Export / Replay Validation

`--record`를 켜면 로봇 trajectory뿐 아니라 external camera 관찰 → YOLO detection → Real2Sim(ROI/ArUco) mapping → wrist camera refinement → robot 실행 → 최종 결과로 이어지는 전체 perception-to-action 체인이 episode마다 `metadata.json`으로 함께 저장되고, `benchmark/inspect_recorded_episode.py`로 요약을 확인할 수 있습니다. 자세한 스키마와 파일 구조는 [dataset_pipeline.md](dataset_pipeline.md)를 참고하세요.

## 다음 단계: FastAPI dummy OpenVLA server 연결

`DummyOpenVLAPolicy`는 `BasePolicy` 인터페이스만 만족하면 어떤 구현으로도 교체할 수 있도록 설계했습니다. 다음 단계로 예정된 경로는 다음과 같습니다.

```text
1. openvla_server_dummy/ (기존 FastAPI dummy 서버)를
   PolicyInput -> 7-DoF action 형식에 맞게 요청/응답 스펙 정리
2. BasePolicy를 구현하는 OpenVLAServerClientPolicy 추가
   - predict_action()이 내부적으로 FastAPI 서버에 HTTP 요청
   - 응답을 PolicyOutput(action, phase, done, info)으로 변환
3. run_full_recycling_cell_demo.py의 --policy 선택지에
   scripted / dummy-openvla / openvla-server 를 추가
4. 통신 실패, timeout 등 실패 케이스에 대한 안전한 fallback 정의
   (예: 통신 실패 시 emergency stop과 동일하게 처리)
5. 이 클라이언트 policy로 기존 online control loop 회귀 테스트 재실행
```

이 단계에서도 실제 OpenVLA 모델을 학습/서빙하지는 않고, "정책을 네트워크 너머의 서버로 분리해도 control loop가 그대로 동작하는지"를 검증하는 것이 목표입니다.

## 다음 단계: Real2Sim mapping 정교화 (ArUco + wrist camera)

최종적으로 그리려는 방향은 다음 3단계이며, **이번 단계로 3번까지 모두 구현했습니다** (3번은 PyBullet segmentation/depth 기반의 v1 -- 아래 참고).

```text
1. [구현됨] 외부 카메라 + ArUco marker 4개로 테이블 기준 좌표계를 매 프레임 계산
   -> ArUcoTableMapper: marker 4개의 이미지 좌표 <-> 알려진 sim_xy로 homography 계산
   -> ROI/axis_mapping 수동 calibration과 달리 카메라가 살짝 움직여도 marker만 보이면 재보정 불필요
   -> 이것이 "coarse global localization": 물체가 테이블 어디에 있는지 로봇 기준 좌표로 대략 알려줌
2. [구현됨] PyBullet Eye-in-Hand wrist camera로 로봇이 물체 가까이 다가간 뒤 위치를 다시 관찰
   -> 이것이 "local perception": 1단계의 전역 추정치와 별개로, 로봇 손목 카메라가
      근접 거리에서 물체를 다시 보고 위치를 추정 (--wrist-camera-mode observe)
3. [v1 구현됨] wrist camera 추정값으로 grasp target을 보정(refine)해서 Panda pick-and-place 수행
   -> --wrist-camera-mode refine, 신뢰 조건을 만족할 때만 보정, 실패 시 coarse target으로 fallback
```

ROI linear mapping(`CalibratedImageToSimMapper`)은 여전히 baseline으로 남아 있습니다 -- 사람이 손으로 `image_roi`를 눈금 맞추듯 튜닝하는 대신 빠르게 확인/디버깅하고 싶을 때, 혹은 ArUco marker를 아직 붙이지 않은 환경에서 사용합니다. ArUco 방식은 카메라 위치/각도 변화에 baseline보다 강하지만, 여전히 "테이블 평면 + object_z 고정" 가정 위에 있고 marker의 3D pose(rvec/tvec, 카메라 intrinsics 필요)까지는 추정하지 않습니다.

### Wrist camera (v0 observe-only + v1 grasp refinement)

`robot_sim/pybullet_wrist_camera.py`의 `PyBulletWristCamera`가 Panda의 `panda_grasptarget` 링크(`end_effector_link_index=11`)에 매 프레임 다시 부착되는 가상 카메라입니다. 외부 ArUco 카메라와 달리 world-fixed가 아니라 end-effector의 현재 world pose(`p.getLinkState`)에 `configs/wrist_camera_config.json`의 `camera_local_position`/`camera_forward_local`/`camera_up_local`을 곱해서 매번 새로 카메라 pose를 계산합니다(`p.computeViewMatrix`/`computeProjectionMatrixFOV`/`getCameraImage`).

`estimate_object_position_from_segmentation()`이 `p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX` segmentation buffer에서 대상 `object_body_id`에 해당하는 픽셀만 골라 bbox/center/median depth를 구하고, 카메라의 world position + forward/right/up 기저벡터 + 수직 FOV로 center pixel을 depth만큼 world 좌표로 unproject합니다.

`--wrist-camera-mode`는 세 단계입니다.

- **off** (기본값): 기존 동작 그대로, wrist camera를 아예 사용하지 않음.
- **observe**: Real2Sim mapping으로 정한 `sim_position` 위쪽(`WRIST_CAMERA_OBSERVE_HEIGHT=0.25`)으로 end-effector를 한 번 이동시켜 wrist camera로 관찰하고 `estimated_world_position`을 실제 `sim_position`과 비교해서 출력만 합니다. grasp target은 바뀌지 않습니다.
- **refine** (`run_dummy_openvla_policy()`에서만 연결, `--policy scripted`에서는 무시하고 안내 메시지만 출력): `move_above_bin` 오실레이션 방지 로직과 비슷하게, 정책 루프 안에서 `policy.phase == "move_to_object"`이고 `policy.last_info["distance_to_target"] <= --refine-distance-threshold`(기본 0.08)가 될 때 **에피소드당 정확히 한 번** `robot_sim/pybullet_wrist_camera.py`의 `refine_target_with_wrist_camera()`를 호출합니다. wrist camera가 현재 위치에서 렌더링한 뒤 segmentation/depth로 물체 위치를 추정하고, 신뢰 조건(`object_visible`, `object_pixel_count >= min_object_pixels`, `xy_delta_from_coarse <= max_refinement_delta`)을 모두 만족하면 `--wrist-refinement-policy`(`blend`/`override`, 기본 `blend`, `alpha=0.7`)에 따라 x/y만 보정한 새 target을 이후 스텝의 `PolicyInput.target_object_position`으로 사용합니다(z는 항상 원래 Real2Sim `object_z` 유지). 신뢰 조건을 만족하지 못하면 원래 coarse target을 그대로 사용합니다(fallback, `debug["fallback_reason"]`에 사유 기록). 어떤 target을 넘겨줄지는 항상 control loop가 결정하고, policy는 그 target을 향해 그대로 움직일 뿐입니다.

이 구조는 최종 아키텍처의 의도를 그대로 반영합니다: 외부 ArUco 카메라 = coarse global localization("테이블 어디쯤"), wrist camera observe = 로컬 관찰, wrist camera refine = 그 로컬 관찰을 실제 grasp target 보정에 사용. v1은 여전히 PyBullet의 완전한 ground-truth segmentation/depth 기반이라 노이즈가 거의 없고, 실제 로봇에서는 이 자리를 RGB-D 센서나 학습된 perception 모델로 대체하게 됩니다.

### VLA-ready control loop: `--policy-observation-source wrist`

`DummyOpenVLAPolicy`는 real OpenVLA를 대신하는 placeholder policy입니다 -- image content를 실제로 해석(learned visual reasoning)하지는 않습니다. 하지만 real VLA를 나중에 꽂아 넣으려면, control loop 자체가 "매 step 카메라 관찰을 policy에 넘기는" 구조여야 합니다. `--policy-observation-source {none, wrist}`가 그 구조를 만듭니다.

- **none** (기본값): 기존 그대로 `PolicyInput.image`에 초기 detection에 쓰인 정적 프레임(`task_frame`)이 매 step 그대로 들어감(v0부터 있던 동작).
- **wrist**: 매 policy step마다 `PyBulletWristCamera.render()` + `estimate_object_position_from_segmentation()`을 호출해서, 그 프레임의 RGB를 `PolicyInput.image`로, 그리고 `object_visible`/`object_pixel_count`/`estimated_world_position`을 `PolicyInput.visual_observation`으로 넣습니다(`PolicyInput.observation_source="wrist"`). `--wrist-camera-mode refine`이 같은 step에서 트리거되면, 같은 프레임을 재사용해서 wrist camera를 이중으로 렌더링하지 않습니다(`refine_target_with_wrist_camera(..., frame=이미_렌더링된_frame)`).

`policy/policy_types.py`의 `PolicyInput`에 `observation_source`/`visual_observation` 필드를 추가했습니다(둘 다 optional, 기본 `None`이라 기존 호출부는 전혀 안 바뀝니다). `DummyOpenVLAPolicy.predict_action()`은 아주 얇은 wrapper로 감쌌습니다 -- 기존 phase 로직(`_predict_phase_action`으로 이름만 옮김)은 한 글자도 안 바꾸고, 그 결과에 `used_image_input`/`image_shape`/`observation_source`/(있으면) `object_visible`/`estimated_world_position`을 `info`에 덧붙이기만 합니다. 즉 **이번 v0은 policy의 action 자체를 이미지로 바꾸지 않습니다** -- 목표는 오직 "VLA가 들어올 수 있는 입출력 루프"를 만드는 것이고, 나중에 real OpenVLA가 같은 `predict_action(PolicyInput) -> PolicyOutput` 인터페이스로 `DummyOpenVLAPolicy`를 대체하면 이 루프가 그대로 재사용됩니다.

`--record-policy-observations`(`--record`와 함께 사용)를 켜면 wrist observation이 일어난 step마다 `episode.json`의 해당 step에 `policy_input`/`wrist_observation`/`policy_output` 정보가 `extra`로 남고, `--record-images`까지 있으면 `--policy-observation-save-interval`(기본 5) step마다 하나씩 wrist RGB 프레임이 `frames/wrist_policy_step_<step>.png`로 저장됩니다.

### Policy backend: `--policy-backend local-dummy | fastapi-dummy`

같은 `--policy dummy-openvla`가 두 가지 **backend**로 실행될 수 있습니다 -- policy 종류(dummy-openvla)와 그 policy를 어디서 실행하는지(local-dummy/fastapi-dummy)를 분리했습니다.

- **local-dummy** (기본값): 지금까지처럼 `policy/dummy_openvla_policy.py`의 `DummyOpenVLAPolicy`를 in-process로 직접 호출합니다. deterministic placeholder.
- **fastapi-dummy**: 매 step `PolicyInput`을 `openvla_server_dummy/dummy_server.py`(FastAPI)의 `POST /predict`로 전송하고, 응답 action으로 기존 `ActionAdapter`/`RobotCommand` 흐름을 그대로 탑니다. `policy/fastapi_vla_policy_client.py`의 `FastAPIVLAPolicyClient`가 `BasePolicy` 인터페이스(`reset()`/`predict_action()`)를 구현해서, control loop 입장에서는 local-dummy와 완전히 동일하게 보입니다.

**두 backend가 항상 같은 결과를 내는 이유**: `dummy_server.py`가 새로운 phase 로직을 재구현하지 않고, `DummyOpenVLAPolicy`를 **서버 프로세스 안에서 그대로** 인스턴스화해서 호출합니다. 즉 local-dummy와 fastapi-dummy는 물리적으로 다른 프로세스(in-process 함수 호출 vs. HTTP 요청)에서 실행될 뿐, 실제로 action을 계산하는 코드는 완전히 동일합니다. 이 덕분에 같은 full demo에서 두 backend 모두 `final_status=success`가 나옵니다(둘 다 실측 확인, 8~9번 참고).

**request/response**:

```json
// POST /predict request
{
  "instruction": "...", "robot_state": {...}, "task_goal": {...},
  "target_object_position": [...], "bin_position": [...],
  "step_index": 12, "phase": "move_to_object",
  "observation_source": "wrist",
  "visual_observation": {"object_visible": true, "object_pixel_count": 1234, "estimated_world_position": [...]},
  "image": {"encoding": "jpg_base64", "shape": [240, 320, 3], "data": "..."}
}
// response
{"action": [...], "phase": "move_to_object", "done": false, "info": {"policy_backend": "fastapi-dummy", ...}}
```

`image`는 numpy 배열을 그대로 JSON에 넣지 않고 `FastAPIVLAPolicyClient._encode_image()`가 JPEG(quality 조절 가능, 기본 80)로 인코딩한 뒤 base64 문자열로 전송하고, 서버가 다시 디코딩합니다(둘 다 image content를 실제로 해석하지는 않고, `DummyOpenVLAPolicy`의 image-input bookkeeping에만 씀).

`FastAPIVLAPolicyClient.predict_action()`이 요청 왕복 시간을 측정해서 `PolicyOutput.info["inference_latency_ms"]`에 기록합니다(향후 real VLA 모델의 실제 추론 시간과 비교할 수 있는 자리를 미리 만들어 둔 것). connection refused/timeout/invalid response/action 누락은 모두 traceback 없이 사람이 읽을 수 있는 메시지로 안내하고, `run_full_recycling_cell_demo.py`는 PyBullet을 켜기 전에 `/health`를 먼저 확인해서 서버가 없으면 조기에 `FAIL`로 끝납니다.

`GET /health`, `POST /reset`(episode 시작 시 서버의 `DummyOpenVLAPolicy` phase를 리셋)도 있습니다. 서버는 단일 글로벌 policy 인스턴스만 유지하므로, 한 번에 하나의 episode만 구동하는 이번 v0 용도로 충분하고 동시 다중 episode 서빙은 지원하지 않습니다.

**향후**: `openvla_server_dummy/dummy_server.py`의 `/predict` 구현부만 실제 OpenVLA 추론으로 바꾸면, `FastAPIVLAPolicyClient`/`run_full_recycling_cell_demo.py` 어느 쪽도 변경할 필요가 없습니다 -- 지금 이 v0의 request/response 스키마가 바로 그 자리입니다.
