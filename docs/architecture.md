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

## 구성 요소

### TaskGoal Parser

`llm_agent/rule_based_parser.py`의 `RuleBasedTaskGoalParser`가 한국어 명령을 `TaskGoal`(`action`, `target_object`, `target_bin`, `instruction`, `constraints`)로 변환합니다. 실제 LLM을 사용하지 않는 rule-based 구현이며, 나중에 LLM Agent로 교체할 수 있도록 인터페이스만 고정해 두었습니다.

### YOLO Detector

`perception/onnx_yolo_detector.py`의 `ONNXYOLODetector`가 ONNX Runtime으로 이미지에서 객체를 탐지해 `Detection`(label, confidence, bbox_xyxy) 리스트를 반환합니다.

### Target Selector

`real2sim/recyclable_object_mapper.py`의 `RecyclableObjectMapper.select_recyclable_by_target()`이 탐지 결과 중 `TaskGoal.target_object`와 일치하는 대상을 고릅니다.

### Real2Sim Mapper

두 가지 매퍼가 있습니다.

- `real2sim/image_to_sim_mapper.py`의 `ImageToSimMapper` (v0): bbox 중심 픽셀 좌표를 image_x→sim_x, image_y→sim_y로 그대로 늘려 붙이는 단순 선형 매핑. `run_task_goal_real2sim_panda_demo.py` 계열 등 이전 데모들이 계속 사용합니다.
- `real2sim/calibrated_image_to_sim_mapper.py`의 `CalibratedImageToSimMapper` (v1, `run_full_recycling_cell_demo.py`와 `benchmark/probe_real2sim_mapping.py`가 `--real2sim-calibration`으로 사용): ROI(`image_roi`), 축 매핑(`axis_mapping`), ROI-clamp 여부(`clamp_to_roi`)를 `configs/real2sim_webcam_calibration.json`으로 명시적으로 분리한 calibrated 버전. depth 카메라나 depth estimation 없이, "카메라가 고정되어 있고 물체는 테이블 평면 위에 있다"는 가정만으로 물체가 카메라에서 가까워지거나 멀어질 때(대개 image y가 변함)를 Panda의 forward/depth 축(`sim_x`)에, 좌우 이동(image x)을 `sim_y`에 반영합니다(기본 `axis_mapping`: `image_y_to_sim_x=true`, `image_x_to_sim_y=true`, `invert_image_x`/`invert_image_y`로 방향 반전 가능). ROI 밖 bbox는 `clamp_to_roi=true`면 가장 가까운 ROI 경계로 clamp하고(`false`면 그대로 workspace 밖으로 나갈 수 있음을 경고), 항상 `clamped` 여부를 debug에 기록합니다. `map_bbox_to_sim()`은 `mapped_position`과 함께 `bbox_center`/`bbox_size`/`bbox_area_ratio`/`normalized_center`/`image_roi`/`clamped` 등을 담은 debug dict를 반환해서 `print_mapping_debug()`로 매 실행마다 원인 분석이 가능하고, `draw_roi_rectangle()`로 `--save-debug-image` 이미지에 ROI 사각형도 함께 그립니다.

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

### PyBullet Panda Backend

`robot_sim/pybullet_panda_backend.py`의 `PyBulletPandaBackend`가 실제 Franka Panda URDF를 로드해서 IK 기반 end-effector 이동, gripper open/close, 물체 grasp/place, `task_status` 관리를 담당합니다. `SimulatorBackend` 인터페이스(`reset`, `apply_command`, `get_state`, `close`)를 구현하고 있어서, 이전 단계의 단순 sphere backend(`robot_sim/pybullet_backend.py`)와 같은 방식으로 다룰 수 있습니다.

### TrajectoryRecorder / Dataset Export / Replay Validation

자세한 내용은 [dataset_pipeline.md](dataset_pipeline.md)를 참고하세요.

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

## 다음 단계: Real2Sim mapping 정교화 (ArUco/AprilTag + wrist camera)

현재 `CalibratedImageToSimMapper`는 여전히 "고정 카메라 + 테이블 평면" 가정 위의 2D 선형 매핑이고, 실제 depth 정보는 전혀 사용하지 않습니다. 최종적으로 그리려는 방향은 다음 2단계입니다(아직 미구현).

```text
1. 외부 카메라 + ArUco/AprilTag로 현실 물체(또는 테이블 기준점)의 전역 위치를 시뮬레이터에 매핑
   -> 마커 기반 pose 추정으로 image_roi/axis_mapping 같은 수동 calibration을 대체
2. PyBullet Eye-in-Hand wrist camera로 로봇이 물체 가까이 다가간 뒤 위치를 재확인/보정
   -> 1단계의 전역 추정치를 근접 관측으로 보정해서 grasp 정밀도를 높임
3. 보정된 위치로 Panda pick-and-place 수행
```

이번 v1 calibration(ROI + axis mapping + debug)은 이 방향으로 가기 전에 "지금 있는 단순 매핑이 왜 이런 위치를 내놓는지 눈으로 확인하고 설정을 바꿔볼 수 있게" 만드는 중간 단계입니다. ArUco/AprilTag, wrist camera, monocular/스테레오 depth 추정은 이번 단계에 포함하지 않았습니다.
