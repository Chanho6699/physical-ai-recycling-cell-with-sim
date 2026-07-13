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

### PyBullet Panda Backend

`robot_sim/pybullet_panda_backend.py`의 `PyBulletPandaBackend`가 실제 Franka Panda URDF를 로드해서 IK 기반 end-effector 이동, gripper open/close, 물체 grasp/place, `task_status` 관리를 담당합니다. `SimulatorBackend` 인터페이스(`reset`, `apply_command`, `get_state`, `close`)를 구현하고 있어서, 이전 단계의 단순 sphere backend(`robot_sim/pybullet_backend.py`)와 같은 방식으로 다룰 수 있습니다.

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
- **refine** (`run_dummy_openvla_policy()`에서만 연결, `--policy scripted`에서는 무시하고 안내 메시지만 출력): `move_above_bin` 오실레이션 방지 로직과 비슷하게, 정책 루프 안에서 `policy.phase == "move_to_object"`이고 `policy.last_info["distance_to_target"] <= --refine-distance-threshold`(기본 0.08)가 될 때 **에피소드당 정확히 한 번** `robot_sim/pybullet_wrist_camera.py`의 `refine_target_with_wrist_camera()`를 호출합니다. wrist camera가 현재 위치에서 렌더링한 뒤 segmentation/depth로 물체 위치를 추정하고, 신뢰 조건(`object_visible`, `object_pixel_count >= min_object_pixels`, `xy_delta_from_coarse <= max_refinement_delta`)을 모두 만족하면 `--wrist-refinement-policy`(`blend`/`override`, 기본 `blend`, `alpha=0.7`)에 따라 x/y만 보정한 새 target을 이후 스텝의 `PolicyInput.target_object_position`으로 사용합니다(z는 항상 원래 Real2Sim `object_z` 유지). 신뢰 조건을 만족하지 못하면 원래 coarse target을 그대로 사용합니다(fallback, `debug["fallback_reason"]`에 사유 기록). `DummyOpenVLAPolicy` 자체는 전혀 수정하지 않았습니다 -- 어떤 target을 넘겨줄지는 항상 control loop가 결정하고, policy는 그 target을 향해 그대로 움직일 뿐입니다.

이 구조는 최종 아키텍처의 의도를 그대로 반영합니다: 외부 ArUco 카메라 = coarse global localization("테이블 어디쯤"), wrist camera observe = 로컬 관찰, wrist camera refine = 그 로컬 관찰을 실제 grasp target 보정에 사용. v1은 여전히 PyBullet의 완전한 ground-truth segmentation/depth 기반이라 노이즈가 거의 없고, 실제 로봇에서는 이 자리를 RGB-D 센서나 학습된 perception 모델로 대체하게 됩니다.
