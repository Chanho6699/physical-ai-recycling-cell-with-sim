# Physical AI Recycling Cell

> This project is not just a PyBullet demo -- it is a hardware-portable Physical AI / VLA-ready software brain, currently validated end-to-end in simulation, that maps a natural-language instruction and camera input into a Panda robot pick-and-place action sequence.

자연어 명령과 카메라 이미지를 입력받아, 인식 → Real2Sim 좌표 변환 → policy → 로봇 제어 → 안전 점검(pause/resume 포함) → 데이터 기록까지 이어지는 recycling cell 파이프라인을 구현한 개인 프로젝트입니다. 현재는 PyBullet 시뮬레이션 위에서 전체 파이프라인을 검증하고 있지만, **핵심 목표는 시뮬레이션 데모가 아니라 실제 하드웨어 로봇팔에 이식 가능한 구조**입니다 -- 그래서 로봇/카메라/policy/safety를 모두 인터페이스 뒤에 두고, PyBulletPandaBackend/iVCam/dummy VLA/mock-hand-safety를 각각 실제 하드웨어 구현체로 교체할 수 있도록 경계를 정리했습니다(자세한 내용은 [docs/hardware_portability.md](docs/hardware_portability.md) 참고).

## 1. 한 줄 소개

"플라스틱 병을 플라스틱 수거함에 넣어줘" 같은 한국어 명령과 이미지 한 장으로 시작해서, 인식 → 좌표 변환 → 로봇 제어 → 안전 점검 → 데이터 기록까지 이어지는 recycling cell 파이프라인을 PyBullet 시뮬레이션으로 구현하고 검증했으며, 각 단계를 실제 하드웨어로 교체 가능한 인터페이스(RobotBackend/CameraBackend/PolicyBackend/SafetySupervisor) 뒤에 두도록 정리한 프로젝트입니다.

## 2. 문제 정의

재활용 분류는 보통 "이미지에서 어떤 쓰레기인지 분류하는" YOLO 수준의 인식 문제로 다뤄집니다. 하지만 실제 recycling cell은 인식만으로 끝나지 않고, 인식 결과를 실제 로봇 동작(집기, 옮기기, 놓기)으로 이어야 하고, 그 과정에서 안전 점검과 데이터 기록이 함께 필요합니다.

이 프로젝트는 "인식 → 조작(manipulation) → 안전 → 데이터"로 이어지는 최소한의 end-to-end 구조를 로컬 환경에서 직접 구현하고 검증하는 것을 목표로 합니다.

## 3. 핵심 목표

- 자연어 명령을 구조화된 작업 목표(TaskGoal)로 변환한다.
- 이미지 기반 YOLO 인식 결과를 시뮬레이션 로봇 workspace 좌표로 매핑한다(Real2Sim).
- 실제 Franka Panda URDF 기반 IK 제어로 pick-and-place를 수행한다.
- 로봇 동작 실행 전/중에 안전 점검(SafetyGate)을 거치도록 한다.
- 실행 과정을 기록하고, LeRobot 스타일 데이터셋으로 변환하고, 재현 검증까지 수행한다.
- 이 모든 것이 나중에 실제 OpenVLA 정책으로 교체 가능한 구조 위에서 동작하도록 한다.

## 4. 전체 파이프라인

```text
Instruction + Image
        ↓
TaskGoal Parser (rule-based Korean NLU)
        ↓
YOLO/ONNX Detector
        ↓
Target Selector (recyclable object matching)
        ↓
Real2Sim Mapper (image bbox → Panda workspace 좌표)
        ↓
Policy (scripted / dummy-openvla / future: real OpenVLA)
        ↓
7-DoF Action
        ↓
ActionAdapter
        ↓
RobotCommand
        ↓
SafetyGate (optional)
        ↓
PyBullet Panda Backend (IK + gripper + grasp/place)
        ↓
TrajectoryRecorder
        ↓
Dataset Export / Replay Validation
```

세부 구성 요소 설명은 [docs/architecture.md](docs/architecture.md)에 정리되어 있습니다.

## 5. 주요 기능

| 영역 | 내용 |
|---|---|
| TaskGoal parsing | 한국어 명령을 `action` / `target_object` / `target_bin`으로 구조화 (rule-based) |
| YOLO/ONNX detection | ONNX Runtime 기반 YOLO로 이미지에서 재활용 대상 탐지 |
| Target selection | 탐지 결과 중 TaskGoal의 target_object와 일치하는 대상 선택 |
| Real2Sim mapping | 이미지 bbox 중심 좌표를 Panda workspace 3D 좌표로 변환 |
| PyBulletPandaBackend | Franka Panda URDF, IK 기반 end-effector 이동, gripper open/close, grasp/place 판정, task_status 관리 (`RobotBackend`/`SimulatorBackend` 인터페이스 구현) |
| Wrist camera + grasp refinement | end-effector에 붙은 가상 eye-in-hand 카메라로 물체 위치를 근거리에서 재추정, grasp target 보정 |
| VLA-ready per-step observation | 매 policy step마다 wrist camera RGB를 `PolicyInput.image`로 전달하는 online control loop |
| PolicyBackend (local-dummy / fastapi-dummy / real-vla) | `DummyOpenVLAPolicy`(in-process), `FastAPIVLAPolicyClient`(HTTP, `openvla_server_dummy/dummy_server.py`), `RealVLAPolicyClient`(HTTP + config 기반 image preprocessing/action postprocessing/fallback, `openvla_server_dummy/real_vla_compatible_server.py`)가 동일한 `BasePolicy`/`PolicyBackend` 인터페이스를 구현 -- 실제 OpenVLA 서버로 교체 가능한 자리 |
| SafetyGate | action 실행 전(및 이동 중) hazard 여부를 점검해서 차단 (mock / ONNX YOLO 모두 지원) |
| Safety Pause/Resume + SafetySupervisor | hand 침입 시 로봇 action 적용을 일시정지하고, 손이 사라지면 종료 없이 이어서 진행 (mock-timed v0, 실제 외부 카메라 MediaPipe hand detector v1) |
| CameraBackend / RobotBackend 추상화 | 외부 카메라(iVCam/webcam/정적 이미지)/로봇(PyBullet/향후 실제 하드웨어)을 인터페이스 뒤에 두어 하드웨어 이식 경로를 명확히 함 |
| TrajectoryRecorder + perception-to-action episode metadata | raw episode(JSON) + frame(PNG) + 인식/매핑/safety 전체 체인 기록 |
| LeRobot 스타일 dataset exporter | raw episode를 image/state/instruction/action 중심 JSONL로 변환 |
| Dataset replay validator | 저장된 action이 PyBullet Panda에서 물리적으로 재현되는지 검증 |
| OpenVLA 스타일 action adapter | dataset action ↔ `[dx, dy, dz, droll, dpitch, dyaw, gripper]` 7-DoF 변환 검증 |
| Full demo runner | 위 전체 흐름을 하나의 진입점으로 실행 (`benchmark/run_full_recycling_cell_demo.py`) |

## 6. 대표 실행 명령

가장 대표적인 실행(전체 파이프라인 + 기록):

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --record --record-images \
  --headless
```

scripted policy로 실행:

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy scripted \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --headless
```

mock safety hazard 차단 시나리오:

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --safety-monitor mock \
  --simulate-hazard \
  --headless
```

더 많은 실행 명령(개별 단계 데모, dataset export/replay 등)은 [docs/demo_commands.md](docs/demo_commands.md)를 참고하세요.

## 7. 구현된 policy 종류

- **scripted**: `move_end_effector_to(object) → close_gripper → move_end_effector_to(bin 위) → open_gripper` 순서로 동작하는 결정론적(deterministic) 4단계 IK 이동. 별도 모델 없이 좌표 계산만으로 동작합니다.
- **dummy-openvla**: 실제 OpenVLA 없이 rule-based phase state machine으로 동작하는 scripted oracle policy(`DummyOpenVLAPolicy`)입니다. `PolicyInput → predict_action() → 7-DoF action → ActionAdapter → RobotCommand`라는, 실제 정책을 나중에 그대로 꽂아 넣을 수 있는 online control loop 구조를 검증하기 위한 것입니다.
  - phase: `move_to_object → close_gripper → lift_object → move_above_bin → open_gripper → done`
  - 기본 실행/기록 실행/mock hazard 시나리오 모두 PASS로 검증했습니다.
- **fastapi-dummy** (`--policy-backend fastapi-dummy`, `policy/fastapi_vla_policy_client.py::FastAPIVLAPolicyClient`): `DummyOpenVLAPolicy`를 그대로 호스팅하는 FastAPI 서버(`openvla_server_dummy/dummy_server.py`)에 HTTP로 요청하는 클라이언트입니다. `local-dummy`와 동일한 `BasePolicy`/`PolicyBackend` 인터페이스를 구현하므로 control loop는 어느 backend인지 신경 쓰지 않고, 실제로 같은 `policy_steps`로 끝납니다(같은 phase 엔진을 실행하기 때문).
- **real-vla** (`--policy-backend real-vla`, `policy/real_vla_policy_client.py::RealVLAPolicyClient`): 실제 OpenVLA/VLA 모델 서버가 들어올 수 있는 adapter 계층입니다. 서버 주소/이미지 인코딩/action clipping을 `configs/real_vla_backend_config.json`에서 읽고, VLA 응답 action은 항상 `policy/vla_action_postprocessor.py`로 검증/후처리한 뒤에만 적용됩니다. 서버 연결 실패 시 `--real-vla-fallback-backend`(기본 `local-dummy`)로 자동 전환합니다. 로컬 GPU/VRAM 부담을 피하기 위해 이번 v0은 실제 대형 모델을 로딩하지 않고, adapter-test mock 서버(`openvla_server_dummy/real_vla_compatible_server.py`, 내부적으로 `DummyOpenVLAPolicy` 재사용)로 스키마를 검증했습니다.
  - **Colab VLA Server Spike**: 로컬 머신에 GPU가 없어도 실제 OpenVLA 로딩을 시도해볼 수 있도록, Google Colab GPU 런타임에 같은 adapter 스키마를 쓰는 임시 서버(`openvla_server_real/colab_vla_server.py`, `notebooks/colab_vla_server_spike_v0.ipynb`)를 ngrok/cloudflared tunnel로 연결하는 spike를 추가했습니다. Colab 서버는 action proposal만 반환할 뿐 로봇을 직접 실행하지 않으며, 세션이 끊겨도 로컬 fallback이 그대로 동작합니다. 자세한 내용은 [docs/colab_vla_server_spike.md](docs/colab_vla_server_spike.md) 참고.
- **future**: 위 policy들과 동일한 `BasePolicy`/`PolicyBackend` 인터페이스를 구현하는 실제 OpenVLA 모델로 서버 내부를 교체. 아직 구현하지 않았습니다(Colab spike로 GPU 확보 경로만 검증).

## 8. dataset / replay 검증

- `TrajectoryRecorder`로 저장한 raw episode를 LeRobot 스타일 JSONL(`meta/info.json`, `meta/episodes.jsonl`, `data/episodes.jsonl`)로 변환하는 exporter를 구현했습니다.
- 변환된 dataset의 action을 다시 PyBullet Panda에 그대로 재생(replay)해서, 기록된 action이 실제로 동일한 결과(성공/실패)를 재현하는지 검증하는 replay validator를 구현했습니다.
- success episode만 모은 dataset에서 `replay_success_rate=1.00`을 확인했습니다.
- dataset action을 OpenVLA 스타일 7-DoF action으로 변환하는 어댑터(`OpenVLAActionAdapter`)도 같은 replay 구조로 검증했습니다.
- 자세한 데이터 흐름과, replay validator가 실제로 잡아낸 버그 사례는 [docs/dataset_pipeline.md](docs/dataset_pipeline.md)에 정리했습니다.

## 9. safety 기능

- `SafetyGate`는 action을 실제로 실행하기 전에 카메라 프레임을 보고 위험 요소(기본: 사람)가 있는지 확인합니다.
- `--safety-monitor none/mock/onnx` 세 가지 모드를 지원합니다.
  - `none`: safety 점검 없이 실행 (기본값)
  - `mock`: 실제 모델 없이 hazard 여부를 강제로 지정할 수 있는 테스트용 monitor (`--simulate-hazard`)
  - `onnx`: ONNX Runtime YOLO 기반으로 실제 이미지에서 person 등 hazard를 탐지
- hazard가 감지되면 로봇 동작이 즉시 차단되고 `task_status=blocked_by_safety`로 종료됩니다.
- 별도 데모(`run_task_goal_real2sim_panda_interrupt_demo.py`)에서는 이동 도중(mid-motion)에도 주기적으로 안전 점검을 수행해서 중간에 정지시키는 구조(Layer 2)까지 검증했습니다.
- **Safety Pause/Resume** (`--safety-mode pause-resume`): 위 hard-block 경로와 별개로, hand 침입이 감지되면 로봇 action 적용만 멈추고(policy 자체를 호출하지 않음) episode를 종료하지 않습니다. 손이 사라져도 `--safety-resume-stable-steps`번 연속 clear를 확인한 뒤에야 같은 episode/policy phase를 이어서 진행합니다. `--hand-safety-source mock`(시간 기반 mock-timed, v0)과 `--hand-safety-source external-camera`(MediaPipe HandLandmarker로 실제 외부 카메라 frame에서 hand/arm 침입을 검출, ArUco workspace polygon 안에 들어올 때만 pause, v1) 모두 `safety/safety_supervisor.py`의 동일한 `SafetySupervisor` 상태 머신을 공유합니다. 이 상태 머신은 VLA policy 바깥에 있습니다: policy는 action을 제안할 뿐이고, 실제 적용 여부는 SafetySupervisor가 결정합니다.

## 10. 현재 하지 않은 것

이 프로젝트는 아직 다음을 포함하지 않습니다. 과장하지 않기 위해 명확히 남겨둡니다.

- 실제 OpenVLA 모델은 아직 연결하지 않았습니다. `DummyOpenVLAPolicy`는 이름과 달리 모델이 아니라 rule-based scripted policy이고, `openvla_server_dummy/`의 두 서버(`dummy_server.py`, `real_vla_compatible_server.py`) 모두 이 `DummyOpenVLAPolicy`를 그대로 호스팅할 뿐 실제 VLA 모델이 아닙니다. `--policy-backend real-vla`는 실제 서버가 들어올 adapter 구조(스키마/전처리/후처리/fallback)만 준비했을 뿐, GPU 추론이나 대형 모델 로딩은 의도적으로 하지 않았습니다.
- OpenVLA fine-tuning은 아직 수행하지 않았습니다.
- LeRobot 공식 parquet/video 포맷 변환은 아직 수행하지 않았습니다. 현재 dataset exporter는 JSONL 기반의 "LeRobot 스타일" 로컬 포맷입니다.
- LLM Agent 기반 multi-step planning은 아직 없습니다. `RuleBasedTaskGoalParser`는 rule-based 단일 명령 파서입니다.
- ROS2 실제 제어는 아직 없습니다. `robot_core/ros2_robot_backend.py`/`vision/camera_backend.py`의 `ROS2CameraBackend`는 인터페이스 skeleton일 뿐, 모든 메서드가 `NotImplementedError`를 던집니다.
- Isaac Sim, TensorRT는 이번 runner에는 포함되지 않습니다. (Isaac Sim 검토 배경은 [docs/00_project_goal.md](docs/00_project_goal.md), [docs/03_architecture.md](docs/03_architecture.md) 참고)
- 실제 로봇 하드웨어 연동은 다루지 않습니다. `robot_core/real_robot_backend.py`의 `RealRobotBackend`도 인터페이스 skeleton일 뿐이고, 모든 실행은 PyBullet 시뮬레이션 안에서만 이루어집니다.

## 11. 다음 확장 계획

- `RealRobotBackend`/`ROS2RobotBackend` skeleton을 실제 하드웨어 제어 코드로 채우기
- 실제 OpenVLA 모델로 `real_vla_compatible_server.py`의 내부를 교체 (지금의 `RealVLAPolicyClient`/`configs/real_vla_backend_config.json` 계약은 이미 준비됨)
- LeRobot 공식 parquet/video 포맷 변환 지원
- LLM Agent 기반 multi-step task planning
- wrist camera 기반 hand safety + 외부 카메라와의 fusion
- 고사양 GPU 또는 클라우드 환경에서 Isaac Sim backend 검증

## 문서

- [docs/architecture.md](docs/architecture.md) — 현재 아키텍처와 구성 요소별 역할
- [docs/hardware_portability.md](docs/hardware_portability.md) — 하드웨어 이식 시 무엇을 교체해야 하는지 정리한 manifest
- [docs/colab_vla_server_spike.md](docs/colab_vla_server_spike.md) — Google Colab GPU에 임시 VLA 서버를 띄우고 로컬과 연결하는 spike
- [docs/demo_commands.md](docs/demo_commands.md) — 실행 명령 모음
- [docs/dataset_pipeline.md](docs/dataset_pipeline.md) — dataset 기록/변환/검증 흐름
- [docs/](docs/) — 개발 단계별 기록(00~07)과 그 이후 진행 상황
