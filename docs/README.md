# Physical AI Recycling Cell

스마트 재활용 분류기를 단순 객체 인식 프로젝트가 아니라, 로봇 조작 파이프라인으로 확장하는 시뮬레이션 기반 Physical AI 프로젝트입니다.

## 현재 목표

실제 OpenVLA, YOLO, ROS 2, Isaac Sim을 붙이기 전에, 로컬에서 동작 가능한 경량 로봇 시뮬레이션 파이프라인을 먼저 구축합니다.

현재 검증된 흐름은 다음과 같습니다.

```text
한국어 사용자 명령
→ RuleBasedTaskParser
→ TaskGoal JSON
→ Dummy OpenVLA Server
→ 7-DoF Action
→ ActionAdapter
→ SimulatorBackend
→ PyBulletBackend
→ Pick-and-Place 성공
```

## 왜 PyBullet부터 사용하는가?

초기에는 Isaac Sim을 메인 시뮬레이터로 고려했지만, 현재 로컬 환경에서 Compatibility Check가 실패했습니다.

```text
GPU: RTX 3050 8GB
RAM: 16GB
환경: WSL2
결과: Isaac Sim Compatibility Check FAILED
```

주요 원인은 다음과 같습니다.

```text
VRAM 요구사항 부족
RAM 요구사항 부족
WSL/Vulkan GPU device 생성 실패
```

따라서 이 프로젝트는 특정 시뮬레이터에 강하게 의존하지 않도록 `SimulatorBackend` 추상화 구조를 사용합니다.

```text
SimulatorBackend
├── DummyRobotBackend
├── PyBulletBackend
└── IsaacSimBackend, future
```

이를 통해 로컬에서는 PyBullet 기반으로 개발을 진행하고, 추후 고사양 GPU 또는 클라우드 환경에서 Isaac Sim backend로 확장할 수 있도록 설계했습니다.

## 현재 구현 기능

```text
- 한국어 명령 파싱
- TaskGoal JSON 생성
- Dummy OpenVLA FastAPI 서버
- 7-DoF action 형식
- ActionAdapter
- SimulatorBackend 인터페이스
- DummyRobotBackend
- PyBulletBackend
- 거리 기반 grasp/place 판정
- Pick-and-place 성공 상태 추적
- JSONL 로그 저장
```

## 실행 방법

### 1. Dummy OpenVLA Server 실행

```bash
source .venv/bin/activate
uvicorn openvla_server_dummy.dummy_server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. PyBullet Backend Pipeline 실행

```bash
source .venv/bin/activate
python -m benchmark.run_pybullet_backend_pipeline
```

### 3. Pick-and-Place Demo 실행

```bash
source .venv/bin/activate
python -m benchmark.run_pybullet_pick_place_demo
```

## 현재 결과

PyBullet 기반 pick-and-place demo에서 다음 결과를 확인했습니다.

```text
task_status = success
last_event = object_placed_in_bin
PASS
```

## 앞으로의 개발 계획

```text
1. RuleBasedPickPlaceExecutor 구현
2. Multi-step control loop 구현
3. PyBullet camera rendering 추가
4. YOLO perception baseline 연결
5. ONNX Model Evaluator 연동
6. TensorRT benchmark 수행
7. ROS 2 node/topic 구조로 리팩토링
8. 실제 OpenVLA server 연동
9. 고사양 GPU 또는 클라우드 환경에서 Isaac Sim backend 검증
```
