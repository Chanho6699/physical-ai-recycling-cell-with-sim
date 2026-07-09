# 03. Architecture

## 현재 파이프라인

현재 프로젝트의 기본 흐름은 다음과 같습니다.

```text
한국어 사용자 명령
→ RuleBasedTaskParser
→ TaskGoal JSON
→ Dummy OpenVLA FastAPI Server
→ 7-DoF Action
→ ActionAdapter
→ RobotCommand
→ SimulatorBackend
→ PyBulletBackend
→ Task State / Success Log
```

## 주요 구성 요소

### RuleBasedTaskParser

한국어 사용자 명령을 구조화된 작업 목표로 변환합니다.

예시:

```json
{
  "task": "pick_and_place",
  "target_object": "plastic_cup",
  "target_bin": "plastic_bin",
  "vla_instruction": "Pick the plastic cup and place it in the plastic recycling bin.",
  "success_condition": "The plastic cup is inside the plastic recycling bin."
}
```

현재는 실제 LLM을 사용하지 않고, rule-based 방식으로 명령을 파싱합니다. 추후 Claude 또는 다른 LLM Agent로 교체할 수 있도록 구조를 단순하게 유지합니다.

### Dummy OpenVLA Server

FastAPI 기반의 가짜 OpenVLA 추론 서버입니다.

실제 OpenVLA 모델을 바로 연결하기 전에, 외부 VLA 서버와 통신하는 구조를 먼저 검증하기 위해 사용합니다.

현재는 다음 형식의 dummy 7-DoF action을 반환합니다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

추후 이 서버 내부를 실제 OpenVLA inference 코드로 교체할 수 있습니다.

### ActionAdapter

OpenVLA 또는 Dummy OpenVLA Server에서 받은 7-DoF action을 시뮬레이터에서 사용할 수 있는 `RobotCommand`로 변환합니다.

예시:

```text
7-DoF Action
→ 위치 이동량 변환
→ 회전 이동량 변환
→ gripper open/close 변환
→ RobotCommand 생성
```

### SimulatorBackend

여러 시뮬레이터를 공통 방식으로 다루기 위한 추상화 계층입니다.

현재 및 계획된 backend는 다음과 같습니다.

```text
SimulatorBackend
├── DummyRobotBackend
├── PyBulletBackend
└── IsaacSimBackend, future
```

이 구조를 사용하면 PyBullet, Isaac Sim, Dummy Robot을 같은 파이프라인에서 교체 가능한 방식으로 사용할 수 있습니다.

### PyBulletBackend

로컬에서 실행 가능한 경량 물리 시뮬레이션 backend입니다.

현재 구현된 요소는 다음과 같습니다.

```text
- plane
- table
- recyclable object
- recycling bin
- end-effector sphere
- gripper open/close state
- 거리 기반 grasp/place logic
```

현재는 실제 로봇팔 URDF를 사용하지 않고, end-effector sphere를 직접 이동시키는 단순화된 방식으로 pick-and-place 흐름을 검증합니다.

## 설계 결정

초기에는 Isaac Sim을 메인 시뮬레이터로 고려했습니다. 하지만 로컬 RTX 3050 8GB / RAM 16GB / WSL2 환경에서 Isaac Sim Compatibility Check가 실패했습니다.

주요 실패 원인은 다음과 같습니다.

```text
VRAM 부족
RAM 부족
WSL/Vulkan GPU device 생성 실패
```

따라서 현재 개발 단계에서는 PyBullet을 사용하고, Isaac Sim은 추후 고사양 GPU 또는 클라우드 환경에서 검증할 future backend로 남겨둡니다.

이 결정의 목적은 다음과 같습니다.

```text
1. 로컬 환경에서 안정적으로 개발 진행
2. 시뮬레이터 의존성 최소화
3. 추후 Isaac Sim으로 확장 가능한 구조 유지
4. 로봇 제어 파이프라인 자체를 먼저 검증
```
