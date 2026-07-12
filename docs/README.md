# docs/

프로젝트 최신 상태와 대표 실행 명령은 루트 [README.md](../README.md)를 참고하세요. 이 폴더는 현재 상태를 요약한 참조 문서와, 프로젝트가 진행되어 온 개발 단계별 기록을 함께 보관합니다.

## 현재 상태 참조 문서

- [architecture.md](architecture.md) — 현재 전체 파이프라인과 구성 요소별 역할 (TaskGoal → 인식 → Real2Sim → policy → SafetyGate → PyBullet Panda → 기록)
- [demo_commands.md](demo_commands.md) — 실행 명령 모음
- [dataset_pipeline.md](dataset_pipeline.md) — raw episode → dataset export → replay 검증 → OpenVLA 스타일 action 변환 흐름

## 개발 단계별 기록 (00~07)

프로젝트 초기 단계의 설계 배경과 각 단계에서의 검증 기록입니다. 이 시점 이후(TaskGoal, Real2Sim, SafetyGate, TrajectoryRecorder, dataset export/replay, DummyOpenVLAPolicy, full demo runner)의 내용은 위 참조 문서에서 다룹니다.

- [00_project_goal.md](00_project_goal.md) — 프로젝트 목표
- [01_environment_check.md](01_environment_check.md) — 로컬 환경 점검
- [02_isaacsim_minimal_scene.md](02_isaacsim_minimal_scene.md) — Isaac Sim 최소 씬 시도
- [03_architecture.md](03_architecture.md) — 초기 아키텍처 (dummy sphere backend + FastAPI dummy OpenVLA server 중심 구조)
- [04_pybullet_pick_place_result.md](04_pybullet_pick_place_result.md) — PyBullet 기반 pick-and-place 최초 검증
- [05_yolo_safety_monitor.md](05_yolo_safety_monitor.md) — YOLO Safety Monitor 도입
- [06_image_based_safety_test.md](06_image_based_safety_test.md) — 이미지 기반 safety 검증
- [07_pybullet_panda_backend.md](07_pybullet_panda_backend.md) — sphere backend에서 실제 Franka Panda URDF backend로 전환

이 폴더의 초기 문서에서 언급된 Isaac Sim compatibility 이슈(RTX 3050 8GB / RAM 16GB / WSL2 환경에서 Compatibility Check 실패)는 여전히 유효하며, Isaac Sim backend는 아직 future 항목으로 남아 있습니다.
