# 04. PyBullet Pick-and-Place Result

## 목표

현재 로봇 명령 파이프라인이 PyBullet 기반 경량 시뮬레이션 환경에서 기본적인 pick-and-place 작업을 수행할 수 있는지 검증합니다.

## 테스트 명령어

```bash
python -m benchmark.run_pybullet_pick_place_demo
```

## 테스트 시나리오

```text
1. PyBullet world 초기화
2. end-effector를 재활용 물체 근처로 이동
3. gripper close
4. grasp event 감지
5. end-effector를 수거함 위치로 이동
6. gripper open
7. place event 감지
8. task_status를 success로 변경
```

## 결과 요약

PyBullet 기반 pick-and-place demo가 성공적으로 완료되었습니다.

중요한 상태 변화는 다음과 같습니다.

```text
Step 2:
task_status = grasped
last_event = object_grasped
held_object = true

Step 4:
task_status = success
last_event = object_placed_in_bin
held_object = false
```

최종 결과:

```text
Demo finished: task_status=success
PASS
```

## 의미

이 결과는 현재 파이프라인이 고수준 작업 명령을 로봇 명령으로 변환하고, 이를 경량 물리 시뮬레이션 backend에 적용하여 pick-and-place 작업 성공 여부까지 판단할 수 있음을 보여줍니다.

현재 검증된 흐름은 다음과 같습니다.

```text
TaskGoal
→ RobotCommand sequence
→ PyBulletBackend
→ object grasp
→ object place
→ task success
```

## 현재 한계

현재 구현은 실제 로봇팔 URDF, IK, 실제 gripper 물리 제어를 포함하지 않습니다.

현재는 다음과 같은 단순화된 방식입니다.

```text
- end-effector sphere를 직접 이동
- object와 end-effector 사이 거리로 grasp 판정
- object와 bin 사이 거리로 place 판정
- gripper 상태는 open/close 문자열로 관리
```

## 다음 단계

다음 개발 단계에서는 사람이 직접 RobotCommand sequence를 작성하는 방식에서 벗어나, `TaskGoal`과 현재 simulator state를 기반으로 자동으로 명령 sequence를 생성하는 `RuleBasedPickPlaceExecutor`를 구현합니다.

목표 흐름:

```text
TaskGoal
+ Current Simulator State
→ RuleBasedPickPlaceExecutor
→ approach
→ grasp
→ carry
→ place
→ success/fail 판정
```
