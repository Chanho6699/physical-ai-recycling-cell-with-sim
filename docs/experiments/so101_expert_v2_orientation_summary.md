# SO-101 Expert V2 (Orientation-aware) 사전 검증 요약 -- Stage 1C Go/No-Go

## 1. 목적과 결론 먼저

Stage 1C(box yaw 학습) 시작 전, rectangular box의 yaw에 대응하는
Orientation-aware Expert V2를 구현하고 Expert 단독으로 yaw-grid를
평가했다. **결론: 이번 구조로는 Stage 1C dataset을 만들 수 없다.**
전체 valid-scene 성공률 **6.1%**(11/180)로 통과 기준(95%)에 크게
못 미친다. 원인은 구현 버그가 아니라 **SO-101 5-DOF 팔 자체의
기구학적 한계**다 -- yaw=0/180도(기존 V1과 동일한 orientation)에서만
성공하고, 그 외 모든 yaw(±15~90도)에서 100% 실패한다.

## 2. Expert V1 구조 조사 (수정 없이 조사만)

- V1 entry point: `benchmark/so101_scripted_expert.py::run_pick_and_place_episode()`.
- IK 호출은 정확히 한 곳: `robot_sim/so101_pybullet_backend.py::compute_joint_target_from_ee_delta()` -- `p.calculateInverseKinematics(robot_id, ee_link_index, target_position, ...)`. **`targetOrientation` 인자를 아예 넘기지 않는다** -- position-only IK. 함수 자신의 docstring이 "delta_orientation is accepted for interface symmetry with a future full-6DOF version, but is currently ignored"라고 명시.
- 즉 V1이 보이는 "고정된 orientation"은 하드코딩된 값이 아니라, 매 에피소드 동일한 `NEUTRAL_ARM_POSITIONS=[0,0,0,0,0]`에서 시작해 position-only IK가 수렴하는 **부수 효과**다.
- `ee_link_index=5`. Neutral pose EE quaternion(xyzw) = `[0.0172, 0.7070, 0.0172, 0.7068]`.
- `object_yaw_rad`(생성자 인자)와 `get_object_pose()`(orientation 포함 반환)는 **이미 존재** -- 이번 작업에서 backend 수정이 전혀 필요 없었다.

## 3. Gripper closing axis와 EE link frame (실측, 추측 아님)

- **Jacobian 실측**: neutral pose에서 각 관절의 world-frame 회전축(angular Jacobian column)을 직접 질의:
  - `shoulder_pan` -> world **Z** 축 (`[0, 0, -1]`)
  - `shoulder_lift`/`elbow_flex`/`wrist_flex` -> 공통 축 (`[0, 1, 0]` 근방, 팔의 수직 평면 내 관절들)
  - `wrist_roll` -> world **X** 축 (`[-1, 0, 0]`)
- **EE link 자신의 local axis를 world로 매핑** (neutral pose): local X -> world `(0, 0.05, -1)`(아래쪽), local Y -> world `(0, 1, 0.05)`, **local Z -> world `(1, 0, 0)`**.
- 따라서 Stage 1B에서 확인된 "gripper closing axis = world X(yaw=0)"는 **EE link의 local Z축**에 해당한다.
- **결정적 사실**: world-frame yaw(수평면 내 회전)를 만들 수 있는 관절은 `shoulder_pan` 하나뿐이며, 이 관절은 동시에 base로부터 target까지의 수평 방향(사실상 XY 위치)을 결정하는 관절이다. `wrist_roll`은 world X축 회전(그리퍼를 그 축을 중심으로 spin -- closing axis 자체는 안 바뀜)만 만든다.

## 4. Quaternion 합성 방식 (검증됨, 두 가지 방법으로 교차 확인)

- 공식: `R_target = multiplyTransforms([0,0,0], Rz_world(yaw), [0,0,0], R_base)` (world-frame yaw를 R_base 위에 추가 합성).
- 이 합성 순서는 (a) 기존 코드 `object_offset_in_ee_frame()`이 이미 사용하는 `p.multiplyTransforms(parent, child)` 관례와 일치하고, (b) **closing axis(EE local Z) 벡터를 실제로 회전시켜 봤을 때** 정확히 box의 회전된 local-X 방향 `(cos(yaw), sin(yaw), 0)`과 일치함을 직접 확인했다 (yaw=15/45/90도 전부 소수점 3자리까지 일치).
- 반대 합성 순서(`R_base * Rz_local(yaw)`, EE 자신의 local frame에서 yaw 적용)는 IK가 쉽게 수렴하지만(위치 오차 4mm 이하), **closing axis가 전혀 회전하지 않는다** (yaw 몇도든 항상 world X 그대로) -- 이는 물리적으로 아무 의미 없는 wrist_roll spin일 뿐, box yaw를 전혀 따라가지 않는 **잘못된 공식**이다. 두 방법 모두 실측으로 검증했으며 추측으로 결정하지 않았다.

## 5. IK가 실제로 실패하는 방식 (반복 확인)

단일 IK 호출(step 0)뿐 아니라, 같은 target을 여러 iteration 반복 적용해도(step 8회) 위치 오차가 **0으로 수렴하지 않고 고정된 값에 머문다**:

| yaw | 위치 오차(고정값) | orientation 오차 |
|---|---|---|
| 2도 | 0.008m | 0.17도 |
| 5도 | 0.025m | 0.50도 |
| 10도 | 0.052m | 2.26도 |
| 15도 | 0.079m | 2.05도 |
| 30도 | 0.159m | 2.88도 |

오차가 yaw에 거의 선형(~0.0053m/도, yaw>=10도)으로 증가 -- 수치 노이즈가 아니라 **shoulder_pan의 권한을 position과 orientation이 나눠 쓰는 구조적 trade-off**. 목표 위치 자체를 residual만큼 보정해 재입력하는 outer-loop 보정도 시도했으나, 15도에서는 느리게 진동하며 완전히 수렴하지 않고, 30도 이상에서는 발산한다(오차가 iteration마다 커짐).

## 6. Expert V2 구조 (신규, V1 완전 별도)

파일: `benchmark/so101_expert_v2_orientation.py` (신규). V1/backend는 **한 줄도 수정하지 않았다** -- `p.calculateInverseKinematics()`를 V2 자신의 코드에서 직접 호출(backend의 public 속성 `robot_id`/`ee_link_index`/`arm_joint_indices`/`joint_info_by_name`/`min_ee_height_m`/`client_id`만 재사용)하기 때문에 backend에 훅을 추가할 필요조차 없었다.

- `ObjectGraspMetadata`: position/yaw/half-extent/closing_axis 메타데이터.
- `OrientationAwareGraspPlanner`: 위 공식으로 `target_end_effector_quaternion` 계산, 180도 대칭 후보 중 회전량이 작은 쪽 선택(`canonicalize_grasp_yaw`).
- `compute_joint_target_with_orientation`: V1의 `compute_joint_target_from_ee_delta()`와 동일한 로직(min height floor, 동일 IK 파라미터, 동일 joint-limit clip) + `targetOrientation` 추가.
- `move_to_target_with_orientation`: V1 `move_to_target()`의 orientation-aware 버전(같은 step-loop, 같은 MAX_STEP_M/CONVERGENCE_TOLERANCE_M/STEP_ERROR_FAILURE_THRESHOLD_M 상수 재사용) -- 새 함수로 분리, `move_to_target()` 자체는 무수정.
- `rotate_to_neutral_orientation`: pre_grasp~lift 이후, lateral 이동 없이 wrist를 V1의 neutral orientation으로 되돌리는 신규 transition phase.
- 이후(transport~release~settle)는 V1의 `move_to_target()`/`run_bin_place_segment()`를 **그대로 재사용**(orientation이 이미 V1의 neutral로 복귀했으므로 orientation-aware 버전이 필요 없음).
- `ExpertExecutionMonitor`: max_joint_jump/joint_limit_violation/orientation_error/collision을 매 step 기록.

## 7. Orientation-aware IK 적용 phase

pre_grasp -> approach -> grasp -> lift: object-yaw 정렬 orientation 유지.
lift 직후 신규 `transition_to_neutral` phase(위치 고정, orientation만 최대 30 step에 걸쳐 복귀) -> transport부터는 V1 그대로.

## 8. Yaw 대칭 / joint limit / discontinuity

- 180도 대칭 처리: `canonicalize_grasp_yaw()`가 `object_yaw`와 `object_yaw+pi` 중 절대값이 작은 쪽(중립 자세로부터 회전량이 작은 쪽)을 선택 -- yaw=180도는 target_gripper_yaw=0으로 정확히 환원되어 V1과 동일하게 성공했다(11개 성공 episode 중 일부).
- Joint limit violation: **전체 180 episode 중 0건**.
- NaN/Inf: 0건.
- max_joint_jump 전체 최댓값: 0.160 rad -- excessive-jump로 별도 abort하는 로직은 넣지 않았고(V1도 그런 hard abort가 없음) monitoring 값으로만 기록.
- wrist_roll 자체의 discontinuity(±180도 wrap)는 관측되지 않음 -- 애초에 실패가 position-error 임계값에서 먼저 발생하므로 wrist_roll이 그 지점까지 가지 않음.

## 9. V1 regression 결과

`benchmark/test_so101_expert_v1_regression.py` (신규): 10/10 통과 (cube/box yaw=0 결정론성, 기존 Stage 1B 데이터셋 `episode_index=0` 재현 포함). **V2 구현 및 yaw-grid 실행 이후 재실행해도 동일하게 10/10 통과** -- V1은 전혀 영향받지 않았다.

## 10. Yaw-grid 평가 결과

`benchmark/evaluate_so101_expert_v2_yaw_grid.py` (신규) 실행 결과, 총 180회 시도 (yaw 12개 값 x position group 5개 x seed 3개):

- attempted=180, scene_valid=180, discarded=0
- **success=11, failed=169 -> 전체 성공률 6.1%**

### Yaw별 성공률
| yaw(도) | 성공률 |
|---|---|
| 0 | 11/15 = 73.3% |
| ±15, ±30, ±45, ±60, 90 | 0/15 = 0.0% (전부) |
| ±75 | 0/15 = 0.0% |

yaw=0을 제외한 **모든 yaw 구간에서 완전 실패(0%)** -- 실패가 특정 구간에 국한되지 않고 전 yaw 범위에 걸쳐 있다.

### 위치 그룹별 성공률 (전부 yaw=0에서만 나온 성공)
| group | 성공률 |
|---|---|
| center | 3/36 = 8.3% |
| interior | 2/36 = 5.6% |
| edge | 3/36 = 8.3% |
| corner | 0/36 = 0.0% |
| x_min_corridor | 3/36 = 8.3% |

### 실패 원인별 개수
| failure_reason | 개수 |
|---|---|
| orientation_unreachable | 140 |
| ik_failed | 25 |
| place_waypoint_failed | 2 |
| grasp_failed | 2 |

`orientation_unreachable`(신규 분류, 이번 task에서 정의)은 "orientation은 목표에 근접(<15도)했지만 position이 임계값을 초과"한 경우 -- 5절에서 실측한 shoulder_pan 결합 현상과 정확히 일치하는 실패다. `ik_failed`는 orientation 자체도 크게 벗어난, 더 발산적인 경우(대체로 ±75/90도). `place_waypoint_failed`/`grasp_failed` 4건은 전부 `corner`(corner_pn) 위치의 **yaw=0**에서 발생 -- 이는 Stage 1B held-out test에서 이미 확인된 corner_pn의 기존 약점(40% 성공률)과 일치하는 V1 자신의 한계이며, V2가 새로 만든 회귀가 아니다.

## 11. 통과 기준 대비 판정

- yaw=0에서 V1과 동등한 수준: **충족** (V2의 yaw=0 경로는 target_gripper_yaw=0으로 정확히 환원되어 V1과 수학적으로 동일한 target orientation을 사용; corner 성능 저하도 V1 자신의 기존 패턴과 일치).
- 전체 valid-scene 성공률 95% 이상: **미충족** (6.1%).
- 특정 yaw 구간 전면 실패 없음: **미충족** (yaw≠0/180 전 구간 0%).
- center/interior 매우 높은 성공률: **미충족** (yaw 포함 전체로는 각 8.3%/5.6% -- yaw=0만 보면 V1 수준으로 정상).
- joint limit violation 0, NaN/Inf 0: **충족**.
- 실패/discard 전부 기록: **충족** (`results/so101_expert_v2_yaw_grid/yaw_grid_records.jsonl`, 180 rows 전부).
- V1 regression 통과: **충족**.

**종합: Section 10 기준 미충족 -> Stage 1C dataset 생성 불가.**

## 12. 남은 한계와 근본 원인

이번 실패는 orientation 공식이 틀려서가 아니라 (4절/5절에서 두 가지 독립적 방법으로 공식 자체는 올바름을 확인했다), **SO-101 팔이 5개의 revolute 관절만 가지고 있고 그중 world-Z축 회전을 만드는 관절이 `shoulder_pan` 하나뿐이며, 그 관절이 이미 목표 위치의 수평 방향을 결정하는 데 전량 사용되고 있기 때문**이다. 즉 "물체 위치를 유지하면서 그리퍼를 수평으로 yaw"하는 동작은, 이 팔의 현재 모델(5-DOF, 이 URDF의 관절 배치)로는 **일반적으로 불가능**하다 -- 이는 구현 버그가 아니라 로봇 기구학 자체의 한계로, 이번 작업 전에는 문서화되지 않았던 사실이다.

## 13. 다음 단계 제안 (구현 아님, 제안만)

1. **접근 방향(approach azimuth) 자체를 바꾸는 대체 IK 해**를 찾는 방법 -- 같은 (x,y,z)를 다른 `shoulder_pan` + 팔꿈치 결합("elbow-up/down" 류)으로도 도달 가능한지 nullspace IK(`p.calculateInverseKinematics`의 `lowerLimits/upperLimits/jointRanges/restPoses` 인자 활용) 또는 여러 초기 seed에서 IK를 재시도하는 방식을 조사할 가치가 있다. 다만 이 팔의 경우 상위 sub-chain(shoulder_lift/elbow_flex/wrist_flex)이 전부 같은 평면 내 회전이라, 다른 azimuth로 도달하려면 근본적으로 다른 shoulder_pan 값이 필요해 near-solution이 존재하지 않을 가능성이 높다 -- 실제로 존재하는지는 미검증.
2. **작은 yaw 범위(±5도 이내)로 제한**하는 방안 -- 5도에서도 이미 2.5cm 오차가 나서 grasp 정확도 요구(수 mm)를 만족하기 어려워 보이지만, box의 grasp width(4cm)를 고려하면 약간의 tolerance 여유가 있을 수도 있다 -- 정량적으로 별도 검증 필요.
3. **물체를 approach 전에 회전시키는 방식**(EE가 아니라 object 자체를 pre-grasp 단계에서 원하는 yaw로 미리 정렬하는 fixture/push 등)은 이번 범위(recovery/push 등 제외) 밖이라 이번에 조사하지 않았다.
4. 가장 근본적인 대안은 **object yaw randomization 없이 진행하고, cylinder/can처럼 회전 대칭인 형태로 다음 shape 확장을 진행**하는 것 -- Stage 1B 문서(9번 항목)에서 이미 이 경로를 경고했지만("회전 대칭 물체는 문제를 우연히 회피"), 현재로선 이 팔의 구조적 한계 자체가 명확해졌으므로, yaw 대응은 **하드웨어/기구학 재검토 없이는 해결되지 않는 문제**로 결론짓는다.

## 14. Git staging 추천 파일 / 제외 대상

- 추천(신규 파일만, 모두 이번 task에서 생성): `benchmark/so101_expert_v2_orientation.py`, `benchmark/evaluate_so101_expert_v2_yaw_grid.py`, `benchmark/test_so101_expert_v1_regression.py`, `docs/experiments/so101_expert_v2_orientation_summary.md`.
- 제외: `results/so101_expert_v2_yaw_grid/`(gitignored 대상 결과 디렉토리, 기존 프로젝트 관례와 동일하게 datasets/outputs/results는 git 미포함), 스크래치패드의 임시 sanity-check 스크립트(프로젝트 디렉토리 밖).
- **이번 작업에서 git add/commit/push는 수행하지 않았다** (사용자 지시대로).

## 15. 공용/Laptop 트랙과의 충돌 여부

Laptop/ROS2/YOLO/Safety/Dashboard 관련 파일은 조회도 수정도 하지 않았다. `robot_sim/so101_pybullet_backend.py`도 전혀 수정하지 않았으므로 (V2가 필요한 모든 것을 이미 public API로 제공했다) 다른 트랙과의 충돌 가능성은 없다.

## 16. 실행 명령

```
.venv-vla/bin/python -m benchmark.test_so101_expert_v1_regression
.venv-vla/bin/python -m benchmark.evaluate_so101_expert_v2_yaw_grid
```

결과 파일: `results/so101_expert_v2_yaw_grid/yaw_grid_records.jsonl`(180 rows, 전체 상세), `results/so101_expert_v2_yaw_grid/yaw_grid_summary.json`(집계). `results/`는 Git에 포함되지 않는다.
