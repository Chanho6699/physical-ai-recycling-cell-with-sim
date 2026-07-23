# SO-101 Stage 1C: Cylinder/Can Feasibility 사전 검증 요약

## 1. 목적과 결론 먼저

Stage 1B 결과(cube retention 82.0%, box validation 76.0%, box held-out 66.7%)와 Expert V1을 그대로 보존한 채, 물체 크기 기반 Scripted Expert V2.1을 구현하고 cylinder/can 형태의 Expert 타당성 및 Stage 1B checkpoint zero-shot 성능을 사전 검증했다. **결론: Stage 1C cylinder dataset 수집을 진행해도 좋다.** Expert V2.1은 선택된 cylinder(반지름 0.02m, 지름 4cm, 높이 4cm) 후보에서 125/125 = **100%** (legacy_success와 physical_success 모두)를 기록했다. Stage 1B 공식 checkpoint의 zero-shot 성능은 46/75 = 61.3%였으나, 이는 Expert/scene 문제와 섞이지 않은 순수한 정책 일반화 격차로 확인되었다.

## 2. V1 고정 파라미터 조사

- Pre-grasp 높이: object CENTER + `PRE_GRASP_OFFSET_M[2]=0.08` (고정, object 크기 무관).
- Grasp(approach) 높이: object CENTER + `APPROACH_OFFSET_M[2]=0.03`. V1은 별도의 "grasp 전용 이동"이 없다 -- approach 위치에서 바로 gripper를 닫는다.
- Lift 높이: `LIFT_DISTANCE_M=0.08`, object 크기와 무관하게 **현재 EE 위치 기준 상대값**이라 애초에 크기 일반화가 필요 없다.
- Bin 접근/release 높이: `run_bin_place_segment()`가 bin의 `rim_z`(실측, `get_bin_debug_info()`) 기준 상대 clearance(`BIN_PRE_PLACE_CLEARANCE_M=0.08`, `BIN_RELEASE_CLEARANCE_M=0.03`, `BIN_RETREAT_CLEARANCE_M=0.10`)로 계산 -- object 크기 입력이 전혀 없다.
- Gripper open/close target: `set_gripper(1.0)`/`set_gripper(0.0)`, **normalized [0,1] 단위** (0=완전 닫힘, 1=완전 열림), 내부적으로 URDF 관절 각도(`gripper_lower=-0.174533rad ~ gripper_upper=1.74533rad`)로 변환.
- Grasp 성공 거리 threshold: `GRASP_DISTANCE_THRESHOLD_M=0.04m` (EE-object 중심 거리), `GRASP_GRIPPER_CLOSED_THRESHOLD=0.15`(정규화 gripper 각도) -- **둘 다 shape/크기 무관**.
- object 높이/XY footprint 사용처: `robot_sim/so101_pybullet_backend.py`의 `_object_half_extents()`(collision/visual box 생성), `_default_object_position()`(높이만 사용, footprint 무관), `min_ee_height_m` 계산(footprint 무관).
- cube와 rectangular box는 **동일 상수를 그대로 공유**한다 (`object_height=0.04` 동일, `object_footprint_xy`만 다름) -- 이미 Stage 1B에서 검증된 사실.

## 3. SO-101 gripper opening과 cylinder 크기 제약

- `gripper` 관절은 **revolute**(회전 jaw)이지 linear/parallel-jaw가 아니다 (`third_party/so101_arm/so101_new_calib.urdf`의 `<joint name="gripper" type="revolute">`, axis=[0,0,1], limit -0.174533~1.74533rad).
- **결정적 실측 결과**: 이 시뮬레이션의 grasp 판정은 **실제 접촉(contact-force) 기반이 아니라 거리+gripper 각도 threshold 기반의 고정 constraint**다. radius 0.015m~0.05m(지름 3cm~10cm) 전 범위에서 `contact_count=0`이 측정되었고, **이미 검증된 기존 cube/box(지름 4cm) baseline에서도 동일하게 `contact_count=0`**이었다. 즉 이 프로젝트의 grasp 메커니즘은 애초에 물리적 파지력을 시뮬레이션하지 않으며, 이는 cylinder 도입 이전부터 있던 사실이다.
- 이로 인해 "legacy 판정 기준(거리+gripper 각도)"만으로는 후보 크기의 물리적 타당성을 구분할 수 없었다 (지름 10cm까지도 legacy 기준으로는 100% "성공"). 그래서 후보 선정은 (a) 실측 불가능한 접촉 기반 한계 대신 **이미 수백~수천 episode로 검증된 기존 cube/box 스케일(4cm)을 그대로 재사용**하는 방향으로, (b) scene validity/bin 호환성 실측으로 보강했다.
- gripper 최대 opening에 대한 명시적 수치 스펙은 `third_party/so101_arm/README.md`/`SOURCE.md`에 없다 (추측 금지 원칙에 따라 "모른다"고 명시).

## 4. 검토한 cylinder 후보 크기

`configs/so101_stage1c_cylinder_candidates.json`에 전체 기록. 3개 후보를 5개 위치 그룹(center/interior/edge/corner/x_min_corridor)에서 V1의 **완전 무수정** 파이프라인으로 스크리닝(cylinder shape는 이번 task의 backend additive 변경으로 생성):

| 후보 | radius | diameter | height | aspect ratio | scene validity(5개 그룹) | legacy 성공 |
|---|---|---|---|---|---|---|
| small | 0.015m | 3.0cm | 4cm | 0.75 | 전부 valid | 전부 성공 |
| **medium_can** | **0.02m** | **4.0cm** | **4cm** | **1.0** | 전부 valid | 전부 성공 |
| near_limit | 0.028m | 5.6cm | 4cm | 1.4 | 전부 valid | 전부 성공 |

(mass=0.05kg, friction=미설정/PyBullet 기본값 0.5 -- cube/box와 동일, 이번 task 범위에서 randomization 안 함)

## 5. 최종 선택한 cylinder 크기와 근거

**radius=0.02m(지름 4cm), height=0.04m 선택.** 근거: (1) 기존 cube(`object_footprint_xy=[0.02,0.02]`)와 box의 grasp-축 폭(`BOX_FOOTPRINT_XY[0]*2=0.04m`)과 **정확히 같은 스케일**이라 이번 실험에서 바뀐 변수를 "shape 하나"로 격리할 수 있다(Stage 1B가 "shape 하나만 바꾼다"고 명시한 것과 동일한 원칙). (2) 원통형 collision 형태만으로 cube/box와 시각적으로 명확히 구분되므로 크기를 키울 필요가 없다. (3) Expert 단독 평가(6절)에서 125/125=100%로 실증됨.

## 6. Backend/scene 변경 필요 여부

**필요했음 -- 최소 additive 변경 1건.** `robot_sim/so101_pybullet_backend.py`에 `object_shape`(기본값 `"box"`)와 `object_radius`(cylinder 전용) scene_config 키, `_create_cylinder()` 신규 메서드, `reset()`의 object 생성 분기(1줄 if/else)를 추가했다. **기본값(`"box"`)에서는 기존 코드 경로가 byte-for-byte 그대로 실행**되어, 기존 cube/box를 사용하는 어떤 호출자도 영향받지 않는다 (`test_so101_expert_v1_regression.py` 10/10 통과로 확인). V1(`benchmark/so101_scripted_expert.py`)은 한 줄도 수정하지 않았다.

## 7. Expert V2.1 구조

파일: `benchmark/so101_expert_v2_size_aware.py` (신규). `ObjectMetadata`(shape/position/height/footprint 또는 radius/mass/friction) → `SizeAwareGraspPlanner`(dimension 기반 grasp 계산) → `run_pick_and_place_episode_v2_1()`(V1의 `gripper_phase`/`move_to_target`/`run_bin_place_segment`을 **그대로 재사용**). Orientation은 전혀 다루지 않는다 -- `so101_expert_v2_orientation.py`(V2 orientation 실험)는 이번 task에서 **한 줄도 수정/삭제하지 않았고 import조차 하지 않았다** (절대 원칙 3 준수).

## 8. Dimension 기반 계산식

```
object_top_z = object_center_z + object_height / 2
pre_grasp_z  = object_top_z + PRE_GRASP_APPROACH_CLEARANCE_M(0.06)
grasp_z      = object_top_z + GRASP_APPROACH_CLEARANCE_M(0.01)
```

0.06/0.01 상수는 **height=0.04(기존 cube/box)일 때 V1의 기존 `PRE_GRASP_OFFSET_M[2]=0.08`/`APPROACH_OFFSET_M[2]=0.03`과 정확히 일치하도록 역산**했다 (`0.02+0.06=0.08`, `0.02+0.01=0.03`). Lift/bin-release는 V1 함수를 그대로 재사용(크기 무관, 현재 EE/bin rim_z 기준 상대값이라 일반화가 애초에 불필요).

## 9. Shape-specific 예외 (최소한만)

유일한 shape-specific 코드는 `effective_object_width_m` 계산 한 줄: box는 `footprint_xy_half_extents[0]*2`, cylinder는 `2*radius_m`. 그 외 모든 값(pre_grasp/grasp 높이, gripper target, lift, bin release)은 공통 공식/공통 함수를 그대로 사용한다 -- "if shape==cube: 전부 수동 지정" 같은 분기는 없다.

## 10. 신규/수정 파일

신규: `benchmark/so101_expert_v2_size_aware.py`, `benchmark/evaluate_so101_expert_v2_cylinder.py`, `benchmark/evaluate_so101_stage1b_cylinder_zeroshot.py`, `benchmark/test_so101_expert_v2_size_aware.py`, `configs/so101_stage1c_cylinder_candidates.json`, `docs/experiments/so101_stage1c_cylinder_feasibility_summary.md`.
수정(additive만): `robot_sim/so101_pybullet_backend.py`(`object_shape`/`object_radius`/`_create_cylinder()`), `benchmark/so101_smolvla_rollout.py`(`build_rollout_backend()`/`run_one_rollout()`에 `object_shape_override`/`object_radius_override` 신규 optional 인자, 기본값 None -- 기존 cube/box zero-shot 스크립트는 전혀 영향받지 않음).

## 11. V1 cube regression 결과

`benchmark/test_so101_expert_v1_regression.py`: **10/10 통과** (backend 변경 전/후 모두 재실행하여 확인).

## 12. V1 box regression 결과

동일 파일의 box yaw=0 서브셋 5개 항목 포함, **10/10에 포함되어 통과** (기존 Stage 1B 데이터셋 `episode_index=0` 재현 포함).

## 13. V2.1 cube 결과

`benchmark/test_so101_expert_v2_size_aware.py`: 기존 cube 치수를 V2.1에 입력했을 때 pre_grasp/approach target이 V1과 **1e-6m 이내로 일치**, place_success=True(V1과 동일). 13/13 항목 전체 통과.

## 14. V2.1 box 결과

같은 파일에서 Stage 1B box 치수(`BOX_FOOTPRINT_XY=[0.02,0.03]`) 입력 시에도 pre_grasp/approach target이 V1과 1e-6m 이내로 일치, place_success=True(V1과 동일), `effective_object_width_m=0.04`(검증됨).

## 15. Cylinder Expert 전체 시도 수

`benchmark/evaluate_so101_expert_v2_cylinder.py`: **125회** (position group 5개 × scene yaw 5개[0/45/90/135/180도, 회전 대칭성 확인용] × seed 5개). attempted=125, scene_valid=125, discarded=0.

## 16. Cylinder 위치 그룹별 성공률

| position_group | legacy_success | physical_success |
|---|---|---|
| center | 25/25 | 25/25 |
| interior | 25/25 | 25/25 |
| edge | 25/25 | 25/25 |
| corner | 25/25 | 25/25 |
| x_min_corridor | 25/25 | 25/25 |

**전체 125/125 = 100%.** Yaw별(회전 대칭성 확인)도 0/45/90/135/180도 전부 25/25=100%로 동일 -- cylinder의 시뮬레이션 orientation 값이 결과에 영향을 주지 않음을 확인했다 (단, 이는 orientation-aware grasp를 검증한 것이 아니라 회전 대칭 물체이므로 당연히 나온 결과임 -- 절대 원칙 10에 따라 box arbitrary-yaw 해결로 해석하지 않는다).

Stage 1B에서 box의 corner_pn이 40%로 취약했던 것과 달리, 이번 cylinder의 corner 그룹은 100%였다 -- box의 corner 약점은 **긴 축(6cm)의 bin/table 여유 공간 문제**였고, cylinder는 전 방향으로 대칭인 4cm 폭이라 그 문제 자체가 애초에 발생하지 않는다(별개의 형태적 특성 차이이지, corner 자체가 "해결"된 것이 아님).

## 17. legacy success와 physical success 비교

**차이 없음 (0건 mismatch).** `contact_count`는 125건 전부 0으로 측정되어 (2절 참고) 이번 시뮬레이션에서는 애초에 판별력이 없는 diagnostic 값임을 재확인했다. `physical_success`는 `legacy_success AND object_lift_height>=0.03m AND grasp_maintained_all_steps(lift+transport)`로 정의했으며, 125건 전부 legacy와 physical이 정확히 일치했다 -- "가까이 있었지만 실제로 들어 올리지 못한" 사례는 Expert 레벨에서는 0건이었다.

## 18. 실패 원인별 개수

**Expert 레벨(V2.1, 125회): 실패 0건.** scene_invalid 0, ik_failed 0, grasp_failed 0, place_waypoint_failed 0.

## 19. NaN/Inf/joint-limit/collision/discard 결과

전부 0건 (125/125). `contact_count` 자체도 0이었으나 이는 "충돌 없음"이 아니라 "이 시뮬레이션의 grasp가 애초에 접촉 기반이 아님"이라는 2절의 사실과 같은 맥락이다.

## 20. Stage 1B cylinder zero-shot 결과

`benchmark/evaluate_so101_stage1b_cylinder_zeroshot.py`, Stage 1B 공식 checkpoint(재선정 없이 1회 실행), 15위치 × 5 repeat = 75 rollouts:

- **overall: 46/75 = 61.3%**, grasp_count=47/75=62.7%
- known_interior(center+기존범위 4방향): **19/25 = 76.0%**
- expanded_edge(확대범위 4방향): **10/20 = 50.0%**
- corner(기존+확대 코너 6개): **17/30 = 56.7%**
- clamp_count=0, scene_invalid_count=0
- **grasp/place 모순 1건**: `expanded_corner_pn` repeat=3 (`grasp_ever=True`, `place_success=False`) -- 잡았지만 최종 배치에 실패한 유일한 사례.

Expert 자체가 125/125=100%로 완전히 깨끗했으므로, 이 61.3%는 **scene/Expert 문제와 섞이지 않은 순수한 정책 일반화 격차**로 해석할 수 있다. 흥미롭게도 이 수치는 Stage 1B 문서가 기록한 box의 zero-shot grasp율(47/75=62.7%)과 거의 동일한 값이다 -- 우연이지만 "새로운 shape에 대한 zero-shot 일반화 격차"가 비슷한 크기로 반복되는 패턴으로 참고할 만하다.

## 21. Stage 1C dataset 생성 가능 여부

**가능 (Expert 기준 충족).** 9절 통과 기준 중 Expert 관련 항목(V1 regression 통과, cube/box V2.1 성공률 유지, cylinder valid-scene Expert 성공률 95%↑, cylinder physical success 95%↑, 특정 위치 그룹 전면 실패 없음, NaN/Inf 0, joint limit violation 0, legacy/physical 차이 없음, cylinder 치수/scene config 재현 가능하게 고정됨)를 **전부 충족**했다. Zero-shot 수치(61.3%)는 **go/no-go 게이트가 아니라 학습 전 baseline**으로만 사용한다 (이번 task의 명시적 지시: "결과를 보고 checkpoint를 재선정하지 않는다").

## 22. 아직 남은 cylinder 한계

- 이번 검증은 **upright cylinder만** 다뤘다 (옆으로 눕거나 기울어진 경우, roll/pitch 변화는 범위 밖).
- `contact_count`가 이 시뮬레이션에서 구조적으로 판별력이 없다는 사실(2절) -- 실제 물리적 파지 안정성에 대한 시뮬레이션 차원의 근본적 한계이며, cylinder 도입으로 새로 생긴 문제가 아니라 기존 cube/box부터 있던 사실이다.
- gripper의 실제 최대 opening에 대한 명시적 스펙이 없어, 이번 후보 선정은 "기존 검증된 스케일 재사용"에 의존했다 -- 진짜 물리적 상한을 알지 못한다.
- Zero-shot 성능(61.3%)은 아직 cylinder 형태에 대해 전혀 학습되지 않은 상태의 참고값일 뿐이다.

## 23. 다음 수정 또는 dataset 수집 권장안

**Stage 1C cylinder dataset 수집 진행을 권장한다** (사용자 승인 전제, 이번 task에서는 미수행). 권장 규모/구성은 Stage 1B의 rehearsal 방식(신규 cylinder + 기존 cube/box 일부 재사용)을 그대로 따르는 것을 제안하며, 구체적 episode 수/영역 배분은 다음 단계에서 별도 승인 하에 설계한다.

## 24. Git staging 추천 파일

- 신규 6개: `benchmark/so101_expert_v2_size_aware.py`, `benchmark/evaluate_so101_expert_v2_cylinder.py`, `benchmark/evaluate_so101_stage1b_cylinder_zeroshot.py`, `benchmark/test_so101_expert_v2_size_aware.py`, `configs/so101_stage1c_cylinder_candidates.json`, `docs/experiments/so101_stage1c_cylinder_feasibility_summary.md`.
- 수정 3개(additive만): `robot_sim/so101_pybullet_backend.py`, `benchmark/so101_smolvla_rollout.py`, `docs/experiments/README.md`(인덱스).

## 25. 제외해야 할 dataset/result/log

`results/so101_expert_v2_cylinder/`, `results/so101_stage1c_cylinder_feasibility/`(gitignored, 기존 정책과 동일), 스크래치패드 임시 스크린 스크립트(프로젝트 디렉토리 밖).

## 26. 공용 및 Laptop 트랙 충돌 여부

Laptop/ROS2/YOLO/Safety/Dashboard 파일은 조회도 수정도 하지 않았다. `benchmark/so101_expert_v2_orientation.py`와 그 결과(`results/so101_expert_v2_yaw_grid/`, `docs/experiments/so101_expert_v2_orientation_summary.md`)는 **전혀 건드리지 않았다** -- 조회조차 하지 않아 절대 원칙 3(보존)을 완전히 준수했다. 이번 작업에서 git add/commit/push는 수행하지 않았다.

## 27. 실행 명령

```
.venv-vla/bin/python -m benchmark.test_so101_expert_v1_regression
.venv-vla/bin/python -m benchmark.test_so101_expert_v2_size_aware
.venv-vla/bin/python -m benchmark.evaluate_so101_expert_v2_cylinder
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_cylinder_zeroshot
```

결과 파일: `results/so101_expert_v2_cylinder/cylinder_expert_records.jsonl`(125 rows), `results/so101_expert_v2_cylinder/cylinder_expert_summary.json`, `results/so101_stage1c_cylinder_feasibility/zeroshot_cylinder_on_stage1b_checkpoint/results.json`. `results/`는 Git에 포함되지 않는다.
