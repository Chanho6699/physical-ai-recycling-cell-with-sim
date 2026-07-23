# SO-101 Stage 1C: Upright Cylinder 정식 Dataset·학습·평가 요약

## 1. Stage 1C 목적

Stage 1B 결과(cube retention 82.0%, box validation 76.0%, box held-out 66.7%)와 Expert V1/orientation Expert V2를 그대로 보존한 채, 검증이 완료된 size-aware Expert V2.1을 사용해 upright cylinder를 새 물체 shape으로 SmolVLA에 정식으로 도입하는 첫 실사용 단계다. Stage 1C 사전 검증(Expert V2.1 125/125=100%, zero-shot 61.3%)에서 확인된 "Expert 자체는 문제없음"을 전제로, 실제 dataset 수집·rehearsal 학습·checkpoint 선정·held-out test까지 전체 파이프라인을 완료했다.

## 2. 원통을 재활용 캔의 기초 형태로 선택한 이유

이번 원통은 실제 캔의 정확한 치수·라벨·재질을 복제한 것이 아니라, "재활용 캔"이라는 최종 목표를 향한 **기초(base) 형태**다. 원통은 (a) yaw 회전 대칭이라 orientation 처리 없이도 Expert가 그대로 재사용 가능하고, (b) cube/box와 달리 곡면 collision 형태라는 명확한 시각적 차별점이 있어 shape 일반화 능력을 시험하기 좋으며, (c) 이전 조사에서 이미 지름 4cm(기존 cube/box와 동일 스케일)로 물리적 실현 가능성을 확인했기 때문이다.

## 3. 원통 크기와 scene 설정

radius=0.02m(지름 4cm), height=0.04m — 기존 cube(`object_footprint_xy=[0.02,0.02]`)·box(grasp축 폭 4cm)와 정확히 같은 스케일. mass=0.05kg, lateral_friction=0.5, rolling_friction=0.0, spinning_friction=0.0, restitution=0.0 — 전부 PyBullet 기본값(cube/box와 동일, 이번 task에서 randomization 없음), `p.getDynamicsInfo()`로 실측해 metadata에 고정 기록했다.

## 4. Expert V2.1 검증 결과 (재확인)

이전 사전 검증에서 Expert V2.1은 125/125(=100%) legacy_success/physical_success를 기록했다(5개 위치 그룹 × 5개 scene yaw × 5 seed). 이번 정식 데이터 수집에서도 Expert V2.1을 사용했고, **100/100 saved, discard 0, 실패 0건**으로 그 결과가 실제 데이터 규모에서도 그대로 재현됨을 확인했다.

## 5. Constraint 기반 grasp 한계

이 시뮬레이션의 grasp는 **실제 finger contact/friction 기반이 아니라 EE-object 거리(≤0.04m)와 gripper 닫힘 각도(정규화 ≤0.15) 조건으로 트리거되는 고정(fixed) constraint**다. 이전 조사에서 지름 3cm~10cm 전 범위(및 이미 검증된 cube/box)에서 `contact_count=0`이 실측됐다 -- 즉 실제 물리적 접촉력으로 물체를 붙잡는 것이 아니다. 이 문서 전체에서 "grasp 성공"은 예외 없이 **constraint-based simulated pick-and-place**를 의미하며, physically realistic grasp나 real-world grasp verified, contact-based grasp success로 표현하지 않는다. 결과 metadata에도 `constraint_based_grasp=true, contact_physics_verified=false`를 모든 episode에 명시적으로 기록했다.

## 6. Dataset 분포

`datasets/so101_bin_stage1c_cylinder_100` -- 100 episodes, Expert V2.1로 수집, **100/100 저장, discard 0, 실패 0건**, NaN/Inf 0.

## 7. Train/validation/test split

| split | 수 |
|---|---|
| train | 70 |
| validation | 15 |
| test | 15 |

각 split은 서로 다른 seed 블록(train 20000번대, validation 21000번대, test 22000번대) 사용, 전부 상호 배타적(중복 0). Split은 데이터 생성 **전** 코드(`POSITION_GROUP_PLAN`)로 고정했다.

### 위치 그룹 분포

| group | train | validation | test | 합계 |
|---|---|---|---|---|
| center | 10 | 3 | 3 | 16 |
| interior | 15 | 3 | 3 | 21 |
| edge | 15 | 3 | 3 | 21 |
| corner | 20 | 3 | 3 | 26 |
| x_min_corridor | 10 | 3 | 3 | 16 |

zero-shot 결과(known/interior 76.0%, expanded edge 50.0%, corner 56.7%)를 반영해 권장대로 train에서 corner(20)·edge(15)·interior(15)를 center/x_min_corridor(10)보다 강화했다. validation/test는 각 그룹당 3개로 균형 구성. interior/edge/corner는 REGION_DEFS의 여러 하위 영역(예: interior=existing_x_min/x_max/y_min/y_max, corner=corner_pp/pn/np/nn)에 train을 분산시켰고, validation/test는 이전 Expert 평가에서 사용한 대표 하위 영역(existing_x_min, bridge_plus_x, corner_pn)으로 고정해 이전 평가와 직접 비교 가능하게 했다 -- 이 조정 사유를 그대로 보고한다.

### Yaw metadata 분포

0°/45°/90°/135°/180°를 위치별로 순환 배정(회전 대칭성 검증용 metadata). Grasp strategy는 yaw를 전혀 사용하지 않는다(Expert V2.1은 yaw-무관).

## 8. 데이터 무결성 결과

`benchmark/validate_so101_stage1c_dataset.py`: **79/79 통과** (episode 구조, rehearsal 구성, split leakage 0, feature schema, NaN/Inf 0, gripper 0-100 scale, joint order, fps=10, action/trajectory, scene metadata 전부 포함). 이미지 검증은 결정론적 샘플(cylinder 15개 episode의 첫 프레임)을 PIL로 실제 PNG 디코드해 256×256×3 uint8 RGB임을 직접 확인했고, 전체 470-episode에 대해서는 이미지 전수 decode 대신 parquet 파일 개수/episode 개수 무결성(전수)으로 결합했다 -- 이 전략을 그대로 문서화했다(전수 decode는 과도하게 느려 조합 방식을 채택).

## 9. 발견·수정한 dataset 버그

이미지 검증 초기 구현에서 parquet 컬럼의 이미지가 raw 배열이 아니라 `{"bytes": <PNG>, "path": ...}` dict로 저장된다는 사실을 놓쳐 `np.array()`로 감싸면 shape=()·dtype=object인 무의미한 배열이 되는 버그가 있었다(2건 FAIL). PIL로 PNG bytes를 실제 디코드하도록 수정해 해결했다 -- 학습 전 발견했으며 학습에는 영향 없음.

## 10. Rehearsal manifest 구성

`configs/so101_stage1c_train_episodes.json`: cube 55(원본 28 + Stage1A 27, 둘 다 validation/test 미사용, tertile/region 비례 축소) + box 55(Stage1B 70개 train 중 13개 region 비례 축소, validation/test 미사용) + cylinder 70(Stage1C 전체 train) = **180개**.

## 11. Cube/box/cylinder episode 수

cube 55, box 55, cylinder 70, 합계 180 -- 전부 `validate_so101_stage1c_dataset.py`로 재확인.

## 12. Split leakage와 duplicate 검사

train ∩ (existing_validation, stage1a_validation, stage1a_test, box_validation, box_test, cylinder_validation, cylinder_test) 전부 공집합 확인(**leakage 0**), 180개 train index 중복 0, source episode duplicate 0.

## 13. Training command

```
env VLA_DEVICE=cuda VLA_DTYPE=float32 .venv-vla/bin/lerobot-train \
  --dataset.repo_id=local/so101_bin_stage1c_training_combined \
  --dataset.root=datasets/so101_bin_stage1c_training_combined \
  --dataset.episodes=<configs/so101_stage1c_train_episodes.json의 180개 index> \
  --policy.path=outputs/train/so101_stage1b_box_reinforcement_20260722_101757/checkpoints/007500/pretrained_model \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --output_dir=outputs/train/so101_stage1c_cylinder_reinforcement_20260723_162442 \
  --steps=10000 --save_freq=2500 --batch_size=1 --seed=0
```

`--rename_map`은 정책이 학습 최초 시점부터 declare해 온 `camera1/camera2/camera3` 입력 이름과 이 프로젝트의 단일 카메라(`front`) 데이터셋 이름을 맞추기 위한 필수 인자다(첫 시도에서 이 인자 없이 실행해 `ValueError: Feature mismatch`로 즉시 실패했고, 원인 확인 후 추가해 해결 -- 기존 문서에는 생략되어 있었으나 Stage1A/1B의 모든 checkpoint도 동일하게 `camera1/2/3` 입력을 declare하고 있어 실제로는 매 단계 필요했던 인자로 확인됐다).

## 14. Fresh optimizer/scheduler 확인

로그에서 직접 확인: `resume: False`, `policy.pretrained_path`가 정확히 Stage 1B 7500-step checkpoint, `Creating optimizer and scheduler`가 학습 시작 시 새로 실행됨. **LR 곡선**(scheduler_warmup_steps=1000, peak_lr=1e-4)이 fresh 시작의 결정적 증거: step 200에서 lr=3.0e-05로 시작해 step 600~800 사이 peak(9.9e-05)에 도달한 뒤 cosine decay(step 5000: 5.3e-05, step 7500: 1.9e-05)로 하강 -- resume 시 발생했던 "LR 재상승" 패턴과 다른, 정상적인 단일 warmup-then-decay 곡선이다. Dataset은 180 episodes(cube 55/box 55/cylinder 70)로 정확히 로드됨을 config 로그로 재확인했다. 기존 output 디렉토리를 덮어쓰지 않았다(신규 job_name).

## 15. Candidate checkpoint 목록

2500 / 5000 / 7500 / 10000 (steps).

## 16. Candidate별 cube retention (400 rollouts, 40 seed × 10 policy-noise repeat)

| step | cube retention |
|---|---|
| 2500 | 55/400 (13.8%) |
| 5000 | 200/400 (50.0%) |
| 7500 | 327/400 (81.8%) |
| **10000** | **348/400 (87.0%)** |

## 17. Candidate별 box validation (75 rollouts, Stage 1B validation split 그대로 재사용)

| step | box validation |
|---|---|
| 2500 | 7/75 (9.3%) |
| 5000 | 39/75 (52.0%) |
| 7500 | 50/75 (66.7%) |
| **10000** | **58/75 (77.3%)** |

## 18. Candidate별 cylinder validation (75 rollouts, 15 위치 × 5 policy-noise repeat)

| step | cylinder validation |
|---|---|
| 2500 | 10/75 (13.3%) |
| 5000 | 40/75 (53.3%) |
| 7500 | 54/75 (72.0%) |
| **10000** | **56/75 (74.7%)** |

## 19. Clamp/NaN/모순 결과

4개 candidate 전부 **clamp_count=0**. NaN/Inf: rollout 자체가 non-finite 값에서 즉시 abort하도록 되어 있고 4개 candidate 전부 `aborted_count`/`aborted_early` 이상 없이 정상 종료 -- 0건. Grasp/place 모순(문서 §17 참고, held-out에서 상세): candidate 단계에서는 별도 집계하지 않았고 held-out에서 확인(cylinder 2건, cube/box 0건).

## 20. 공식 checkpoint 선택

**10000-step 선택.** 근거(§21).

## 21. 선택 근거

| checkpoint | cube retention | box validation | cylinder validation | worst category | clamp/NaN/모순 | 선정 |
|---|---|---|---|---|---|---|
| 2500 | 13.8% | 9.3% | 13.3% | box | 0/0/0 | 기각 (전 항목 미달) |
| 5000 | 50.0% | 52.0% | 53.3% | cube | 0/0/0 | 기각 (전 항목 미달) |
| 7500 | 81.8% | 66.7% | 72.0% | box (66.7%<70% 기준 미달) | 0/0/0 | 기각 (box 기준 미달) |
| **10000** | **87.0%** | **77.3%** | **74.7%** | cylinder (그래도 74.7%≥70%) | 0/0/0 | **선정** |

권장 최소 기준(cube≥75%, box≥70%, cylinder≥70%)을 **동시에 충족하는 유일한 candidate는 10000**이다. 7500은 box(66.7%)가 기준에 못 미쳐 기각했다. 10000은 세 항목 모두에서 7500보다 우수해(cube +5.2pp, box +10.6pp, cylinder +2.7pp) trade-off 없이 명확히 우월하다. Zero-shot(61.3%) 대비 cylinder validation은 +13.4%p 개선. Stage 1B 공식 cube retention(82.0%)/box validation(76.0%) 대비 10000은 오히려 소폭 상승(forgetting 없음).

## 22. Cylinder held-out 결과

`outputs/train/so101_stage1c_cylinder_reinforcement_20260723_162442/checkpoints/010000/pretrained_model`, 15 held-out 위치 × 10 repeat = 150 rollouts, base seed 950000:

**111/150 = 74.0%**, clamp 0, grasp-but-no-place(가까이 있었지만 배치 실패) 2건.

## 23. 위치 그룹별 held-out 결과

| group | 성공률 |
|---|---|
| center | 28/30 (93.3%) |
| interior | 23/30 (76.7%) |
| edge | 19/30 (63.3%) |
| corner | 18/30 (60.0%) |
| x_min_corridor | 23/30 (76.7%) |

center가 가장 강하고 edge/corner가 상대적으로 약한 패턴(zero-shot 시점의 known/interior>corner>edge 순서와 유사한 난이도 서열이 학습 후에도 남아 있음).

## 24. Cube 최종 retention

352/400 = **88.0%** (fresh noise base seed 200000로 재확인). Stage 1B 공식 82.0% 대비 forgetting 없이 오히려 상승. clamp 0, grasp/place 모순 0.

## 25. Box 최종 retention

Held-out test(Stage 1B 자신의 test split, base seed 700000): **104/150 = 69.3%**. Stage 1B 공식 held-out(66.7%) 대비 유지/소폭 상승. clamp 0, grasp/place 모순 0. Validation(77.3%) 대비 test는 8.0%p 낮음 -- Stage 1B 자신의 validation-test 격차(9.3%p)와 유사한 크기의 동일 패턴.

영역별: center 16/20(80%), existing_x_min 10/10(100%), existing_x_max 8/10(80%), existing_y_min 10/10(100%), existing_y_max 8/10(80%), x_min_corridor 15/20(75%), corner_pp 4/10(40%), corner_pn 4/10(40%), corner_np 3/10(30%, 최약), corner_nn 6/10(60%), bridge_plus_x 7/10(70%), bridge_minus_y 8/10(80%), bridge_plus_y 5/10(50%) -- corner 영역이 여전히 가장 약함(Stage 1B 자신의 기존 패턴과 일치, box rehearsal 55/70만 사용한 것이 이 패턴에 영향을 줬을 가능성은 배제하지 않는다).

## 26. Zero-shot 61.3% 대비 개선폭

Cylinder validation 74.7% 기준 **+13.4%p**, held-out 74.0% 기준 **+12.7%p** 개선. Zero-shot 평가 자체가 Expert 100%로 이미 scene/Expert 문제와 분리되어 있었으므로, 이 개선폭은 순수한 학습 효과로 해석할 수 있다.

## 27. Validation-test gap

- Cube: validation 87.0% -> 최종 재확인 88.0% (사실상 동일 seed 재사용, gap 없음 -- 별도 held-out env 세트가 없는 cube의 기존 관례).
- Box: validation 77.3% -> held-out 69.3%, **8.0%p 격차** (Stage 1B 자신의 9.3%p 격차와 유사).
- Cylinder: validation 74.7% -> held-out 74.0%, **0.7%p 격차** (매우 작음 -- 새 shape임에도 validation이 test 성능을 과대평가하지 않음).

## 28. Stage 1C 성공 여부

**Stage 1C successful.**

근거: 원통 데이터 수집·무결성 검증 완료(100/100, discard 0, 79/79 검증 통과), split leakage 0, rehearsal merge 정상(180=55+55+70, 470-episode 결합 dataset sha256로 원본 3개 dataset 무손상 확인), fresh optimizer/scheduler LR 곡선으로 직접 확인, cylinder validation(74.7%)이 zero-shot(61.3%)보다 명확히 개선, cube forgetting 없음(오히려 상승), box forgetting 없음(오히려 소폭 상승), held-out 결과가 validation과 과도하게 괴리되지 않음(cylinder 0.7%p, box 8.0%p -- Stage1B 자체 선례 범위 내), clamp/NaN 0. 다만 box held-out(69.3%)은 validation 시점 권장 threshold(70%)에 test에서는 미달했고 corner 영역(특히 corner_np 30%)이 여전히 약하다는 한계가 있어, "완벽"이 아닌 "successful"로 과장 없이 평가한다.

## 29. Constraint-based grasp 한계

전체 파이프라인의 grasp 판정은 EE-object 거리+gripper 각도 기반 constraint이며 실제 finger contact/friction 기반이 아니다(§5). 즉 이 문서의 모든 성공률은 **constraint-based simulated pick-and-place** 성능이며, 실물 로봇에서의 물리적 grasp 견고성을 보장하지 않는다.

## 30. 다음 수정 또는 dataset 수집 권장안

- box corner 영역(특히 corner_np) 약점이 반복 관찰됨 -- box rehearsal 확대 또는 corner 전용 추가 수집 검토.
- 실제 캔 비율(높이/지름 확장), 다른 크기의 원통, cylinder 대비 실제 contact-force 기반 grasp 모델 도입은 이번 범위 밖으로 다음 단계 후보.
- cube/box rehearsal을 55개로 줄인 것이 box corner 성능에 미친 영향은 이번에 별도로 통제 실험하지 않았다 -- 필요시 rehearsal 크기 sensitivity를 별도 확인 권장.

## 31. Git staging 추천 파일

신규: `benchmark/collect_so101_stage1c_cylinder_dataset.py`, `benchmark/build_so101_stage1c_rehearsal_manifest.py`, `benchmark/merge_so101_dataset_for_training_stage1c.py`, `benchmark/validate_so101_stage1c_dataset.py`, `benchmark/evaluate_so101_stage1c_expansion.py`, `configs/so101_stage1c_train_episodes.json`, `docs/experiments/so101_stage1c_cylinder_summary.md`.
수정(additive만): `benchmark/so101_smolvla_rollout.py`(`object_shape_override`/`object_radius_override` 신규 optional 인자), `docs/experiments/README.md`(인덱스).

## 32. 제외할 dataset/output/result 경로

`datasets/so101_bin_stage1c_cylinder_100/`, `datasets/so101_bin_stage1c_training_combined/`, `outputs/train/so101_stage1c_cylinder_reinforcement_20260723_162442/`, `results/so101_stage1c_reinforcement/` -- 전부 기존 `.gitignore` 정책(datasets/outputs/results 전체 미포함) 그대로 적용.

## 33. 공용 및 Laptop 트랙 충돌 여부

Laptop/ROS2/YOLO/Safety/Dashboard 파일은 조회·수정 모두 하지 않았다. 기존 Stage 1A/1B dataset·checkpoint는 sha256로 무손상 확인했다. Expert V1(`so101_scripted_expert.py`)과 orientation Expert V2(`so101_expert_v2_orientation.py`) 파일은 이번 task에서 조회조차 하지 않아 완전히 보존됐다.

## 34. 실행 명령

```
.venv-vla/bin/python -m benchmark.collect_so101_stage1c_cylinder_dataset
.venv-vla/bin/python -m benchmark.build_so101_stage1c_rehearsal_manifest
.venv-vla/bin/python -m benchmark.merge_so101_dataset_for_training_stage1c
.venv-vla/bin/python -m benchmark.validate_so101_stage1c_dataset
# training: 위 §13 참고
.venv-vla/bin/python -m benchmark.so101_smolvla_rollout --checkpoint-dir <ckpt> --rollout-seeds <40 seeds> --policy-noise-repeats 10 --policy-noise-base-seed <seed> --output-path <path>
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_expansion --checkpoint-dir <ckpt> --split validation|test --policy-noise-repeats 5|10 --policy-noise-base-seed <seed> --output-path <path>
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1c_expansion --checkpoint-dir <ckpt> --split validation|test --policy-noise-repeats 5|10 --policy-noise-base-seed <seed> --output-path <path>
```

경로: `datasets/so101_bin_stage1c_cylinder_100`, `datasets/so101_bin_stage1c_training_combined`, `configs/so101_stage1c_train_episodes.json`, 최종 선택 checkpoint `outputs/train/so101_stage1c_cylinder_reinforcement_20260723_162442/checkpoints/010000/pretrained_model`, `results/so101_stage1c_reinforcement/`. `datasets/`, `outputs/`, `results/`는 Git에 포함되지 않는다.
