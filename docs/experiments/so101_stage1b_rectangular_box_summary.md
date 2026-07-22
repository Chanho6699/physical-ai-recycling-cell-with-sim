# SO-101 Stage 1B: Rectangular-Box Shape Generalization Summary

## 1. Stage 1B 목적

Stage 1A에서 확보한 object-XY 위치 일반화 능력을 유지하면서, 큐브와 다른 종횡비를 가진 직육면체 상자를 pick-and-place할 수 있도록 확장한다. 이번 단계에서 바꾼 변수는 **object shape 하나뿐**이다(yaw/bin 위치/시작 joint pose/camera/instruction/mass/friction은 전부 고정).

## 2. 왜 rectangular box를 첫 shape 확장으로 선택했는지

큐브 이후 가장 단순한 형태 변화이면서, 기존 scripted expert(위치 전용 IK, 고정 gripper 방향)의 한계를 드러내기에 적합한 대상이었다. 실제로 이번 조사에서 **그리퍼가 닫히는 축이 월드 X축으로 고정**되어 있다는 사실이 드러났고(2.0배 비율 box를 반대 방향으로 놓으면 expert 자체가 100% 실패), 이는 향후 회전(yaw)/다른 형태 확장 전에 반드시 파악해야 할 구조적 사실이었다.

## 3. 물체 크기와 물성

- `object_footprint_xy = [0.02, 0.03]` (half-extent) → 실제 크기 **4cm(X, 그립축) × 6cm(Y, 긴축) × 4cm(height, 큐브와 동일)**, 종횡비 1.5배
- Mass: 큐브와 동일(`OBJECT_MASS=0.05kg`, 코드 상수 미변경)
- Friction: 큐브와 동일(PyBullet 기본값 0.5, 명시적 설정 없음 — 둘 다 미설정이므로 완전히 동일)
- **결정 근거(실측)**: 2.0배 비율(4cm×8cm)을 테스트한 결과 올바른 방향(X=그립축 짧게)에서는 3/3 성공했지만, **+Y 방향 모서리(Stage 1A 확대범위)에서 `object_bin_overlap`로 씬 자체가 무효화됨**을 직접 확인 → 1.5배(4cm×6cm)로 축소해 Stage 1A 전체 XY 범위(기존+확대, 4개 모서리 포함)에서 씬 유효성과 scripted expert 성공을 전부 재확인한 뒤 채택. 반대 방향(X=긴축)은 방향 자체가 grasp에 반복 실패(`place_waypoint_failed` 3/3)해 기각.
- `robot_sim/so101_pybullet_backend.py`는 **코드 수정 없이** `scene_config["object_footprint_xy"]` override만으로 형태 교체 가능함을 확인.

## 4. Zero-shot 결과 (Stage 1A 7500-step checkpoint, box 형태에 대해 학습 전)

15개 고정 위치 × 5 repeat = 75 rollouts, base seed 500000:
- **47/75 = 62.7%**
- `existing_corner_pp` 완전 실패(0/5), 확대 범위 위치 전반이 기존 범위보다 약함
- clamp/NaN·Inf/scene_invalid 전부 0

## 5. 신규 데이터셋 구성

`datasets/so101_bin_stage1b_box_100` — 100 episodes (train 70 / validation 15 / test 15), 13개 stratified 영역(중앙, 기존범위 단일축 4개, -X corridor, 4개 모서리, 3개 bridge)에 걸쳐 연속좌표 seed 샘플링. Seed 블록: train 15000대, validation 17000대, test 18000대(전부 상호 배타적, 기존 프로젝트의 어떤 seed 블록과도 안 겹침). **100/100 저장, discard 0건.**

## 6. Rehearsal cube/box 학습 구성 (총 160 episodes)

- 신규 train 70개(box) 전부 사용
- cube 45개: 기존 200-episode dataset의 train 160개 중 중심으로부터 거리 3분위(near/mid/far) 층화 추출
- cube 45개: Stage 1A의 신규 train 50개(-X corridor/모서리/bridge) 중, 8개 영역 중 6개 이상인 5개 영역에서 각 1개씩(최신 episode_index) 제외한 결정론적 축소 선택
- `configs/so101_stage1b_train_episodes.json`에 선택된 index/seed 전체 기록. **무결성 검증 30/30 통과**(원본 dataset sha256 불변 포함) — 검증 과정에서 발견한 버그 2건(검증기 자체의 과도한 370-합계 기대치, rehearsal manifest의 Stage 1A val/test index 이중 오프셋)을 학습 전에 발견·수정.

## 7. Fresh optimizer fine-tuning

`--policy.path=<Stage 1A 7500-step checkpoint>`(모델 weight만), `--resume` 미사용. 로그로 확인: `pretrained_path`가 정확히 Stage 1A 7500 checkpoint, step 10부터 fresh warmup 시작(resume 아님), `dataset.num_episodes=160`. `--steps=10000 --save_freq=2500`로 4개 candidate(2500/5000/7500/10000) 확보.

## 8. Candidate별 validation 결과

| candidate | cube retention (400) | box validation (75) | clamp | 모순 |
|---|---|---|---|---|
| 2500 | 65/400 = 16.25% | 15/75 = 20.0% | 0 | 0 |
| 5000 | 185/400 = 46.25% | 31/75 = 41.3% | 0 | 0 |
| **7500** | **328/400 = 82.00%** | **57/75 = 76.0%** | **0** | **0** |
| 10000 | 322/400 = 80.50% | 57/75 = 76.0% | 0 | 0 |

## 9. 최종 checkpoint 선택 이유

**7500-step 선택.** box validation은 10000-step과 동률(76.0%)이지만 cube retention이 4개 후보 중 최고(82.0%)이고 step 수도 더 이르다 — 우선순위 1(box 성능)·2(retention 유지)·4(clamp/NaN-Inf)·5(동률 시 더 이른 checkpoint) 전부 충족. **주의**: 우선순위 3("확대 XY cube 성능이 심하게 악화되지 않음")은 이번 라운드에서 별도로 재측정하지 않았다 — 표준 40-seed retention은 기존 ±0.015 범위만 다루므로, Stage 1A 확대범위 cube 성능은 이번 후보 비교에 포함되지 않은 한계로 남긴다.

## 10. Held-out test

`outputs/train/so101_stage1b_box_reinforcement_20260722_101757/checkpoints/007500/pretrained_model`, box held-out test 15위치 × 10 repeat = 150 rollouts, base seed 700000:
- **100/150 = 66.7%**, clamp 0, grasp/place 모순 0
- 영역별: center 14/20(70%), existing_x_min 9/10(90%), existing_x_max 8/10(80%), existing_y_min 9/10(90%), existing_y_max 8/10(80%), x_min_corridor 12/20(60%), corner_pp 5/10(50%), corner_pn 4/10(40%, 최약), corner_np 5/10(50%), corner_nn 6/10(60%), bridge_plus_x 7/10(70%), bridge_minus_y 8/10(80%), bridge_plus_y 5/10(50%)
- **validation(76.0%) 대비 test(66.7%)는 9.3%p 낮음** — Stage 1A의 선례(validation 70.0%→test 57.0%, 13%p 격차)보다는 작지만, 여전히 validation이 test보다 낙관적이라는 동일한 패턴 반복. **선택된 checkpoint는 이 결과를 보고 재선정하지 않았다.**

## 11. Cube retention

**82.0%로 Stage 1A 기준(78.5%) 대비 오히려 상승** — box 학습을 rehearsal과 함께 진행한 결과 cube 능력 손실(forgetting) 없음.

## 12. 실패 단계 분석

전체 결과에서 `grasp_was_ever_established`와 `model_rollout_place_success`가 **완전히 일치**(모순 0건, 전 candidate·held-out test 공통) — grasp이 성공하면 항상 place까지 성공하는 이 프로젝트의 기존 패턴이 box에서도 동일하게 유지됨. 즉 실패는 전부 **pre-grasp 단계**(애초에 잡지 못함)이고, transport/place 단계에서 새로 발생한 실패는 없음. 다만 diagnostic_log를 활성화하지 않아 step 단위(예: gripper close 타이밍)의 세부 원인 분석은 이번에 수행하지 않았다.

## 13. 실험 한계

- 위치당 반복 수가 작음(validation 5회, test 10회) — 개별 위치 순위를 과도하게 일반화하지 않는다.
- Fine-tuning 깊이 하나(10,000 step, 4개 candidate)만 탐색.
- Box 크기/형태/질량/마찰 하나만 검증 — 다른 크기·비율에 대한 일반화는 미검증.
- Object yaw는 전혀 변경하지 않음 — 그리퍼의 고정 그립축이라는 구조적 한계가 여전히 남아 있음.
- Validation-test 격차(9.3%p)가 존재해, validation 성능만으로 실제 배포 성능을 과신하면 안 됨.

## 14. 다음 단계 추천

**Box yaw 랜덤화를 cylinder/can 형태 확장보다 먼저 권장한다.** 이번 조사에서 그리퍼가 닫히는 축이 월드 X로 고정돼 있다는 구조적 사실이 드러났고, 이는 형태를 다양화하는 것보다 더 근본적인 격차다 — 회전이 조금이라도 생기면 지금 구조(position-only IK, orientation 미제어)로는 대응이 안 될 가능성이 높다. Cylinder/can처럼 회전 대칭인 물체는 이 문제를 우연히 회피할 수 있어, 오히려 문제를 늦게 발견하게 만들 위험이 있다.

## 15. 실행 명령과 repository-relative 경로

```
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_zeroshot
.venv-vla/bin/python -m benchmark.collect_so101_stage1b_box_dataset
.venv-vla/bin/python -m benchmark.build_so101_stage1b_rehearsal_manifest
.venv-vla/bin/python -m benchmark.merge_so101_dataset_for_training_stage1b
.venv-vla/bin/python -m benchmark.validate_so101_stage1b_dataset
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_expansion --checkpoint-dir <ckpt> --split validation --policy-noise-repeats 5 --policy-noise-base-seed 600000 --output-path <path>
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_expansion --checkpoint-dir <ckpt> --split test --policy-noise-repeats 10 --policy-noise-base-seed 700000 --output-path <path>
```

경로: `datasets/so101_bin_stage1b_box_100`, `datasets/so101_bin_stage1b_training_combined`, `configs/so101_stage1b_train_episodes.json`, `outputs/train/so101_stage1b_box_reinforcement_20260722_101757/checkpoints/007500/pretrained_model`(최종 선택), `results/so101_stage1b_reinforcement/`.

`datasets/`, `outputs/`, `results/`는 Git에 포함되지 않는다.
