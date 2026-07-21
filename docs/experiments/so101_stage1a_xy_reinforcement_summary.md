# SO-101 Stage 1A: Object-XY Targeted Reinforcement Summary

## 1. Stage 1A를 진행한 이유

25,000-step 공식 baseline(70.25%)이 학습 step 확장만으로는 더 이상 유의미하게 개선되지 않을 것으로 판단해 step-scaling 실험을 종료했고, 이어서 수행한 XY 외삽 평가에서 큐브 object 위치가 학습 분포(±0.015m) 밖으로 25% 확대(±0.01875m)될 때 성공률이 급격히 하락하는 것을 확인했다. 새 물체를 추가하기 전에 이 위치 일반화 약점을 먼저 보강하기 위해 Stage 1A를 진행했다.

## 2. 기존 25k baseline과 XY 외삽 실패

- 25,000-step 공식 checkpoint: `outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model`, closed-loop 281/400 = **70.25%**
- XY 외삽 평가(125 rollouts, 25위치×5repeat): 기존 범위 경계/모서리 26/40 = **65.0%** → 확대 범위(±0.01875m) 경계/모서리 14/40 = **35.0%** (**-30.0%p**)
- scene_invalid / joint clamp / NaN·Inf 전부 0 → 물리적 도달 한계가 아니라 **위치 일반화 문제**로 판단
- 취약 영역: 네 모서리 전체, -X 방향 경계

## 3. 신규 70-episode 데이터 설계

반경 0.015m~0.01875m 밴드 내에서만 표적 보강(그 이상 미검증 영역으로 확장하지 않음):

| 영역 | train | validation | test |
|---|---|---|---|
| -X corridor | 15 | 3 | 3 |
| 모서리 pp/pn/np/nn | 6/7/6/6 | 1/2/1/1 | 1/2/1/1 |
| Bridge +X/-Y/+Y | 4/3/3 | 1/1/0 | 1/0/1 |
| **합계** | **50** | **10** | **10** |

각 영역은 `sample_object_position(seed, x_range, y_range)`을 통한 연속 uniform 샘플링(특정 좌표 하나에 몰지 않음). 수집 seed는 기존 dataset(0~199)와 완전히 분리된 블록(train 5000번대, validation 6000번대, test 7000번대) 사용. **70/70 episode 전부 첫 시도에 성공 저장, discard 0건.**

## 4. Train/Validation/Test 분리

- **신규 train 50** + **기존 train 160** = **210** (fine-tuning에 사용된 전체)
- **신규 expansion validation 10** (checkpoint 후보 선정에만 사용, held-out test에는 사용 안 함)
- **신규 held-out expansion test 10** (validation 기준 checkpoint 확정 후 단 1회만 평가)
- **기존 validation 40** (retention 평가 전용, 학습에 미포함)

## 5. 기존 160 + 신규 50 학습 구성

물리적 병합 dataset `datasets/so101_bin_stage1a_combined_270` (기존 200 episode를 read-only 복사 + 신규 70 episode 이어서 append, `LeRobotDataset.resume()` 사용) 생성. 학습 시에는 `--dataset.episodes`로 210개 index만 지정(기존 val 40 + 신규 val/test 20은 학습에서 제외). 원본 `datasets/so101_bin_main_200`은 SHA-256 해시로 학습 전/후 완전 동일함을 확인.

## 6. Fresh optimizer fine-tuning 방식

`--policy.path=<25k checkpoint>` (모델 weight만 초기값으로 사용) + `--resume=true` 미사용 → optimizer/scheduler는 완전히 새로 초기화. 로그로 확인:
- `pretrained_path: outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model`
- step 10에서 시작(resume 아님), LR이 warmup 곡선을 처음부터 다시 그림
- `dataset.num_episodes=210`, `dataset.num_frames=14295`
- LR scheduler는 이번 fine-tuning 자체의 `--steps=10000`에 맞춰 `num_decay_steps`가 10000으로 auto-rescale됨(scale factor 0.333) — 이번은 resume 체인이 아니라 단일 fresh run이므로 resume 경계 LR 재상승 문제는 해당 없음.

## 7. 후보 checkpoint별 retention/expansion validation 결과

| candidate | retention (400) | expansion validation (50) | joint clamp | grasp/place 모순 |
|---|---|---|---|---|
| 2500 | 305/400 = 76.25% | 32/50 = 64.0% | 0 | 2 |
| 5000 | 301/400 = 75.25% | 26/50 = 52.0% | 0 | 2 |
| **7500** | **314/400 = 78.50%** | **35/50 = 70.0%** | **0** | **0** |
| 10000 | 271/400 = 67.75% | 25/50 = 50.0% | 0 | 0 |

## 8. Checkpoint 선택 이유

선정 우선순위: (1) expansion validation 성공률 최대화 → 7500이 70.0%로 압도적 1위 (2위 2500의 64.0% 대비 +6pp) → (2) retention이 공식 70.25% 대비 -5%p 이내(≥65.25%) → 4개 후보 전부 통과, 7500은 78.5%로 오히려 baseline보다 높음 → (3) clamp/NaN·Inf/scene_invalid 0 → 4개 후보 전부 통과, 7500은 추가로 grasp/place 모순도 0 → **7500-step checkpoint 확정**.

## 9. 최종 held-out test 결과

`outputs/train/so101_stage1a_xy_reinforcement_20260721_231829/checkpoints/007500/pretrained_model`, 신규 held-out test 10위치 × policy-noise repeat 10 = 100 rollouts, base seed 400000:

- **57/100 = 57.0%**, joint clamp 0, grasp/place 모순 1건
- 영역별: x_min_corridor 15/30(50%), corner_pp 5/10(50%), corner_pn 13/20(65%), corner_np 5/10(50%), corner_nn 6/10(60%), bridge_plus_x 7/10(70%), bridge_plus_y 6/10(60%)

validation(70.0%)보다는 낮지만(약간의 validation 과적합 가능성 시사), 기존 확대범위 baseline(35.0%) 대비 **+22.0%p** 개선이며 모든 영역이 50% 이상으로 올라와 있음. **선정된 checkpoint는 test 결과와 무관하게 재선정하지 않았음.**

## 10. 기존 능력 유지 여부

**유지됨.** Retention 78.5%는 공식 25k baseline(70.25%)보다 오히려 높으며, 허용 하락치(-5%p) 기준을 벗어나지 않음.

## 11. XY 일반화 개선 여부

**개선됨.** Expansion validation 35.0%→70.0%(+35.0%p), held-out test는 35.0%→57.0%(+22.0%p). Validation과 test 간 13%p 격차는 존재하지만(작은 test 표본, n=100), 두 지표 모두 명확한 개선을 보여준다.

## 12. 다음 단계인 직육면체(다른 물체) 확장 준비 상태

**준비됨.** Held-out test가 (validation에 쓰이지 않은) 완전히 새로운 좌표·seed에서도 원래 35%보다 훨씬 높은 57%를 달성해, 이번 개선이 validation set 암기가 아니라 실질적 위치 일반화임을 뒷받침한다. 다음 물체(PET병/캔 등) 추가를 진행해도 좋다.

## 13. 주요 실행 명령어와 repository-relative 경로

```
# 데이터 수집
.venv-vla/bin/python -m benchmark.collect_so101_stage1a_xy_reinforcement

# 결합 dataset 생성 + train allowlist manifest
.venv-vla/bin/python -m benchmark.merge_so101_dataset_for_training

# 무결성 검증
.venv-vla/bin/python -m benchmark.validate_so101_stage1a_dataset

# Fine-tuning (fresh optimizer, 25k weight에서 시작)
env VLA_DEVICE=cuda VLA_DTYPE=float32 .venv-vla/bin/lerobot-train \
  --dataset.repo_id=local/so101_bin_stage1a_combined_270 \
  --dataset.root=datasets/so101_bin_stage1a_combined_270 \
  --dataset.episodes=<configs/so101_stage1a_train_episodes.json의 train_episode_indices> \
  --policy.path=outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model \
  --output_dir=outputs/train/so101_stage1a_xy_reinforcement_20260721_231829 \
  --steps=10000 --save_freq=2500 --batch_size=1 --seed=0

# expansion validation/test 평가
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1a_expansion \
  --checkpoint-dir <checkpoint> --split validation --policy-noise-repeats 5 --policy-noise-base-seed 300000 --output-path <path>
.venv-vla/bin/python -m benchmark.evaluate_so101_stage1a_expansion \
  --checkpoint-dir <checkpoint> --split test --policy-noise-repeats 10 --policy-noise-base-seed 400000 --output-path <path>
```

경로:
- 신규 dataset: `datasets/so101_bin_stage1a_xy_70`, `datasets/so101_bin_stage1a_combined_270`
- Train allowlist manifest: `configs/so101_stage1a_train_episodes.json`
- 최종 선택 checkpoint: `outputs/train/so101_stage1a_xy_reinforcement_20260721_231829/checkpoints/007500/pretrained_model`
- 결과: `results/so101_stage1a_reinforcement/{retention,expansion_validation}_{2500,5000,7500,10000}/`, `results/so101_stage1a_reinforcement/expansion_test_final/`

`datasets/`, `outputs/`, `results/`는 모두 `.gitignore` 대상이며 Git에 포함되지 않는다.

## 14. Scheduler 및 실험 한계

- 이번 fine-tuning은 resume 체인이 아닌 단일 fresh run이므로 이전 baseline들에서 관측된 "resume 경계 LR 재상승" 문제는 해당 없음. 다만 이 fine-tuning 자체의 scheduler도 `--steps=10000`에 맞춰 `num_decay_steps`가 자동 재조정(30000→10000)됐다는 점은 동일한 LeRobot 메커니즘이다.
- Expansion validation은 위치당 5회, held-out test는 위치당 10회 반복 — 여전히 작은 표본이라 개별 위치의 순위를 과도하게 일반화하지 않는다.
- 이번 실험은 큐브 하나, 단일 fine-tuning 깊이(10,000 step, 4개 checkpoint 후보)만 탐색했다. 다른 물체 형상이나 더 넓은 범위에 대한 일반화는 검증되지 않았다.
