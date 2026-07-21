# SO-101 SmolVLA Cube Baseline and XY Generalization Summary

## 1. 목표

- SO-101 PyBullet 환경에서 큐브 Pick-and-Place baseline 구축
- 재현 가능한 closed-loop 평가 프로토콜 확립
- 위치(XY) 일반화 성능 확인

## 2. 데이터셋

- Dataset: `datasets/so101_bin_main_200`
- 총 200 episodes (train 160 / validation 40)
- 수집 환경 seed: 0~199, split seed: 42
- object XY offset: ±0.015m (`FIXED_BIN_OBJECT_X_RANGE`/`FIXED_BIN_OBJECT_Y_RANGE`)
- 고정 조건: cube 형상, object yaw, bin 위치, 시작 joint pose, 물체 질량/마찰

## 3. 학습 경로

10,000 → 15,000 → 20,000 → 25,000 step까지 **단계적 resume 학습**으로 진행.

| checkpoint | 경로 | closed-loop 성공률 | arm MAE | gripper accuracy | joint clamp | NaN/Inf |
|---|---|---|---|---|---|---|
| 10,000 | `outputs/train/all_20260720_114358/checkpoints/010000/pretrained_model` | 210/400 = 52.50% | 0.0114 | 98.42% | 0 | 없음 |
| 15,000 | `outputs/train/all_20260720_114358_resume_15000/checkpoints/015000/pretrained_model` | 232/400 = 58.00% | 0.0106 | 98.42% | 0 | 없음 |
| 20,000 | `outputs/train/all_20260720_114358_resume_20000/checkpoints/020000/pretrained_model` | 258/400 = 64.50% | 0.0101 | 98.64% | 0 | 없음 |
| 25,000 | `outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model` | 281/400 = 70.25% | 0.0096 | 98.86% | 0 | 없음 |

closed-loop 성공률은 40 environment seeds × 10 independent policy-noise repeats = 400 rollouts 기준(5절 참조), 4개 checkpoint 모두 동일 프로토콜.

**중요**: 매 resume(10k→15k, 15k→20k, 20k→25k)마다 LeRobot의 `CosineDecayWithWarmupSchedulerConfig`가 그때그때의 새 목표 총 step 수에 맞춰 **자동 재조정(auto-rescale)** 됨 (`num_training_steps < num_decay_steps=30000`일 때 `actual_decay_steps = num_training_steps`로 재계산). 그 결과 resume 경계마다 learning rate가 아래로 수렴했다가 다시 위로 튀는 패턴이 반복됐다(예: 20k 학습 종료 시점 LR 2.5e-06 → 25k 학습 시작 시점 같은 step에서 LR 1.2e-05로 재상승). 따라서 이 4개 checkpoint는 **처음부터 하나의 연속된 25,000-step LR schedule로 학습한 결과와는 다르다.**

## 4. 공식 최고 Baseline

- Checkpoint: `outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model`
- Closed-loop: 281/400 = **70.25%**
- Arm MAE: **0.0096**
- Gripper accuracy: **98.86%**
- joint clamp: **0**
- NaN/Inf: **0**

## 5. 평가 재현성 개선

기존에는 `SmolVLAPolicy`의 flow-matching 행동 생성(`sample_actions(noise=None)`)이 매 프로세스 실행마다 시딩되지 않은 전역 torch RNG로 noise를 새로 뽑았기 때문에, **동일 checkpoint·동일 environment seed로 재실행해도 grasp/place 성공 여부가 실행마다 달라지는 문제**가 있었다. 처음에는 noise seed 하나를 40개 environment seed 전체가 공유하는 방식으로 고쳤으나, 이는 그룹 내 평균화가 전혀 없어 noise seed 하나의 우연에 결과가 좌우되는(성공률이 0%~100%까지 극단적으로 쏠리는) 잘못된 결론을 낳았다.

이를 바로잡아 각 `(environment_seed, repeat_id)` 조합마다 독립적이고 재현 가능한 policy noise seed를 사용하도록 수정했다. Derived seed 규칙은 `100000 + environment_seed * 1000 + repeat_id`이며, 공식 평가는 40 environment seeds × 10 independent policy-noise repeats = **400 rollouts**로 고정됐다. 이 방식으로 재실행 시 첫 action chunk, 전체 action sequence, 최종 grasp/place 결과가 별도 프로세스 간에 완전히 동일함을 확인했다.

## 6. Pre-grasp 실패 분석

- grasp이 성공한 경우 place까지 항상 성공(grasp/place가 항상 함께 움직임, 이번 프로젝트 전 구간에서 관측된 일관된 패턴)
- 주요 병목은 transport/place 단계가 아니라 **pre-grasp 단계**
- step별 진단 로그에서, 실패한 rollout 상당수가 gripper close 명령이 팔의 최근접 시점보다 한 스텝 늦게 발생하는 패턴을 보임
- 다만 이는 진단 표본(diagnostic log가 활성화된 일부 rollout)에서 관측된 패턴이며, **최종 모델 전체의 모든 실패를 이 하나의 원인으로 단정하지 않는다**

## 7. XY 외삽 평가

25,000-step checkpoint 기준, 학습 없이 순수 평가만 수행.

- 기존 범위: offset ±0.015m
- 확대 범위: offset ±0.01875m (25% 확대)
- 총 25개 고정 위치 × 5 repeats = **125 rollouts**
- 기존 범위 경계/모서리: 26/40 = **65.0%**
- 확대 범위 경계/모서리: 14/40 = **35.0%**
- 차이: **-30.0%p**
- scene_invalid / joint clamp / NaN·Inf: 전부 **0**

주요 패턴:
- 단일 축 이동보다 모서리(X+Y 복합 오프셋)에서 성능 하락이 더 큼
- -X 방향에서 반경이 커질수록 성공률이 단조 하락하는 패턴 관측
- 위치별 반복이 5회로 적어, 특정 한 좌표의 순위를 과도하게 일반화하지 않는다
- scene_invalid/clamp/NaN·Inf가 전혀 없어, **물리적 도달 한계가 아니라 학습 분포에 대한 위치 일반화 문제**로 판단

## 8. 다음 결정

- 동일 200-episode 큐브 dataset에서 학습 step만 늘리는 실험은 **25,000-step에서 종료**
- PET병/캔 등 새로운 물체는 바로 추가하지 않음
- **Stage 1A**: object XY 범위만 표적 보강(yaw/bin 위치/시작 joint pose/물체 형상·물리 속성은 고정)
- 추천 신규 데이터 규모: 약 70 episodes (train 50 / validation 10 / held-out test 10), 취약했던 -X corridor와 모서리 영역 중심으로 배분
- 신규 학습은 25,000-step model weights에서 시작하되 optimizer/scheduler는 새로 초기화(resume 아님)
- 평가는 기존 40 seed 기준 **retention**과 신규 XY 위치 기준 **expansion**을 분리해서 측정

## 9. 재현 경로

- Rollout 평가 스크립트: `benchmark/so101_smolvla_rollout.py` (policy noise seed 옵션: `--policy-noise-seed`, `--policy-noise-repeats`, `--policy-noise-base-seed`)
- XY 외삽 평가 스크립트: `benchmark/evaluate_so101_xy_extrapolation.py`
- Baseline 결과 경로:
  - `results/so101_pipeline_runs/all_20260720_114358/` (10,000-step 원본 실행)
  - `results/so101_reproducibility_sweep/baseline_v2_15000_offline_eval/`, `baseline_v2_15000_independent_noise/`
  - `results/so101_reproducibility_sweep/baseline_v2_20000_offline_eval/`, `baseline_v2_20000_independent_noise/`
  - `results/so101_reproducibility_sweep/baseline_v2_25000_offline_eval/`, `baseline_v2_25000_independent_noise/`
  - `results/so101_xy_extrapolation/xy_extrapolation_results.json`
- Checkpoint 경로:
  - `outputs/train/all_20260720_114358/checkpoints/010000/pretrained_model`
  - `outputs/train/all_20260720_114358_resume_15000/checkpoints/015000/pretrained_model`
  - `outputs/train/all_20260720_114358_resume_20000/checkpoints/020000/pretrained_model`
  - `outputs/train/all_20260720_114358_resume_25000/checkpoints/025000/pretrained_model`

`datasets/`, `outputs/`, `results/`는 모두 `.gitignore` 대상이며 Git에 포함되지 않는다. 위 경로는 로컬 재현 참고용이다.
