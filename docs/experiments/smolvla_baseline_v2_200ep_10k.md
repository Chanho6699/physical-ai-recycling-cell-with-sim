# SmolVLA Baseline v2 -- 200 episodes / 10,000 steps

## 1. 실험 목적

기존 100-episode / 5000-step baseline과 비교해, 동일한 데이터 생성
조건에서 dataset 규모를 200 episodes로, 학습을 10,000 steps로
확장했을 때 SO-101 SmolVLA closed-loop pick-and-place 성공률이 실질적으로
향상되는지 확인한다.

## 2. 실행 구성

- pretrained base: `lerobot/smolvla_base`
- collection seed: 0~199
- split seed: 42
- train/validation: 160/40
- collection mode: `fixed_bin_object_xy`
- save frequency: 2500
- fresh training (기존 5000-step checkpoint에서 resume하지 않음)

## 3. 재현 명령

```
.venv-vla/bin/python -m benchmark.run_so101_smolvla_pipeline \
  --stage all \
  --dataset-name so101_bin_main_200 \
  --episodes 200 \
  --collection-mode fixed_bin_object_xy \
  --training-steps 10000 \
  --save-freq 2500
```

## 4. Dataset 검증

- 200/200 정상 저장, discard 0
- 총 13,606 frames
- episode length: min 68, max 69, mean 68.03
- NaN/Inf 없음
- seed 중복 없음

## 5. Checkpoint별 결과

| checkpoint | arm MAE (rad) | gripper accuracy | grasp/place | 성공률 | min distance mean | median | std | ≤3cm | ≤5cm |
|---|---|---|---|---|---|---|---|---|---|
| 5000  | 0.0206 | 97.98% | 8/40  | 20.0% | 0.0356 | 0.0360 | 0.0064 | 25.0% | 97.5% |
| 7500  | 0.0129 | 98.09% | 7/40  | 17.5% | 0.0377 | 0.0362 | 0.0075 | 17.5% | 92.5% |
| 10000 | 0.0114 | 98.42% | 21/40 | 52.5% | 0.0312 | 0.0307 | 0.0043 | 42.5% | 100%  |

(grasp count와 place count는 이 실험에서 동일하다 -- grasp이 형성된 경우
전부 place까지 성공했다.)

## 6. 안정성

- 3 checkpoint x 40 seed = 120 rollout 전체에서 NaN/Inf **0건**
- joint limit clamp **0건**
- 90-step 정상 완료 120/120

## 7. Checkpoint 선택 근거

Primary metric: `place_success_rate`. 10000-step checkpoint가 52.5%로
세 checkpoint 중 최고치를 기록해 선택했다. 5000-step(20.0%)과
7500-step(17.5%)은 서로 비슷하거나 7500이 더 낮았으므로, "마지막
checkpoint라서" 선택한 것이 아니라 명시된 순위 기준(1순위
place_success_rate)에 따라 결정된 결과다.

## 8. 기존 baseline 비교

| 항목 | 기존 (100ep / 5000-step) | 신규 (200ep / 10000-step) |
|---|---|---|
| place/grasp 성공률 | 40.0% | 52.5% |
| arm MAE | 0.0164 rad | 0.0114 rad |
| gripper accuracy | 97.0% | 98.42% |

place/grasp 성공률 **+12.5%p** 향상.

## 9. Baseline 확정

- 이름: **SmolVLA Baseline v2**
- dataset: 200 episodes
- selected checkpoint: 10000-step
- 현재 기본 실험 baseline으로 사용한다.

## 10. 로컬 산출물 경로

다음 경로는 로컬 참고용이며 **Git에 포함되지 않는다** (`datasets/`,
`outputs/`, `results/`는 모두 gitignore 대상).

- dataset: `datasets/so101_bin_main_200`
- checkpoint: `outputs/train/all_20260720_114358/checkpoints/010000/pretrained_model`
- result: `results/so101_pipeline_runs/all_20260720_114358/`

## 11. 한계와 다음 단계

- 검증 환경은 여전히 제한적이다(단일 object, 단일 bin 위치, 고정
  카메라 시점).
- place_success_rate 52.5%이므로 이 checkpoint를 일반화된 manipulation
  policy라고 부르지 않는다.
- 다음 단계에서는 한 번에 하나의 환경 변수만 제한적으로 확대한다.
- ROS2/시스템 통합 작업과 병행 진행한다.
