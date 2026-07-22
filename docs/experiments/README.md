# Experiments

Dated/versioned records of individual training or evaluation runs --
each file documents one experiment's configuration, results, and
reproduction command. Dataset/checkpoint/result files themselves are
not stored here (they live under `datasets/`, `outputs/`, `results/`,
all gitignored); these documents record the paths and numbers only.

- [smolvla_baseline_v2_200ep_10k.md](smolvla_baseline_v2_200ep_10k.md) --
  SmolVLA Baseline v2: 200-episode dataset, 10,000-step fresh
  fine-tune from `lerobot/smolvla_base`, 52.5% closed-loop
  place-success rate on 40 validation seeds. Early project baseline
  (superseded by the step-scaling/XY summaries below).
- [so101_cube_baseline_and_xy_expansion_summary.md](so101_cube_baseline_and_xy_expansion_summary.md) --
  10k->15k->20k->25k step-scaling results (25k official: 70.25%
  closed-loop, reproducible independent-noise eval protocol), plus the
  XY-extrapolation evaluation that motivated Stage 1A (65.0% in-range
  vs 35.0% at 25%-expanded range).
- [so101_stage1a_xy_reinforcement_summary.md](so101_stage1a_xy_reinforcement_summary.md) --
  Stage 1A object-XY targeted reinforcement: 70 new boundary/corner
  episodes, fresh fine-tune from the 25k checkpoint. Selected
  checkpoint (7500-step): cube retention 78.5%, expanded-XY held-out
  test 57.0% (vs. 35.0% before). Current official checkpoint for cube
  pick-and-place.
- [so101_stage1b_rectangular_box_summary.md](so101_stage1b_rectangular_box_summary.md) --
  Stage 1B rectangular-box shape generalization (rehearsal fine-tune:
  90 cube + 70 box episodes from the Stage 1A checkpoint). Selected
  checkpoint (7500-step): cube retention 82.0% (no forgetting), box
  held-out test 66.7% (vs. 62.7% zero-shot). Current official
  checkpoint overall.
