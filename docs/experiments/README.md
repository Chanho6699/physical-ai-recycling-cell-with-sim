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
  held-out test 66.7% (vs. 62.7% zero-shot). Superseded as the overall
  official checkpoint by Stage 1C below.
- [so101_expert_v2_orientation_summary.md](so101_expert_v2_orientation_summary.md) --
  Stage 1C pre-check: Orientation-aware Expert V2 (new, separate from
  V1) and a standalone yaw-grid evaluation (180 episodes). Result: 6.1%
  overall success (yaw=0/180 succeed at V1 parity, every other yaw
  0%) -- a confirmed kinematic limit of the 5-DOF arm (only
  `shoulder_pan` has a world-Z axis, and it's already fully committed
  to aiming at the target position), not an implementation bug.
  **Stage 1C dataset generation is not authorized on this expert.**
- [so101_stage1c_cylinder_feasibility_summary.md](so101_stage1c_cylinder_feasibility_summary.md) --
  Stage 1C pre-check: size-aware Expert V2.1 (new, separate from V1 and
  from the orientation-aware V2 experiment) generalizes V1's fixed
  per-object offsets to dimension-based formulas, verified to
  reproduce V1's cube/box waypoints exactly. A candidate upright
  cylinder (radius=0.02m, matching the validated cube/box scale) scored
  125/125 = 100% on both legacy_success and physical_success across 5
  position groups and 5 scene-yaw values (rotational-symmetry check).
  Stage 1B checkpoint zero-shot on this cylinder: 46/75 = 61.3%
  (known/interior 76.0%, expanded edge 50.0%, corner 56.7%), a clean
  policy-generalization reference point (Expert itself was clean, so
  this gap is not scene/Expert-confounded). **Stage 1C cylinder dataset
  generation is authorized to proceed (Expert threshold met); the
  zero-shot number is the pre-training baseline, not a go/no-go gate.**
- [so101_stage1c_cylinder_summary.md](so101_stage1c_cylinder_summary.md) --
  Stage 1C official run: 100-episode upright-cylinder dataset (Expert
  V2.1, 100/100 saved, 0 discarded), 180-episode rehearsal fine-tune
  (cube 55 + box 55 + cylinder 70) from the Stage 1B 7500-step
  checkpoint. Selected checkpoint (10000-step, only candidate meeting
  all three validation thresholds): cube retention 87.0% (held-out
  88.0%, no forgetting vs. Stage 1B's 82.0%), box validation 77.3%
  (held-out 69.3%, vs. Stage 1B's 76.0%/66.7%), cylinder validation
  74.7% (held-out 74.0%, +13.4pp over the 61.3% zero-shot baseline).
  All grasps remain constraint-based simulated pick-and-place (EE-
  object distance + gripper angle trigger), not contact-force verified.
  Current official checkpoint overall.
