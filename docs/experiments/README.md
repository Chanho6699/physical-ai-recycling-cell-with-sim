# Experiments

Dated/versioned records of individual training or evaluation runs --
each file documents one experiment's configuration, results, and
reproduction command. Dataset/checkpoint/result files themselves are
not stored here (they live under `datasets/`, `outputs/`, `results/`,
all gitignored); these documents record the paths and numbers only.

- [smolvla_baseline_v2_200ep_10k.md](smolvla_baseline_v2_200ep_10k.md) --
  SmolVLA Baseline v2: 200-episode dataset, 10,000-step fresh
  fine-tune from `lerobot/smolvla_base`, 52.5% closed-loop
  place-success rate on 40 validation seeds. Current project baseline.
