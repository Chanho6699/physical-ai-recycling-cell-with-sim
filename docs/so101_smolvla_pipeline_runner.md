# SO-101 SmolVLA pipeline runner

Thin subprocess orchestrator over already-validated scripts (does not
reimplement collection/training/evaluation/rollout logic) --
`benchmark/run_so101_smolvla_pipeline.py`. See that file's own module
docstring for stage/argument details.

## Existing data, train + evaluate

```
.venv-vla/bin/python benchmark/run_so101_smolvla_pipeline.py \
  --stage train-eval \
  --dataset-path datasets/so101_bin_fixed_pilot_30 \
  --training-steps 5000 \
  --save-freq 2500 \
  --rollout-seeds 0 3 7
```

## Full run: collect new data, train, evaluate

`--collection-mode` must be an actual mode `benchmark/collect_so101_bin_dataset.py`
supports -- currently `coupled_small` or `fixed_bin_object_xy` (see
`benchmark/benchmark_so101_bin_diagnostic.py`'s own
`RANDOMIZATION_MODE_*` constants; there is no separate "final 5-box"
mode in this codebase yet, so this example uses the real
`fixed_bin_object_xy` mode with a descriptive dataset name):

```
.venv-vla/bin/python benchmark/run_so101_smolvla_pipeline.py \
  --stage all \
  --dataset-name so101_bin_main_100 \
  --episodes 100 \
  --collection-mode fixed_bin_object_xy \
  --training-steps 5000 \
  --save-freq 2500 \
  --rollout-seeds 0 3 7
```

Add `--dry-run` to either command to print the exact subprocess
commands without executing anything.
