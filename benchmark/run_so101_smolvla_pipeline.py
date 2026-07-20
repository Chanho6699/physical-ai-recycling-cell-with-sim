"""Thin SO-101 SmolVLA pipeline runner (see this task's chat report,
"검증된 기존 명령들을 한 번에 순차 실행할 수 있는 얇은 ... runner").

NOT a new training/experiment framework -- every actual stage is a
subprocess call to an ALREADY-VALIDATED existing script/CLI:
  - collect  -> benchmark.collect_so101_bin_dataset (unmodified)
  - validate -> benchmark.collect_so101_episode.verify_dataset() (unmodified,
                invoked via a one-line subprocess, not reimplemented)
  - train    -> lerobot-train CLI (same invocation pattern this project's
                own benchmark/train_v3.py already established)
  - offline eval -> benchmark.so101_smolvla_checkpoint_inference_eval
                (unmodified evaluation logic; this task only added CLI
                path overrides so it can be pointed at an arbitrary
                checkpoint/dataset/split)
  - rollout  -> benchmark.so101_smolvla_rollout, specifically its
                CORRECTED action-queue path (policy.reset() once per
                episode, then select_action() -- see that file's own
                predict_action_in_rollout() docstring). This runner
                NEVER calls the deprecated per-step-reset
                so101_smolvla_checkpoint_inference_eval.predict_action()
                for rollout -- only run_one_rollout() (which now always
                uses the corrected queue path).

This file contains ZERO training/inference/rollout logic of its own --
only: argument parsing, path safety checks, subprocess invocation,
log capture, and JSON summary writing.

Stages:
  --stage collect     : collect -> validate
  --stage train-eval   : (existing dataset OR --dataset-path) -> split
                         -> train -> offline eval -> rollout
  --stage eval          : existing checkpoint (--resume-checkpoint) ->
                         offline eval -> rollout
  --stage all            : collect -> validate -> split -> train ->
                         offline eval -> rollout

Run:
  .venv-vla/bin/python -m benchmark.run_so101_smolvla_pipeline \\
    --stage train-eval --dataset-path datasets/so101_bin_fixed_pilot_30 \\
    --training-steps 200 --save-freq 100 --rollout-seeds 0 3 7

  .venv-vla/bin/python -m benchmark.run_so101_smolvla_pipeline \\
    --stage all --dry-run --dataset-name so101_smoke_test --episodes 5 \\
    --collection-mode fixed_bin_object_xy --training-steps 10 --save-freq 10
"""

import argparse
import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv-vla" / "bin" / "python"
LEROBOT_TRAIN = PROJECT_ROOT / ".venv-vla" / "bin" / "lerobot-train"
RUNS_ROOT = PROJECT_ROOT / "results" / "so101_pipeline_runs"

DEFAULT_TASK_TEXT = "Pick up the object and place it in the bin."
DEFAULT_RENAME_MAP = '{"observation.images.front": "observation.images.camera1"}'  # lerobot/smolvla_base's own pretrained config declares camera1/camera2/camera3; see benchmark/so101_smolvla_checkpoint_inference_eval.py's own config.json this task's earlier chat report already established this mapping from.
SPLIT_SEED = 42  # matches this project's own established SO-101 sanity-training split (results/so101_smolvla_sanity_training/split.json)
TRAIN_FRACTION = 0.8  # 24/30 in the already-validated sanity run


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class StageFailure(Exception):
    def __init__(self, stage: str, message: str, command: list = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.command = command


class Runner:
    def __init__(self, args, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.commands = []
        self.artifacts = {}

    def run(self, stage_name: str, cmd: list, log_name: str) -> int:
        """Every actual command execution funnels through here -- ONE
        place that prints, logs, and (in --dry-run) skips execution."""
        printable = " ".join(str(c) for c in cmd)
        print(f"\n[{stage_name}] {printable}")
        self.commands.append({"stage": stage_name, "command": [str(c) for c in cmd]})

        if self.args.dry_run:
            return 0

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / log_name
        with open(log_path, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
        return result.returncode

    def fail(self, stage: str, message: str, cmd: list = None):
        full_message = f"STAGE FAILED: {stage}\nREASON: {message}"
        if cmd:
            full_message += f"\nCOMMAND: {' '.join(str(c) for c in cmd)}"
        print(f"\n{'=' * 60}\n{full_message}\n{'=' * 60}")
        raise StageFailure(stage, message, cmd)


def compute_split(total_episodes: int, seed: int = SPLIT_SEED, train_fraction: float = TRAIN_FRACTION) -> dict:
    """Same method already used for datasets/so101_bin_fixed_pilot_30's
    own results/so101_smolvla_sanity_training/split.json (seed=42,
    random.Random(seed).shuffle then a fixed train/validation cut) --
    generalized here to an arbitrary episode count so this runner does
    not hardcode 24/6. Deterministic in seed+total_episodes, so
    re-running this function for the SAME dataset always reproduces
    the SAME split without needing a separate "reuse" flag."""
    episodes = list(range(total_episodes))
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_train = round(total_episodes * train_fraction)
    train_episodes = sorted(shuffled[:n_train])
    validation_episodes = sorted(shuffled[n_train:])
    return {
        "split_seed": seed, "total_episodes": total_episodes, "train_fraction": train_fraction,
        "train_episode_count": len(train_episodes), "validation_episode_count": len(validation_episodes),
        "train_episodes": train_episodes, "validation_episodes": validation_episodes,
    }


def read_dataset_episode_count(dataset_root: Path) -> int:
    summary_path = dataset_root / "collection_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())["saved_episode_count"]
    # Fallback: count unique episode_index directly from the parquet (works even for a dataset this runner did not itself collect).
    import pandas as pd
    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    return int(frames["episode_index"].nunique())


def stage_collect(runner: Runner, args, dataset_root: Path, repo_id: str) -> None:
    if dataset_root.exists():
        runner.fail("collect", f"Refusing to overwrite existing dataset path: {dataset_root}")

    cmd = [
        str(VENV_PYTHON), "-m", "benchmark.collect_so101_bin_dataset",
        "--dataset-root", str(dataset_root), "--repo-id", repo_id,
        "--num-episodes", str(args.episodes), "--mode", args.collection_mode,
    ]
    rc = runner.run("collect", cmd, "collect.log")
    if rc != 0:
        runner.fail("collect", f"collect_so101_bin_dataset.py exited with code {rc} -- collection aborted immediately, per its own contract-check/discard-policy gating.", cmd)
    runner.artifacts["dataset_root"] = str(dataset_root)
    runner.artifacts["dataset_repo_id"] = repo_id


def stage_validate(runner: Runner, dataset_root: Path) -> None:
    # Reuses benchmark.collect_so101_episode's own verify_dataset() --
    # not reimplemented -- via a one-line subprocess so this stage gets
    # its OWN log file and its OWN pass/fail exit code, distinct from
    # collect's (collect_so101_bin_dataset.py already calls this same
    # function internally too, but only as a print-out, not a gate a
    # caller can act on independently).
    snippet = (
        "import json, sys; from pathlib import Path; "
        "from benchmark.collect_so101_episode import verify_dataset; "
        f"r = verify_dataset(Path({str(dataset_root)!r})); "
        "print(json.dumps(r, indent=2, default=str)); "
        "sys.exit(1 if (r['state_has_nan_or_inf'] or r['action_has_nan_or_inf']) else 0)"
    )
    cmd = [str(VENV_PYTHON), "-c", snippet]
    rc = runner.run("validate", cmd, "validate.log")
    if rc != 0:
        runner.fail("validate", f"Dataset contract check (verify_dataset) failed with code {rc} -- refusing to train on this dataset.", cmd)


def stage_train(runner: Runner, args, dataset_root: Path, repo_id: str, output_dir: Path) -> Path:
    if args.resume_checkpoint:
        resume_from = resolve(args.resume_checkpoint)
        config_path = resume_from / "pretrained_model" / "train_config.json"
        if not config_path.exists():
            runner.fail("train", f"--resume-checkpoint given but no train_config.json found at {config_path}")
        if not output_dir.exists():
            runner.fail("train", f"Resuming requires an EXISTING --output-dir (the same run being continued), but {output_dir} does not exist.")
        cmd = [
            "env", "VLA_DEVICE=cuda", "VLA_DTYPE=float32", str(LEROBOT_TRAIN),
            f"--config_path={config_path}", "--resume=true", f"--output_dir={output_dir}",
        ]
    else:
        if output_dir.exists():
            runner.fail("train", f"Refusing to overwrite existing training output dir: {output_dir}")
        cmd = [
            "env", "VLA_DEVICE=cuda", "VLA_DTYPE=float32", str(LEROBOT_TRAIN),
            f"--dataset.repo_id={repo_id}", f"--dataset.root={dataset_root}",
            f"--dataset.episodes={runner.artifacts['train_episodes']}",
            f"--policy.path={args.pretrained_model}", "--policy.device=cuda",
            f"--rename_map={DEFAULT_RENAME_MAP}",
            f"--output_dir={output_dir}", f"--job_name={output_dir.name}",
        ]
    cmd += [
        f"--steps={args.training_steps}", f"--save_freq={args.save_freq}", "--log_freq=10",
        "--batch_size=1", "--num_workers=2", "--seed=0",
        "--policy.push_to_hub=false", "--wandb.enable=false",
    ]

    rc = runner.run("train", cmd, "train.log")
    if rc != 0:
        runner.fail("train", f"lerobot-train exited with code {rc}.", cmd)

    last_checkpoint = output_dir / "checkpoints" / f"{args.training_steps:06d}"
    if not args.dry_run and not (last_checkpoint / "pretrained_model" / "model.safetensors").exists():
        runner.fail("train", f"Training reported success but no checkpoint was actually written at {last_checkpoint}/pretrained_model/model.safetensors -- refusing to evaluate.", cmd)

    runner.artifacts["output_dir"] = str(output_dir)
    runner.artifacts["final_checkpoint_dir"] = str(last_checkpoint / "pretrained_model")
    return last_checkpoint / "pretrained_model"


def stage_offline_eval(runner: Runner, checkpoint_dir: Path, dataset_root: Path, split_path: Path) -> Path:
    validation_metrics_path = runner.run_dir / "validation_metrics.json"
    offline_predictions_path = runner.run_dir / "offline_predictions.json"
    cmd = [
        str(VENV_PYTHON), "-m", "benchmark.so101_smolvla_checkpoint_inference_eval",
        "--checkpoint-dir", str(checkpoint_dir), "--dataset-root", str(dataset_root),
        "--split-path", str(split_path),
        "--validation-metrics-path", str(validation_metrics_path),
        "--offline-predictions-path", str(offline_predictions_path),
    ]
    rc = runner.run("offline_eval", cmd, "offline_eval.log")
    if rc != 0:
        runner.fail("offline_eval", f"Offline evaluation exited with code {rc} (non-finite NaN/Inf output, or a contract check failed) -- refusing to run rollout.", cmd)
    runner.artifacts["validation_metrics_path"] = str(validation_metrics_path)
    runner.artifacts["offline_predictions_path"] = str(offline_predictions_path)
    return offline_predictions_path


def stage_rollout(runner: Runner, args, checkpoint_dir: Path, split_path: Path) -> None:
    rollout_results_path = runner.run_dir / "rollout_results.json"
    cmd = [
        str(VENV_PYTHON), "-m", "benchmark.so101_smolvla_rollout",
        "--checkpoint-dir", str(checkpoint_dir), "--split-path", str(split_path),
        "--max-rollout-steps", str(args.max_rollout_steps), "--output-path", str(rollout_results_path),
    ]
    if args.rollout_seeds:
        cmd += ["--rollout-seeds"] + [str(s) for s in args.rollout_seeds]
    rc = runner.run("rollout", cmd, "rollout.log")
    if rc != 0:
        runner.fail("rollout", f"Rollout exited with code {rc}.", cmd)
    runner.artifacts["rollout_results_path"] = str(rollout_results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["all", "train-eval", "eval", "collect"])
    parser.add_argument("--dataset-path", type=str, default=None, help="existing dataset root (train-eval/eval); ignored for collect/all")
    parser.add_argument("--dataset-name", type=str, default=None, help="new dataset dir name under datasets/ (collect/all)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--collection-mode", type=str, default="fixed_bin_object_xy", choices=["coupled_small", "fixed_bin_object_xy"])
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--pretrained-model", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default=None, help="training output dir under outputs/train/ (default: outputs/train/<run_id>)")
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=None, help="default: first 3 validation-split episodes")
    parser.add_argument("--max-rollout-steps", type=int, default=90)
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="existing outputs/train/.../checkpoints/NNNNNN dir to resume from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.stage}_{timestamp}"
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing pipeline run dir: {run_dir}")

    runner = Runner(args, run_dir)
    pipeline_config = {"run_id": run_id, "stage": args.stage, "args": vars(args), "timestamp": timestamp}

    failure = None
    try:
        if args.stage in ("collect", "all"):
            if not args.dataset_name:
                raise StageFailure("collect", "--dataset-name is required for --stage collect/all")
            dataset_root = PROJECT_ROOT / "datasets" / args.dataset_name
            repo_id = f"local/{args.dataset_name}"
            stage_collect(runner, args, dataset_root, repo_id)
            stage_validate(runner, dataset_root)
        else:
            if not args.dataset_path:
                raise StageFailure(args.stage, "--dataset-path is required for --stage train-eval/eval")
            dataset_root = resolve(args.dataset_path)
            if not dataset_root.exists():
                raise StageFailure(args.stage, f"--dataset-path does not exist: {dataset_root}")
            repo_id = f"local/{dataset_root.name}"
            runner.artifacts["dataset_root"] = str(dataset_root)
            runner.artifacts["dataset_repo_id"] = repo_id

        if args.stage in ("collect",):
            pass  # collect-only: nothing further
        elif args.stage == "eval":
            if not args.resume_checkpoint:
                raise StageFailure("eval", "--resume-checkpoint is required for --stage eval")
            checkpoint_root = resolve(args.resume_checkpoint)
            checkpoint_dir = checkpoint_root / "pretrained_model"
            total_episodes = args.episodes if args.dry_run else read_dataset_episode_count(dataset_root)
            split = compute_split(total_episodes)
            split_path = run_dir / "split.json"
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                split_path.write_text(json.dumps(split, indent=2))
            runner.artifacts.update({"checkpoint_dir": str(checkpoint_dir), "split_path": str(split_path)})
            stage_offline_eval(runner, checkpoint_dir, dataset_root, split_path)
            stage_rollout(runner, args, checkpoint_dir, split_path)
        else:  # train-eval or all
            total_episodes = args.episodes if args.dry_run else read_dataset_episode_count(dataset_root)
            split = compute_split(total_episodes)
            split_path = run_dir / "split.json"
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                split_path.write_text(json.dumps(split, indent=2))
            runner.artifacts["split_path"] = str(split_path)
            runner.artifacts["train_episodes"] = split["train_episodes"]

            output_dir = resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "train" / run_id
            checkpoint_dir = stage_train(runner, args, dataset_root, repo_id, output_dir)
            stage_offline_eval(runner, checkpoint_dir, dataset_root, split_path)
            stage_rollout(runner, args, checkpoint_dir, split_path)

    except StageFailure as exc:
        failure = {"stage": exc.stage, "message": exc.message, "command": exc.command}

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2, default=str)
    with open(run_dir / "commands.json", "w", encoding="utf-8") as f:
        json.dump(runner.commands, f, indent=2, default=str)
    with open(run_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(runner.artifacts, f, indent=2, default=str)

    summary = {
        "run_id": run_id, "stage": args.stage, "dry_run": args.dry_run,
        "succeeded": failure is None, "failure": failure, "artifacts": runner.artifacts,
    }
    with open(run_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Pipeline run {'FAILED' if failure else 'SUCCEEDED'} ({run_id}) ===")
    if failure:
        print(f"failed_stage: {failure['stage']}")
        print(f"reason: {failure['message']}")
    print(f"Run dir: {run_dir}")

    if failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
