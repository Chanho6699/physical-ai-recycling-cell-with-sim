"""V3 End-to-End Training Pipeline -- Training Launcher (see this task's
chat report). Wraps the EXACT, already-verified lerobot-train fresh-start
and --resume command patterns this project used for
outputs/train/smolvla_recycling_v2_train160 (same optimizer/scheduler/
batch_size/seed defaults, same --policy.push_to_hub=false fix) -- no new
training code, no new hyperparameters.

Pipeline this script drives:
  1. benchmark.validate_v3_dataset.validate() -- BLOCKS (raises
     SystemExit(1)) if the dataset doesn't clear the same thresholds
     smoke50 was judged against.
  2. lerobot-train in --eval-every-sized increments (fresh start for the
     first block, --resume=true for every subsequent block -- identical
     mechanics to this project's v2-2000->4000 resume run).
  3. After each increment's checkpoint is saved, starts a vla_server
     with it, runs benchmark.evaluate_v3_checkpoint's rollout evaluation
     (40-episode fixed-center benchmark, unchanged), appends one row to
     the tracking CSV, stops the server.

Run:
  .venv-vla/bin/python -m benchmark.train_v3 \\
    --dataset-root datasets/recycling_v3_dataset --dataset-repo-id local/recycling_cell_v3_dataset \\
    --output-dir outputs/train/smolvla_recycling_v3 --steps 4000 --eval-every 1000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

from benchmark.evaluate_v3_checkpoint import append_csv_row, load_suite, run_evaluation
from benchmark.validate_v3_dataset import validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9000
SERVER_STARTUP_WAIT_S = 8


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_training_block(dataset_root: str, dataset_repo_id: str, output_dir: Path, target_step: int, resume_from: Path = None) -> None:
    cmd = [
        "env", "VLA_DEVICE=cuda", "VLA_DTYPE=float32",
        str(PROJECT_ROOT / ".venv-vla" / "bin" / "lerobot-train"),
    ]
    if resume_from is None:
        cmd += [
            f"--dataset.repo_id={dataset_repo_id}", f"--dataset.root={dataset_root}",
            "--policy.path=HuggingFaceVLA/smolvla_libero", "--policy.device=cuda",
            f"--output_dir={output_dir}", f"--job_name={output_dir.name}",
        ]
    else:
        cmd += [
            f"--config_path={resume_from / 'pretrained_model' / 'train_config.json'}",
            "--resume=true", f"--output_dir={output_dir}",
        ]
    cmd += [
        f"--steps={target_step}", f"--save_freq={target_step if resume_from is None else target_step}",
        "--log_freq=50", "--batch_size=1", "--num_workers=2", "--seed=0",
        "--policy.push_to_hub=false", "--wandb.enable=false",
    ]
    print(f"=== Training block -> step {target_step} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def start_server_and_load(checkpoint_path: Path) -> subprocess.Popen:
    env = {
        "VLA_MODEL_FAMILY": "smolvla", "VLA_MODEL_ID_OR_PATH": str(checkpoint_path),
        "VLA_LOCAL_FILES_ONLY": "1", "VLA_DEVICE": "cuda", "VLA_DTYPE": "float32",
    }
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / ".venv-vla" / "bin" / "uvicorn"), "vla_server.generic_vla_server:app",
         "--host", SERVER_HOST, "--port", str(SERVER_PORT)],
        cwd=str(PROJECT_ROOT), env=full_env,
    )
    time.sleep(SERVER_STARTUP_WAIT_S)
    resp = requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/load_model", timeout=120)
    resp.raise_for_status()
    health = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=30).json()
    if health.get("model_status") != "loaded" or not (health.get("compatibility") or {}).get("passed"):
        proc.terminate()
        raise RuntimeError(f"Checkpoint failed to load/pass compatibility: {health}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    subprocess.run(["pkill", "-f", "generic_vla_server"], check=False)
    time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps (final ceiling).")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--csv", type=str, default="results/v3_pipeline/rollout_eval.csv")
    parser.add_argument("--suite-file", type=str, default=None, help="Frozen eval suite JSON -- see benchmark/evaluate_v3_checkpoint.py --save-suite-to.")
    parser.add_argument("--skip-validation", action="store_true", help="Testing only -- normal use must NOT set this.")
    args = parser.parse_args()

    suite = load_suite(args.suite_file) if args.suite_file else None

    if not args.skip_validation:
        print("=== Step 1: dataset validation ===")
        result = validate(resolve(args.dataset_root))
        for c in result["checks"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  [{status}] {c['check']}: {c['value']:.4f} {c['op']} {c['threshold']}")
        if not result["all_checks_passed"]:
            print("\n=== Dataset validation FAILED -- refusing to start training ===")
            sys.exit(1)
        print("=== Dataset validation PASSED ===\n")

    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing training output dir: {output_dir}")

    checkpoints_dir = output_dir / "checkpoints"
    previous_checkpoint = None
    step = args.eval_every
    while step <= args.steps:
        run_training_block(args.dataset_root, args.dataset_repo_id, output_dir, step, resume_from=previous_checkpoint)
        checkpoint_dir = checkpoints_dir / f"{step:06d}"
        previous_checkpoint = checkpoint_dir

        print(f"=== Step 3: rollout evaluation for checkpoint step {step} ===")
        server_proc = start_server_and_load(checkpoint_dir / "pretrained_model")
        try:
            summary = run_evaluation(
                label=f"v3_step{step}", real_vla_config="configs/real_vla_backend_config.json",
                object_type="plastic_bottle", strict=True, suite=suite,
            )
            summary["step"] = step
            output_json = PROJECT_ROOT / "results" / "v3_pipeline" / f"rollout_eval_step{step}.json"
            output_json.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            append_csv_row(resolve(args.csv), f"v3_step{step}", step, summary)
            print(f"step={step}: success={summary['overall_success_rate']:.2%} pick={summary['pick_rate']:.2%}")
        finally:
            stop_server(server_proc)

        step += args.eval_every

    print(f"\n=== Training + rollout evaluation sweep complete. CSV: {resolve(args.csv)} ===")
    print("Next: .venv-vla/bin/python -m benchmark.select_best_v3_checkpoint --csv " + args.csv)


if __name__ == "__main__":
    main()
