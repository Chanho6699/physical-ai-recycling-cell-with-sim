"""Korean vs. English instruction comparison eval (v0).

Runs benchmark/run_full_recycling_cell_demo.py's real closed loop twice
-- once with a Korean instruction, once with an English one -- against
the exact same PyBullet scene (same --image-path, same backend.reset())
and, if --seed is given, the same per-step RNG seed forwarded to the
VLA server (see policy/policy_types.py's PolicyInput.seed and
vla_server/model_loader.py's torch.manual_seed() call) -- so any
behavioral difference between the two runs is attributable to the
instruction wording, not scene or sampling-noise variance.

This is a wiring/behavior-quality tool, not a benchmark suite: it does
not evaluate real task success RATE (a single scene, a handful of
steps) -- it only makes the two runs comparable and prints their
step-by-step numbers side by side. See docstring at the bottom of this
file's main() for exactly what it prints.

Usage:
  python -m benchmark.run_smolvla_language_comparison_eval \\
    --real-vla-config configs/vla_backend_smolvla_libero_config.json \\
    --image-path data/test_images/recyclable_scene.jpg \\
    --max-policy-steps 20 --seed 42
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_KO_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DEFAULT_EN_INSTRUCTION = "Pick up the plastic bottle and place it in the plastic bin."

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ko-instruction", type=str, default=DEFAULT_KO_INSTRUCTION)
    parser.add_argument("--en-instruction", type=str, default=DEFAULT_EN_INSTRUCTION)
    parser.add_argument("--real-vla-config", type=str, default="configs/vla_backend_smolvla_libero_config.json")
    parser.add_argument("--image-path", type=str, default="data/test_images/recyclable_scene.jpg")
    parser.add_argument("--max-policy-steps", type=int, default=20)
    parser.add_argument("--control-loop-timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/language_comparison_eval")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "By default this eval runs with --strict-real-vla (no fallback allowed, "
            "compatibility/degraded_input/fallback_used all enforced every step) -- this is "
            "the whole point of a quantitative behavior comparison. Pass this flag only for "
            "local development against a server that isn't up yet; the resulting numbers are "
            "NOT a valid comparison and the report should say so."
        ),
    )
    return parser.parse_args()


def build_cmd(args, instruction: str, log_path: Path) -> list:
    """The exact CLI invocation for one run -- pulled out of run_one() so
    a test can assert the Korean and English runs differ ONLY in
    --instruction/--eval-log-path (never scene, seed, step budget,
    strict-mode, etc.) without actually launching a subprocess."""
    cmd = [
        sys.executable, "-m", "benchmark.run_full_recycling_cell_demo",
        "--policy", "dummy-openvla", "--policy-backend", "real-vla",
        "--real-vla-config", args.real_vla_config,
        "--real-vla-observation-mode", "pybullet",
        "--instruction", instruction,
        "--image-path", args.image_path,
        "--headless",
        "--max-policy-steps", str(args.max_policy_steps),
        "--control-loop-timeout-s", str(args.control_loop_timeout_s),
        "--steps-per-action", str(args.steps_per_action),
        "--seed", str(args.seed),
        "--eval-log-path", str(log_path),
    ]
    if not args.allow_fallback:
        cmd += ["--strict-real-vla"]
    return cmd


def run_one(args, instruction: str, tag: str, log_path: Path) -> dict:
    if log_path.exists():
        log_path.unlink()
    cmd = build_cmd(args, instruction, log_path)

    print(f"\n=== Running [{tag}]: {instruction!r} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=args.control_loop_timeout_s + 60)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])

    final_status = None
    for line in result.stdout.splitlines():
        if line.startswith("final_status:"):
            final_status = line.split(":", 1)[1].strip()

    steps = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as log_file:
            for line in log_file:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))

    return {
        "tag": tag,
        "instruction": instruction,
        "returncode": result.returncode,
        "final_status": final_status,
        "steps": steps,
        "stdout_tail": result.stdout[-2000:],
    }


def summarize(run: dict) -> None:
    steps = run["steps"]
    print(f"\n--- [{run['tag']}] {run['instruction']!r} ---")
    print(f"returncode={run['returncode']} final_status={run['final_status']} steps_logged={len(steps)}")
    if not steps:
        print("(no eval log steps recorded -- run likely failed before the control loop, see stdout above)")
        return

    distances = [step["distance_to_object_m"] for step in steps]
    print(f"distance_to_object per step: {[round(d, 3) for d in distances]}")
    print(f"distance_to_object: start={distances[0]:.3f} end={distances[-1]:.3f} "
          f"min={min(distances):.3f} decreased_overall={distances[-1] < distances[0]}")

    gripper_commands = [step["gripper_command"] for step in steps]
    close_count = sum(1 for command in gripper_commands if command == "close")
    print(f"gripper_command sequence: {gripper_commands}")
    print(f"gripper close commands issued: {close_count}/{len(gripper_commands)}")

    repeated_count = sum(1 for step in steps if step.get("action_repeated"))
    print(f"action_repeated steps: {repeated_count}/{len(steps)}")

    translations = [
        step["canonical_command_after_safety_filter"]["translation_m"]
        for step in steps
        if step.get("canonical_command_after_safety_filter")
    ]
    if translations:
        dominant_axis = ["x", "y", "z"]
        avg = [sum(t[i] for t in translations) / len(translations) for i in range(3)]
        print(f"avg translation_m per axis (x,y,z): {[round(v, 4) for v in avg]}")

    degraded_count = sum(1 for step in steps if step.get("degraded_input"))
    fallback_count = sum(1 for step in steps if step.get("fallback_used"))
    print(f"degraded_input steps: {degraded_count}/{len(steps)}  fallback_used steps: {fallback_count}/{len(steps)}")


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ko_run = run_one(args, args.ko_instruction, "KO", output_dir / "ko_steps.jsonl")
    en_run = run_one(args, args.en_instruction, "EN", output_dir / "en_steps.jsonl")

    print("\n" + "=" * 70)
    print("=== Korean vs. English comparison ===")
    summarize(ko_run)
    summarize(en_run)

    comparison_path = output_dir / "comparison_summary.json"
    with open(comparison_path, "w", encoding="utf-8") as comparison_file:
        json.dump(
            {
                "ko": {"instruction": ko_run["instruction"], "final_status": ko_run["final_status"], "steps": ko_run["steps"]},
                "en": {"instruction": en_run["instruction"], "final_status": en_run["final_status"], "steps": en_run["steps"]},
            },
            comparison_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved full comparison data to {comparison_path}")


if __name__ == "__main__":
    main()
