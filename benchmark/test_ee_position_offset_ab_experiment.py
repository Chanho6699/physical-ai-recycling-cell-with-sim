"""Tests for benchmark/run_ee_position_offset_ab_experiment.py (v0).

Most tests run against a FakeServerPolicy mock -- no live GPU/network VLA
server required. compute_offset_candidates() is exercised for real (it
downloads/reads the actual HuggingFaceVLA/libero dataset via
huggingface_hub, which is already cached locally from a prior turn's
investigation, so this is fast and not a live inference call). Covers:

  1. apply_position_offset() correctness + non-mutation (only ee_position.x
     changes; y/z/orientation/gripper/everything else untouched)
  2. run_condition() wiring: the model receives the OFFSET ee_position,
     but ground truth grading (cosine/sign-match/distance) and the real
     backend's physics use the REAL, un-offset position
  3. compute_offset_candidates() produces a "none" (0.0) baseline plus
     fractional candidates, all derived from a real, non-zero
     mean-alignment computation
  4. summarize_offsets()/decide_causal_verdict()'s A-vs-B logic on
     synthetic summaries
  5. this module's functions are never referenced from any production directory
  6. regression: benchmark.test_environment_state_alignment_diagnostic

Run: python -m benchmark.test_ee_position_offset_ab_experiment
"""

from pathlib import Path

from benchmark.run_counterfactual_direction_benchmark import DEFAULT_POSITIONS
from benchmark.run_ee_position_offset_ab_experiment import (
    apply_position_offset,
    compute_offset_candidates,
    decide_causal_verdict,
    run_condition,
    summarize_offsets,
)
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Echoes the MODEL-INPUT ee_position.x straight into commanded x
    (scaled down), so a test can verify the offset actually reaches the
    policy's decision -- and records every PolicyInput it receives so the
    test can also inspect exactly what ee_position value was sent."""

    def __init__(self):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.received_inputs = []

    def reset(self) -> None:
        self.received_inputs = []

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.received_inputs.append(policy_input)
        ee_x = policy_input.robot_state["ee_position"][0]
        translation = [ee_x * 0.01, 0.0, -0.005]
        action = translation + [0.0, 0.0, 0.0, 1.0]
        info = {
            "policy_backend": "real-vla",
            "inference_latency_ms": 700.0,
            "fallback_used": False,
            "real_vla_request_failed": False,
            "compatibility": {"passed": True},
            "semantic_action_valid": True,
            "degraded_input": False,
            "action_postprocess": {
                "canonical_command": {"translation_m": translation, "gripper_opening_01": 1.0},
                "canonical_command_pre_safety_filter": {"translation_m": translation},
                "safety_filter_clipped": False,
                "safety_clipped": False,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


def main() -> None:
    print("=== 1. apply_position_offset() correctness ===")
    base_state = {
        "ee_position": [0.3, 0.1, 0.4], "ee_orientation_axis_angle": [3.14, 0.0, 0.0],
        "gripper_qpos": [0.02, -0.02],
    }
    offset_state = apply_position_offset(base_state, -0.3)
    check("only x changes", offset_state["ee_position"] == [0.0, 0.1, 0.4], f"got {offset_state['ee_position']}")
    check("y/z unchanged", offset_state["ee_position"][1] == 0.1 and offset_state["ee_position"][2] == 0.4)
    check("orientation untouched", offset_state["ee_orientation_axis_angle"] == base_state["ee_orientation_axis_angle"])
    check("gripper_qpos untouched", offset_state["gripper_qpos"] == base_state["gripper_qpos"])
    check("original dict not mutated", base_state["ee_position"] == [0.3, 0.1, 0.4], f"mutated: {base_state['ee_position']}")

    zero_offset_state = apply_position_offset(base_state, 0.0)
    check("zero offset leaves x unchanged", zero_offset_state["ee_position"][0] == 0.3)
    print()

    print("=== 2. run_condition() wiring: model sees offset, ground truth/physics use real position ===")
    fake_policy = FakeServerPolicy()
    offset_x = -0.3
    rows = run_condition(
        fake_policy, "center_right", DEFAULT_POSITIONS["center_right"], offset_x, "pick up the bottle",
        [0.3, 0.35, 0.05], seed=42, steps_per_condition=2, steps_per_action=10, object_type="plastic_bottle",
        strict=True, label="test",
    )
    check("2 rows produced", len(rows) == 2, f"got {len(rows)}")
    first = rows[0]
    check(
        "model_input_ee_position.x == ground_truth_ee_position.x + offset_x",
        abs(first["model_input_ee_position"][0] - (first["ground_truth_ee_position"][0] + offset_x)) < 1e-9,
        f"model_input={first['model_input_ee_position'][0]} ground_truth={first['ground_truth_ee_position'][0]} offset={offset_x}",
    )
    check(
        "model_input_ee_position.y/z == ground_truth (only x offset)",
        abs(first["model_input_ee_position"][1] - first["ground_truth_ee_position"][1]) < 1e-9
        and abs(first["model_input_ee_position"][2] - first["ground_truth_ee_position"][2]) < 1e-9,
    )
    check(
        "ground_truth_ee_position matches the real PyBullet reset EE position (~0.307 in x)",
        abs(first["ground_truth_ee_position"][0] - 0.3069) < 0.01,
        f"got {first['ground_truth_ee_position'][0]}",
    )
    check(
        "FakeServerPolicy received the OFFSET x (proves the override actually reached the policy)",
        abs(fake_policy.received_inputs[0].robot_state["ee_position"][0] - first["model_input_ee_position"][0]) < 1e-9,
    )
    check(
        "step 1's ground truth EE position differs from step 0's (real physics moved the real robot)",
        rows[0]["ground_truth_ee_position"] != rows[1]["ground_truth_ee_position"],
    )
    print()

    print("=== 3. compute_offset_candidates() -- real dataset, auto-computed (not hand-picked) ===")
    offset_info = compute_offset_candidates()
    candidates = offset_info["candidates"]
    check("'none' candidate is exactly 0.0", candidates.get("none") == 0.0, f"got {candidates.get('none')}")
    check("'full' candidate equals the computed real-vs-ours mean gap", abs(candidates["full"] - offset_info["full_offset_x"]) < 1e-9)
    check(
        "'half'/'three_quarter' are proportional fractions of 'full' (auto-derived, not hand-picked)",
        abs(candidates["half"] - offset_info["full_offset_x"] * 0.5) < 1e-9
        and abs(candidates["three_quarter"] - offset_info["full_offset_x"] * 0.75) < 1e-9,
    )
    check(
        "the real-vs-ours mean gap is non-trivial (confirms this project's ee_position.x really differs "
        "from real training data, matching the prior environment-alignment diagnostic's finding)",
        abs(offset_info["full_offset_x"]) > 0.1,
        f"full_offset_x={offset_info['full_offset_x']}",
    )
    print()

    print("=== 4. summarize_offsets()/decide_causal_verdict() ===")
    def make_row(label, position_name, seed, step, cosine, sign_x, dist_before, dist_final):
        return {
            "label": f"{label}__{position_name}", "position_name": position_name, "seed": seed, "step": step,
            "cosine_commanded_vs_object": cosine, "sign_match_x": sign_x, "sign_match_y": None,
            "far_gripper_close": False, "degraded_input": False, "fallback_used": False, "semantic_action_valid": True,
            "distance_to_object_before": dist_before, "final_distance_to_object": dist_final,
        }

    # "full" offset clearly improves cosine/x-sign-accuracy over "none".
    improving_rows = []
    for seed in (0, 42, 123):
        improving_rows.append(make_row("none", "center_right", seed, 0, 0.05, False, 0.4, 0.38))
        improving_rows.append(make_row("full", "center_right", seed, 0, 0.55, True, 0.4, 0.30))
    improving_summary = summarize_offsets(improving_rows, {"none": 0.0, "full": -0.4})
    check("mean_cosine computed per offset", abs(improving_summary["none"]["mean_cosine"] - 0.05) < 1e-9, f"got {improving_summary['none']['mean_cosine']}")
    check("x_sign_accuracy computed per offset", improving_summary["full"]["x_sign_accuracy"] == 1.0)
    verdict_a = decide_causal_verdict(improving_summary)
    check("verdict A when the offset meaningfully improves cosine/x-sign-accuracy", verdict_a["verdict"] == "A", str(verdict_a))

    # "full" offset barely changes anything -> verdict B.
    flat_rows = []
    for seed in (0, 42, 123):
        flat_rows.append(make_row("none", "center_right", seed, 0, 0.10, False, 0.4, 0.38))
        flat_rows.append(make_row("full", "center_right", seed, 0, 0.12, False, 0.4, 0.37))
    flat_summary = summarize_offsets(flat_rows, {"none": 0.0, "full": -0.4})
    verdict_b = decide_causal_verdict(flat_summary)
    check("verdict B when the offset barely changes cosine/x-sign-accuracy", verdict_b["verdict"] == "B", str(verdict_b))
    print()

    print("=== 5. this module never referenced from production ===")
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy"]
    hits = []
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "apply_position_offset(" in text or "from benchmark.run_ee_position_offset_ab_experiment" in text:
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    check("no production file imports/calls this module's functions", len(hits) == 0, f"unexpected: {hits}")
    print()

    print("=== 6. regression: prior environment-alignment diagnostic suite ===")
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "benchmark.test_environment_state_alignment_diagnostic"], capture_output=True, text=True, timeout=300)
    passed = "ALL CHECKS PASSED" in result.stdout
    check("benchmark.test_environment_state_alignment_diagnostic -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
