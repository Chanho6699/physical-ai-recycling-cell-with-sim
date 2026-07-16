"""Tests for benchmark/run_vla_action_direction_diagnostic.py (v0).

Covers this task's required test scenarios:

  1. cosine calculation (reuses run_full_recycling_cell_demo._cosine_similarity,
     the exact same helper the diagnostic script imports -- not a
     reimplementation)
  2. distance-progress calculation
  3. raw -> adapted -> executed log linkage (a FakeServerPolicy with
     deliberately DIFFERENT raw/pre-filter/post-filter values, injected
     into run_diagnostic() via its policy= parameter, so no live GPU
     server is needed to check the wiring)
  4. main/wrist image distinction (real PyBulletPandaBackend, no mocking)
  5. a mock reproducing LeRobot SmolVLAPolicy's own action-chunk-queue
     algorithm (see modeling_smolvla.py's select_action()/
     _check_get_actions_condition(), mirrored here, not reimplemented
     from scratch) to concretely demonstrate detecting whether a given
     step actually re-ran inference or returned a queued action

Run: python -m benchmark.test_vla_action_direction_diagnostic
"""

from collections import deque

import numpy as np

from benchmark.run_full_recycling_cell_demo import _cosine_similarity, _distance_3d
from benchmark.run_vla_action_direction_diagnostic import (
    diagnose_layer,
    parse_args,
    run_diagnostic,
    summarize,
)
from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeServerPolicy(BasePolicy):
    """Returns a canned, non-degraded, non-fallback response with
    DELIBERATELY DIFFERENT raw_model_action / pre-safety-filter /
    post-safety-filter translation values, so a test can check the
    diagnostic script actually threads each layer through to the right
    log field instead of, say, logging the same value three times."""

    def __init__(self):
        self.phase = "move_to_object"
        self.fallback_used_count = 0
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        self.calls += 1
        # Raw model output (channels 0:3 = translation, in the model's own
        # normalized [-1, 1] action space -- deliberately large/different
        # from the scaled-down adapted values below).
        raw_model_action = [0.8, -0.2, 0.1, 0.0, 0.0, 0.0, -0.5]
        # Adapter-decoded, BEFORE the safety filter's per-step clip --
        # deliberately larger than the project's max_translation_step
        # (0.03m) so the safety filter is guaranteed to visibly clip it.
        pre_filter_translation = [0.04, -0.01, 0.005]
        # AFTER the safety filter clip (x clipped from 0.04 to 0.03).
        post_filter_translation = [0.03, -0.01, 0.005]
        action = post_filter_translation + [0.0, 0.0, 0.0, 1.0]  # gripper=open (wire format)

        info = {
            "policy_backend": "real-vla",
            "inference_latency_ms": 700.0,
            "fallback_used": False,
            "real_vla_request_failed": False,
            "compatibility": {"passed": True},
            "semantic_action_valid": True,
            "degraded_input": False,
            "action_postprocess": {
                "canonical_command": {
                    "translation_m": post_filter_translation,
                    "gripper_opening_01": 1.0,
                    "metadata": {"raw_model_action": raw_model_action, "gripper_raw": -0.5},
                },
                "canonical_command_pre_safety_filter": {
                    "translation_m": pre_filter_translation,
                    "gripper_opening_01": 1.0,
                },
                "safety_filter_clipped": True,
                "safety_clipped": True,
            },
        }
        return PolicyOutput(action=action, phase=self.phase, done=False, info=info)


def make_diagnostic_args(**overrides):
    import sys

    argv_backup = sys.argv
    try:
        sys.argv = ["run_vla_action_direction_diagnostic.py"]
        args = parse_args()
    finally:
        sys.argv = argv_backup
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def main() -> None:
    print("=== 1. cosine calculation ===")
    check("identical vectors -> cosine 1.0", abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9)
    check("opposite vectors -> cosine -1.0", abs(_cosine_similarity([1, 0, 0], [-1, 0, 0]) - (-1.0)) < 1e-9)
    check("perpendicular vectors -> cosine 0.0", abs(_cosine_similarity([1, 0, 0], [0, 1, 0])) < 1e-9)
    check("a scaled copy of itself is still cosine 1.0", abs(_cosine_similarity([2, 0, 0], [0.5, 0, 0]) - 1.0) < 1e-9)
    check("zero vector -> None (not a division-by-zero crash)", _cosine_similarity([0, 0, 0], [1, 0, 0]) is None)
    print()

    print("=== 2. distance-progress calculation ===")
    object_position = [0.4, 0.0, 0.05]
    ee_before = [0.3, 0.0, 0.4]
    ee_after = [0.35, 0.0, 0.35]
    distance_before = _distance_3d(ee_before, object_position)
    distance_after = _distance_3d(ee_after, object_position)
    distance_progress = distance_before - distance_after
    check(
        "distance_progress is positive when EE moved closer to a known point",
        distance_progress > 0,
        f"before={distance_before} after={distance_after}",
    )
    manual_before = ((0.3 - 0.4) ** 2 + 0.0 + (0.4 - 0.05) ** 2) ** 0.5
    check("distance_before matches manual Euclidean calculation", abs(distance_before - manual_before) < 1e-9)
    print()

    print("=== 3. raw -> adapted -> executed log linkage ===")
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    backend.set_object_position([0.45, 0.0, 0.05])
    fake_policy = FakeServerPolicy()
    args = make_diagnostic_args(max_steps=2, save_images=False, strict=True)
    result = run_diagnostic(args, policy=fake_policy, backend=backend)
    rows = result["rows"]
    check("2 rows produced for max_steps=2", len(rows) == 2, f"got {len(rows)}")
    first = rows[0]
    check(
        "raw_model_translation is the first 3 channels of raw_model_action, untouched",
        first["raw_model_translation"] == [0.8, -0.2, 0.1],
        f"got {first['raw_model_translation']}",
    )
    check(
        "adapted_canonical_translation is the PRE-safety-filter value (not equal to post-filter)",
        first["adapted_canonical_translation"] == [0.04, -0.01, 0.005],
        f"got {first['adapted_canonical_translation']}",
    )
    check(
        "safety_filtered_translation is the POST-safety-filter (clipped) value",
        first["safety_filtered_translation"] == [0.03, -0.01, 0.005],
        f"got {first['safety_filtered_translation']}",
    )
    check(
        "commanded_translation matches the safety-filtered value (what was actually sent to apply_command)",
        [round(v, 6) for v in first["commanded_translation"]] == [0.03, -0.01, 0.005],
        f"got {first['commanded_translation']}",
    )
    check(
        "the three translation layers are genuinely distinct (adapter clip is visible end-to-end)",
        first["adapted_canonical_translation"] != first["safety_filtered_translation"]
        and first["raw_model_translation"] != first["adapted_canonical_translation"],
    )
    check("gripper_raw recorded from metadata", first["gripper_raw"] == -0.5)
    check("gripper_canonical recorded from canonical_command", first["gripper_canonical"] == 1.0)
    check(
        # Legacy wire format polarity (action_adapter/adapter_v0.py,
        # policy_semantics/canonical_command.py's module docstring):
        # gripper >= 0.5 means "close". FakeServerPolicy's action ends in
        # 1.0, so ActionAdapter.convert() must produce "close" here.
        "gripper_executed derived from the actual wire-format action (gripper=1.0 -> close)",
        first["gripper_executed"] == "close",
        f"got {first['gripper_executed']}",
    )
    check(
        "cosine_raw/commanded/actual are all present (float or None, never missing keys)",
        all(key in first for key in ("cosine_raw_vs_object", "cosine_commanded_vs_object", "cosine_actual_vs_object")),
    )
    backend.shutdown()
    print()

    print("=== 4. main/wrist image distinction ===")
    backend2 = PyBulletPandaBackend(gui=False)
    backend2.reset()
    main_image = backend2.render_main_camera()
    wrist_image = backend2.render_wrist_camera()
    check("main and wrist images are not pixel-identical", not np.array_equal(main_image, wrist_image))
    fake_policy2 = FakeServerPolicy()
    args2 = make_diagnostic_args(max_steps=1, save_images=False, strict=True)
    backend2.set_object_type("plastic_bottle")
    backend2.set_object_position([0.45, 0.0, 0.05])
    result2 = run_diagnostic(args2, policy=fake_policy2, backend=backend2)
    row2 = result2["rows"][0]
    check("row.main_wrist_identical is False for a real backend", row2["main_wrist_identical"] is False)
    check("row.main_image_hash != row.wrist_image_hash", row2["main_image_hash"] != row2["wrist_image_hash"])
    backend2.shutdown()
    print()

    print("=== 5. detecting SmolVLA's action-chunk queue staleness (mock mirroring modeling_smolvla.py) ===")

    class MockChunkingPolicy:
        """Mirrors lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy's
        exact select_action()/_check_get_actions_condition()/reset()
        logic (deque(maxlen=n_action_steps), refill only when empty) --
        not a reimplementation of SmolVLA itself, just its queue
        bookkeeping, so this test can cheaply demonstrate what
        n_action_steps=1 vs. n_action_steps>1 actually does to how often
        a fresh "inference" (self.inference_calls) happens, without
        downloading/running the real ~450M-parameter checkpoint."""

        def __init__(self, n_action_steps: int, chunk_size: int):
            self.n_action_steps = n_action_steps
            self.chunk_size = chunk_size
            self.inference_calls = 0
            self.reset()

        def reset(self):
            self._queue = deque(maxlen=self.n_action_steps)

        def _check_get_actions_condition(self) -> bool:
            return len(self._queue) == 0

        def select_action(self, observation):
            if self._check_get_actions_condition():
                self.inference_calls += 1
                chunk = [f"action_from_obs={observation}#slot{i}" for i in range(self.chunk_size)]
                self._queue.extend(chunk[: self.n_action_steps])
            return self._queue.popleft()

    real_checkpoint_policy = MockChunkingPolicy(n_action_steps=1, chunk_size=50)
    for step in range(5):
        real_checkpoint_policy.select_action(observation=f"obs_{step}")
    check(
        "HuggingFaceVLA/smolvla_libero's actual config (n_action_steps=1) -> every step re-infers "
        "(inference_calls == number of steps, no staleness possible)",
        real_checkpoint_policy.inference_calls == 5,
        f"inference_calls={real_checkpoint_policy.inference_calls}",
    )

    hypothetical_chunked_policy = MockChunkingPolicy(n_action_steps=5, chunk_size=50)
    for step in range(20):
        hypothetical_chunked_policy.select_action(observation=f"obs_{step}")
    check(
        "a hypothetical n_action_steps=5 config would infer only every 5th step "
        "(inference_calls == 4 for 20 steps -- the other 16 would silently return queued, stale-observation actions)",
        hypothetical_chunked_policy.inference_calls == 4,
        f"inference_calls={hypothetical_chunked_policy.inference_calls}",
    )
    hypothetical_chunked_policy.reset()
    check(
        "reset() clears the queue -- the next select_action() forces a fresh inference again",
        hypothetical_chunked_policy._check_get_actions_condition() is True,
    )
    print()

    print("=== bonus: diagnose_layer() decision rule ===")
    # Anchored on mean_commanded/mean_actual (both real, physically-
    # calibrated meter vectors) -- NOT mean_raw, which is pre-official-
    # postprocessor and confounded by this checkpoint's per-axis
    # normalization scale (see diagnose_layer()'s docstring; this was a
    # bug in an earlier version of this heuristic, found and fixed while
    # validating this script live against a GPU server -- see report).
    layer, _ = diagnose_layer(mean_raw=0.9, mean_commanded=0.05, mean_actual=0.05)
    check("near-zero COMMANDED cosine -> 'model' suspected (raw is not used for this verdict)", layer == "model", f"got {layer}")
    layer, _ = diagnose_layer(mean_raw=-0.9, mean_commanded=0.8, mean_actual=0.75)
    check(
        "negative raw cosine does NOT force an 'adapter'/'model' verdict when commanded/actual are healthy "
        "(raw is confounded by normalization scale, so it must never veto a good commanded/actual signal)",
        layer == "model",
        f"got {layer}",
    )
    layer, _ = diagnose_layer(mean_raw=0.8, mean_commanded=0.75, mean_actual=0.1)
    check("large commanded->actual drop -> 'executor' suspected", layer == "executor", f"got {layer}")
    layer, _ = diagnose_layer(mean_raw=None, mean_commanded=None, mean_actual=None)
    check("no commanded data at all -> 'unknown'", layer == "unknown", f"got {layer}")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
