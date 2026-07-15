"""Tests for the policy_semantics manifest/gate/adapter refactor (v0).

Runs standalone (matches this repo's benchmark/*.py convention -- no
pytest), asserting each scenario and printing PASS/FAIL per case plus an
overall summary. Exercises vla_adapters/smolvla_adapter.py's
normalize_model_output() directly against real CompatibilityResults from
policy_semantics/compatibility_gate.py -- no GPU/model load required,
since the scenarios under test are about semantic gating, not inference.

Run: python -m benchmark.test_policy_semantics
"""

import warnings

from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import (
    PANDA_TARGET_EMBODIMENT,
    UNKNOWN,
    ActionSpace,
    PolicyManifest,
    get_manifest,
)
from vla_adapters.smolvla_adapter import SmolVLAActionAdapter

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def make_adapter(model_id_or_path: str) -> SmolVLAActionAdapter:
    return SmolVLAActionAdapter(config={"model_id_or_path": model_id_or_path})


def main() -> None:
    print("=== 1. smolvla_base vs. Panda Cartesian target -- must be INCOMPATIBLE ===")
    base_manifest = get_manifest("lerobot/smolvla_base")
    base_result = CompatibilityGate.check(base_manifest, smoke_test_mode=False)
    check("smolvla_base.passed is False", base_result.passed is False)
    check(
        "smolvla_base fails on action_space (joint_position != ee_delta)",
        not base_result.checks["action_space_matches_target"]["passed"],
    )
    check(
        "smolvla_base fails on source_embodiment (SO-100 != Panda)",
        not base_result.checks["source_embodiment"]["passed"],
    )
    print()

    print("=== 2. Legacy 6D->7D filler must not be reachable in production ===")
    adapter = make_adapter("lerobot/smolvla_base")
    context_production = {
        "step_index": 0,
        "phase": "move_to_object",
        "compatibility": base_result.to_dict(),
        "smoke_test_mode": False,  # production: smoke test NOT enabled
    }
    raw_6d = [-0.080, 0.145, 0.053, -0.087, 0.355, -0.509]  # the real observed smolvla_base raw output
    normalized = adapter.normalize_model_output(raw_6d, context_production)
    check("production call with 6D input refuses (action is None)", normalized["action"] is None)
    check(
        "rejection reason mentions compatibility_gate_rejected",
        "compatibility_gate_rejected" in normalized["info"].get("reason", ""),
    )
    check("info.semantic_action_valid is False", normalized["info"]["semantic_action_valid"] is False)
    print()

    print("=== 3. smolvla_libero manifest loads normally ===")
    libero_manifest = get_manifest("HuggingFaceVLA/smolvla_libero")
    check("action_dimension == 7", libero_manifest.action_dimension == 7)
    check("action_space == EE_DELTA", libero_manifest.action_space == ActionSpace.EE_DELTA)
    check("required_camera_roles == ['main', 'wrist']", libero_manifest.required_camera_roles == ["main", "wrist"])
    check("state_fields has 3 entries summing to 8", sum(libero_manifest.state_fields.values()) == 8)
    # rotation_representation/reference_frame/gripper_convention were
    # confirmed directly from robosuite/LIBERO official source (see
    # policy_semantics/adapters/smolvla_libero_adapter.py's module
    # docstring for exact citations), and axis_convention_verified is now
    # True too -- confirmed via real cross-simulator simulation, see
    # benchmark/verify_panda_axis_convention.py /
    # docs/panda_axis_cross_verification.md. See
    # benchmark/test_smolvla_libero_action_adapter.py for the full
    # rotation/capability/gate test suite this unlocked.
    check(
        "rotation/frame/gripper are now confirmed, not UNKNOWN",
        libero_manifest.rotation_representation == "axis_angle"
        and libero_manifest.reference_frame == "robot_base"
        and libero_manifest.gripper_convention != UNKNOWN,
    )
    check("axis_convention_verified is now True (real cross-sim verification)", libero_manifest.axis_convention_verified is True)
    print()

    print("=== 4. Wrong action dimension is rejected (dimension alone is never sufficient) ===")
    # A synthetic compatibility dict with passed=False, shape_only_allowed=True
    # -- decoupled from any real manifest's current gate status (smolvla_libero's
    # gate now legitimately passes as of the rotation-control turn, see
    # benchmark/test_smolvla_libero_action_adapter.py; this test's point is
    # purely "the legacy shape-only path still refuses a length that isn't 6 or 7",
    # independent of which checkpoint that path happens to be exercised for).
    libero_adapter = make_adapter("HuggingFaceVLA/smolvla_libero")
    bad_length_context = {
        "step_index": 0,
        "phase": "move_to_object",
        "compatibility": {"passed": False, "shape_only_allowed": True, "reasons": []},
        "smoke_test_mode": True,
    }
    raw_5d = [0.1, 0.2, 0.3, 0.4, 0.5]
    rejected = libero_adapter.normalize_model_output(raw_5d, bad_length_context)
    check("length-5 raw output rejected even in smoke_test_mode", rejected["action"] is None)
    check(
        "rejection reason mentions wrong_length",
        "wrong_length" in rejected["info"].get("reason", ""),
    )
    print()

    print("=== 5. UNKNOWN action semantics -- production must refuse ===")
    unknown_semantics_manifest = PolicyManifest(
        model_id="test/unknown-semantics-checkpoint",
        revision=UNKNOWN,
        source_embodiment=UNKNOWN,
        required_camera_roles=["main"],
        state_fields={"observation.state": 6},
        action_dimension=7,
        action_space=ActionSpace.UNKNOWN,
        relative_or_absolute=UNKNOWN,
        rotation_representation=UNKNOWN,
        reference_frame=UNKNOWN,
        gripper_included=False,
        gripper_index=None,
        gripper_convention=UNKNOWN,
        action_chunk_size=1,
        normalization=UNKNOWN,
        official_processor_available=False,
        official_processor_wired=False,
    )
    unknown_result = CompatibilityGate.check(unknown_semantics_manifest, smoke_test_mode=False)
    check("UNKNOWN-semantics manifest fails the gate", unknown_result.passed is False)
    check(
        "action_semantics_known check specifically fails",
        not unknown_result.checks["action_semantics_known"]["passed"],
    )
    unknown_adapter = make_adapter("test/unknown-semantics-checkpoint")
    unknown_context = {
        "step_index": 0,
        "phase": "move_to_object",
        "compatibility": unknown_result.to_dict(),
        "smoke_test_mode": False,
    }
    unknown_rejected = unknown_adapter.normalize_model_output([0.0] * 7, unknown_context)
    check("production call refuses for UNKNOWN-semantics manifest", unknown_rejected["action"] is None)
    print()

    print("=== 6. smoke_test_mode: warning + semantic_action_valid=false ===")
    # Recompute with smoke_test_mode=True so shape_only_allowed=True -- base_result
    # from case 1 was computed with smoke_test_mode=False and would correctly
    # refuse outright even in a smoke-test request, which is a different (also
    # correct) behavior than this case is demonstrating.
    base_result_smoke = CompatibilityGate.check(base_manifest, smoke_test_mode=True)
    smoke_context = {
        "step_index": 0,
        "phase": "move_to_object",
        "compatibility": base_result_smoke.to_dict(),  # still INCOMPATIBLE overall, but shape_only_allowed=True
        "smoke_test_mode": True,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        smoke_result = adapter.normalize_model_output(raw_6d, smoke_context)
    check("smoke test produces a 7-length action", smoke_result["action"] is not None and len(smoke_result["action"]) == 7)
    check("smoke test action's gripper is the neutral fill value", smoke_result["action"][6] == 0.0)
    check("info.semantic_action_valid is False even in smoke_test_mode", smoke_result["info"]["semantic_action_valid"] is False)
    check("info.smoke_test_mode is True", smoke_result["info"].get("smoke_test_mode") is True)
    check("a Python warning was emitted", any("smoke_test_mode only" in str(w.message) for w in caught))
    print()

    print("=== 7. mock-action / existing dummy pipeline untouched (smoke check) ===")
    from vla_adapters.mock_vla_adapter import MockVLAAdapter
    from policy.dummy_openvla_policy import DummyOpenVLAPolicy
    from policy.policy_types import PolicyInput

    mock_adapter = MockVLAAdapter()
    dummy_policy = DummyOpenVLAPolicy()
    policy_input_dict = {
        "instruction": "pick up the bottle",
        "image": None,
        "robot_state": {"end_effector_position": [0.5, 0.0, 0.5], "held_object": False, "task_status": "running"},
        "task_goal": {"action": "pick_and_place", "target_object": "plastic_bottle", "target_bin": "plastic_bin"},
        "target_object_position": [0.4, -0.1, 0.05],
        "bin_position": [0.3, 0.35, 0.05],
        "step_index": 0,
        "phase": "move_to_object",
    }
    model_input = mock_adapter.build_model_input(policy_input_dict)
    policy_input = model_input["policy_input"]
    raw_output = dummy_policy.predict_action(policy_input)
    mock_normalized = mock_adapter.normalize_model_output(raw_output, {"step_index": 0, "phase": "move_to_object"})
    check("mock-action pipeline still produces a 7-length action", len(mock_normalized["action"]) == 7)
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
