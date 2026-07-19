"""Regression tests for connecting a LOCALLY fine-tuned SmolVLA checkpoint
to the production serving path (see this task's chat report):

  - policy_semantics/manifest.py: get_manifest() now also resolves a local
    checkpoint's manifest by reading its own train_config.json (written by
    lerobot_train.py) to find which registered base checkpoint it was
    fine-tuned from, inheriting that manifest's semantic-shape claims
    instead of falling to an all-UNKNOWN manifest -- see
    _resolve_finetuned_manifest().
  - vla_server/model_loader.py: _load_official_smolvla_processors() now
    also wires the official pre/post-processor for any LOCAL directory
    that actually contains policy_preprocessor.json/
    policy_postprocessor.json on disk, not just the hardcoded
    _MODELS_WITH_OFFICIAL_PROCESSOR_FILES Hub-id allowlist -- see
    _has_official_processor_files_on_disk().

Both changes are purely additive: every existing registered model_id
(HuggingFaceVLA/smolvla_libero, lerobot/smolvla_base) and the Hub-id
allowlist path are covered here to confirm neither regressed.

Requires the local checkpoint produced by this task's fine-tuning smoke
test: outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model
(skips the checkpoint-dependent sections with a clear message if absent,
rather than failing hard on an environment that hasn't run that step).

Run: .venv-vla/bin/python -m benchmark.test_local_finetuned_checkpoint_loading
"""

import shutil
import tempfile
from pathlib import Path

from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import (
    MANIFEST_REGISTRY,
    UNKNOWN,
    ActionSpace,
    get_manifest,
)
from vla_server.model_loader import (
    _has_official_processor_files_on_disk,
    _load_official_smolvla_processors,
    _MODELS_WITH_OFFICIAL_PROCESSOR_FILES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model"
)
_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def main() -> None:
    print("=== 1. Registered base manifests: unchanged (regression) ===")
    for model_id in ("HuggingFaceVLA/smolvla_libero", "lerobot/smolvla_base"):
        manifest = get_manifest(model_id)
        check(f"{model_id}: still resolves to the exact registered manifest object", manifest is MANIFEST_REGISTRY[model_id])
    print()

    print("=== 2. Hub-id allowlist path for official processors: unchanged (regression) ===")
    check(
        "HuggingFaceVLA/smolvla_libero is still in the Hub allowlist",
        "HuggingFaceVLA/smolvla_libero" in _MODELS_WITH_OFFICIAL_PROCESSOR_FILES,
    )
    check(
        "a bare Hub-id string is never treated as a local directory with processor files",
        _has_official_processor_files_on_disk("HuggingFaceVLA/smolvla_libero") is False,
    )
    print()

    print("=== 3. Invalid/unregistered paths still hard-fail to all-UNKNOWN (regression) ===")
    nonexistent = get_manifest("/definitely/does/not/exist/anywhere")
    check("nonexistent path -> action_dimension == -1", nonexistent.action_dimension == -1)
    check("nonexistent path -> action_space == UNKNOWN", nonexistent.action_space == ActionSpace.UNKNOWN)
    check(
        "nonexistent path -> CompatibilityGate refuses it",
        CompatibilityGate.check(nonexistent).passed is False,
    )

    scratch_dir = Path(tempfile.mkdtemp(prefix="finetuned_manifest_test_"))
    try:
        # A real directory, but with no train_config.json at all.
        check(
            "existing directory with no train_config.json -> still falls to all-UNKNOWN",
            get_manifest(str(scratch_dir)).action_dimension == -1,
        )

        # A directory with a train_config.json whose pretrained_path points
        # at something NOT in MANIFEST_REGISTRY -- must not fabricate a manifest.
        unregistered_base_dir = scratch_dir / "unregistered_base"
        unregistered_base_dir.mkdir()
        (unregistered_base_dir / "train_config.json").write_text(
            '{"policy": {"pretrained_path": "someone/unregistered-checkpoint", "type": "smolvla"}}'
        )
        check(
            "train_config.json referencing an UNREGISTERED base -> still falls to all-UNKNOWN",
            get_manifest(str(unregistered_base_dir)).action_dimension == -1,
        )

        # A directory with a train_config.json correctly pointing at a real
        # registered base, plus real processor files -- should inherit.
        derived_dir = scratch_dir / "derived_from_libero"
        derived_dir.mkdir()
        (derived_dir / "train_config.json").write_text(
            '{"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero", "type": "smolvla"}}'
        )
        (derived_dir / "policy_preprocessor.json").write_text("{}")
        (derived_dir / "policy_postprocessor.json").write_text("{}")
        derived_manifest = get_manifest(str(derived_dir))
        base_manifest = MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"]
        check("synthetic derived manifest: model_id is the local path, not the base's", derived_manifest.model_id == str(derived_dir))
        check("synthetic derived manifest: action_dimension inherited", derived_manifest.action_dimension == base_manifest.action_dimension)
        check("synthetic derived manifest: gripper_convention inherited", derived_manifest.gripper_convention == base_manifest.gripper_convention)
        check("synthetic derived manifest: axis_convention_verified inherited", derived_manifest.axis_convention_verified == base_manifest.axis_convention_verified)
        check("synthetic derived manifest: revision is UNKNOWN (not inherited)", derived_manifest.revision == UNKNOWN)
        check("synthetic derived manifest: official_processor_available True (files present)", derived_manifest.official_processor_available is True)
        # This fixture's policy_postprocessor.json is a placeholder "{}"
        # (no real "steps"/state_file/safetensors) -- since a later task
        # added PolicyManifest.native_gripper_range (extracted from a
        # checkpoint's own REAL postprocessor safetensors, see
        # policy_semantics/manifest.py's _extract_gripper_native_range_from_postprocessor()),
        # this synthetic fixture correctly can no longer pass
        # CompatibilityGate overall -- it has no real gripper stats to
        # extract. See the real-checkpoint case below (item 4) for the
        # actual passing scenario.
        gate_result = CompatibilityGate.check(derived_manifest)
        check("synthetic derived manifest (placeholder processor, no real stats): native_gripper_range stays None", derived_manifest.native_gripper_range is None)
        check("synthetic derived manifest: CompatibilityGate now correctly fails on gripper_native_range_known (no real stats in this placeholder fixture)", gate_result.checks["gripper_native_range_known"]["passed"] is False)

        # Same, but WITHOUT the processor files -- must not falsely claim official_processor_available.
        derived_no_processors_dir = scratch_dir / "derived_no_processors"
        derived_no_processors_dir.mkdir()
        (derived_no_processors_dir / "train_config.json").write_text(
            '{"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero", "type": "smolvla"}}'
        )
        derived_no_proc_manifest = get_manifest(str(derived_no_processors_dir))
        check(
            "derived manifest without processor files on disk -> official_processor_available False",
            derived_no_proc_manifest.official_processor_available is False,
        )
        check(
            "derived manifest without processor files -> fails CompatibilityGate (processor_and_normalization)",
            CompatibilityGate.check(derived_no_proc_manifest).passed is False,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    print()

    print("=== 4. Real local fine-tuned checkpoint (from this task's smoke test) ===")
    if not LOCAL_CHECKPOINT.is_dir():
        print(f"SKIPPED -- {LOCAL_CHECKPOINT} not present (run the fine-tuning smoke test first)")
    else:
        check(
            "real checkpoint: has train_config.json",
            (LOCAL_CHECKPOINT / "train_config.json").is_file(),
        )
        check(
            "real checkpoint: _has_official_processor_files_on_disk() True",
            _has_official_processor_files_on_disk(str(LOCAL_CHECKPOINT)) is True,
        )
        manifest = get_manifest(str(LOCAL_CHECKPOINT))
        base_manifest = MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"]
        check("real checkpoint: manifest.model_id is the local path", manifest.model_id == str(LOCAL_CHECKPOINT))
        check("real checkpoint: action_dimension == 7 (inherited)", manifest.action_dimension == 7)
        check("real checkpoint: action_space == EE_DELTA (inherited)", manifest.action_space == ActionSpace.EE_DELTA)
        check("real checkpoint: official_processor_available True", manifest.official_processor_available is True)
        check("real checkpoint: official_processor_wired True", manifest.official_processor_wired is True)

        result = CompatibilityGate.check(manifest)
        check("real checkpoint: passes CompatibilityGate", result.passed is True, str(result.reasons))

        # Actually load the processors (lightweight -- config + normalizer
        # safetensors, no VLM/CUDA involved) to confirm processor
        # *resolution*, not just the file-existence precondition.
        preprocessor, postprocessor = _load_official_smolvla_processors(str(LOCAL_CHECKPOINT), local_files_only=True)
        check("real checkpoint: official preprocessor actually loads", preprocessor is not None)
        check("real checkpoint: official postprocessor actually loads", postprocessor is not None)
    print()

    print("=== 5. VLM fallback prohibition: unchanged (regression, not touched by this task's edits) ===")
    from vla_server.model_loader import resolve_allow_vlm_fallback
    import os

    previous = os.environ.pop("VLA_ALLOW_VLM_FALLBACK", None)
    try:
        check("VLA_ALLOW_VLM_FALLBACK unset -> resolve_allow_vlm_fallback() is False", resolve_allow_vlm_fallback() is False)
    finally:
        if previous is not None:
            os.environ["VLA_ALLOW_VLM_FALLBACK"] = previous

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
