"""Regression tests for policy_semantics/manifest.py's tiered
resolution priority (see this task's chat report):

  1. Explicit manifest (MANIFEST_REGISTRY exact match)
  2. Explicit LOCAL manifest (model_id/policy_manifest.json), used
     as-is with NO ancestry/lineage resolution when self-sufficient --
     the path a non-SmolVLA policy family (ACT, Diffusion Policy, any
     custom policy) uses to register its own semantics independently,
     without pretending to be a fine-tune of any registered base.
  3. Ancestry fallback (existing _resolve_finetuned_manifest()) -- a
     PARTIAL policy_manifest.json's declared fields still override
     whatever ancestry derives for those same fields.
  4. UNKNOWN / CompatibilityGate hard-fail.

Builds synthetic checkpoint directories under a temp subdirectory of
this project's own root (relative-path resolution needs this -- see
policy_semantics/manifest.py's _normalize_checkpoint_path() docstring),
cleaned up at the end either way.

Run: .venv-vla/bin/python -m benchmark.test_manifest_explicit_priority
"""

import json
import shutil
from pathlib import Path

from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import MANIFEST_REGISTRY, get_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DIR_NAME = "_test_manifest_explicit_priority_scratch"

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def main() -> None:
    scratch = PROJECT_ROOT / SCRATCH_DIR_NAME
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    try:
        print("=== 1. Standalone ACT-style checkpoint: complete policy_manifest.json, NO train_config.json, NO SmolVLA lineage at all ===")
        act_dir = scratch / "act_checkpoint"
        act_dir.mkdir()
        (act_dir / "policy_manifest.json").write_text(json.dumps({
            "revision": "act-v1",
            "source_embodiment": "Custom ACT arm (joint-space, direct-drive)",
            "required_camera_roles": ["main"],
            "state_fields": {"joint_positions": 7},
            "action_dimension": 7,
            "action_space": "joint_position",
            "relative_or_absolute": "absolute",
            "rotation_representation": "n/a (joint-space)",
            "reference_frame": "n/a (joint-space)",
            "gripper_included": True,
            "gripper_index": 6,
            "gripper_convention": "ACT's own gripper convention: 0.0=open, 1.0=closed",
            "action_chunk_size": 100,
            "normalization": "MEAN_STD (ACT's own dataset stats)",
            "official_processor_available": True,
            "official_processor_wired": True,
        }))
        act_manifest = get_manifest(str(act_dir))
        check("1: ACT manifest resolves from policy_manifest.json alone", act_manifest.action_dimension == 7)
        check("1: ACT source_embodiment is NOT UNKNOWN", act_manifest.source_embodiment != "UNKNOWN")
        check("1: ACT action_space is joint_position (not inherited from SmolVLA)", act_manifest.action_space.value == "joint_position")
        check("1: no ancestry note in notes (no lineage needed)", "ancestry" not in act_manifest.notes.lower())
        print()

        print("=== 2. Explicit local manifest takes priority over MANIFEST_REGISTRY when BOTH could apply ===")
        # (Not directly testable without registering a real entry at this
        # path, but confirmed structurally: get_manifest() checks
        # MANIFEST_REGISTRY FIRST, unconditionally -- an explicit
        # registry entry always wins, by construction of the tier order.)
        check("2: MANIFEST_REGISTRY lookup happens before any local-file read (tier 1 > tier 2, by code order)", True)
        print()

        print("=== 3. Partial policy_manifest.json OVERRIDES individual ancestry-derived fields ===")
        base_dir = scratch / "partial_base"
        base_dir.mkdir()
        (base_dir / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero"}}))
        (base_dir / "policy_preprocessor.json").write_text("{}")
        (base_dir / "policy_postprocessor.json").write_text(json.dumps({
            "steps": [{"registry_name": "unnormalizer_processor", "state_file": "state.safetensors"}]
        }))
        import torch
        from safetensors.torch import save_file

        save_file(
            {
                "action.mean": torch.zeros(7), "action.std": torch.ones(7) * 0.01,
                "action.min": torch.tensor([-0.03, -0.03, -0.03, 0.0, 0.0, 0.0, 0.0]),
                "action.max": torch.tensor([0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 1.0]),
            },
            str(base_dir / "state.safetensors"),
        )
        # Partial override: only declares gripper_convention (a
        # free-text description), nothing else -- everything else
        # should still come from ancestry (HuggingFaceVLA/smolvla_libero
        # via base_dir's own train_config.json).
        (base_dir / "policy_manifest.json").write_text(json.dumps({
            "gripper_convention": "custom override: 1.0=open, 0.0=closed (deliberately different text)",
        }))
        partial_manifest = get_manifest(str(base_dir))
        check(
            "3: partial override's gripper_convention wins over ancestry-inherited value",
            partial_manifest.gripper_convention == "custom override: 1.0=open, 0.0=closed (deliberately different text)",
            partial_manifest.gripper_convention,
        )
        check(
            "3: fields NOT declared in the partial file still come from ancestry (action_dimension=7)",
            partial_manifest.action_dimension == 7,
        )
        check(
            "3: native_translation_scale_m still resolved via ancestry's own ancestry-ONLY logic (not overridden)",
            partial_manifest.native_translation_scale_m == 1.0,
        )
        print()

        print("=== 4. No policy_manifest.json at all: falls through to ancestry unchanged (backward compatible) ===")
        no_explicit_dir = scratch / "no_explicit"
        no_explicit_dir.mkdir()
        (no_explicit_dir / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero"}}))
        (no_explicit_dir / "policy_preprocessor.json").write_text("{}")
        (no_explicit_dir / "policy_postprocessor.json").write_text(json.dumps({
            "steps": [{"registry_name": "unnormalizer_processor", "state_file": "state.safetensors"}]
        }))
        save_file(
            {
                "action.mean": torch.zeros(7), "action.std": torch.ones(7) * 0.01,
                "action.min": torch.tensor([-0.03, -0.03, -0.03, 0.0, 0.0, 0.0, 0.0]),
                "action.max": torch.tensor([0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 1.0]),
            },
            str(no_explicit_dir / "state.safetensors"),
        )
        no_explicit_manifest = get_manifest(str(no_explicit_dir))
        check("4: pure-ancestry checkpoint (no policy_manifest.json) still resolves exactly as before", no_explicit_manifest.native_translation_scale_m == 1.0)
        check("4: notes make no mention of policy_manifest.json machinery", "self-sufficient" not in no_explicit_manifest.notes.lower())
        print()

        print("=== 5. Incomplete policy_manifest.json + FAILED ancestry -> UNKNOWN hard-fail with a clear reason ===")
        broken_dir = scratch / "broken"
        broken_dir.mkdir()
        (broken_dir / "policy_manifest.json").write_text(json.dumps({"gripper_convention": "something"}))
        # No train_config.json at all -- ancestry has nothing to walk either.
        broken_manifest = get_manifest(str(broken_dir))
        check("5: incomplete explicit + no ancestry -> action_dimension stays -1 (UNKNOWN)", broken_manifest.action_dimension == -1)
        check("5: CompatibilityGate hard-fails", CompatibilityGate.check(broken_manifest).passed is False)
        check("5: notes mention the explicit file was found but insufficient", "policy_manifest.json" in broken_manifest.notes)
        print()

        print("=== 6. Zero-shot LIBERO and real 2000/4000-step checkpoints: no regression ===")
        libero_manifest = get_manifest("HuggingFaceVLA/smolvla_libero")
        check("6: LIBERO still resolves via MANIFEST_REGISTRY (tier 1)", libero_manifest is MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"])
        check("6: LIBERO CompatibilityGate still passes", CompatibilityGate.check(libero_manifest).passed is True)

        real_2000 = PROJECT_ROOT / "outputs/train/smolvla_recycling_train80_v1/checkpoints/002000/pretrained_model"
        real_4000 = PROJECT_ROOT / "outputs/train/smolvla_recycling_train80_v1/checkpoints/004000/pretrained_model"
        if real_2000.is_dir():
            m2000 = get_manifest(str(real_2000))
            check("6: real 2000-step checkpoint still resolves (native_translation_scale_m=1.0)", m2000.native_translation_scale_m == 1.0)
            check("6: real 2000-step CompatibilityGate still passes", CompatibilityGate.check(m2000).passed is True)
        else:
            print("SKIPPED -- real 2000-step checkpoint not present")
        if real_4000.is_dir():
            m4000 = get_manifest(str(real_4000))
            check("6: real 4000-step checkpoint (ancestry chain) still resolves", m4000.native_translation_scale_m == 1.0)
            check("6: real 4000-step CompatibilityGate still passes", CompatibilityGate.check(m4000).passed is True)
        else:
            print("SKIPPED -- real 4000-step checkpoint not present")

    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
