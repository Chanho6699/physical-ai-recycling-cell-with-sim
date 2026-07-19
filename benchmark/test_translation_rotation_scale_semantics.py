"""Regression tests for the translation/rotation native-scale
generalization fix (see this task's chat report):
PolicyManifest.native_translation_scale_m/native_rotation_scale_rad/
native_action_clip_range, SmolVLALiberoActionAdapter
._decode_translation_rotation()'s manifest-driven formula (replacing
the old hardcoded TRANSLATION_SCALE_M(0.05)/ROTATION_SCALE_RAD(0.5)/
[-1,1]-clip applied uniformly to every checkpoint), and
_resolve_finetuned_manifest()'s _verify_translation_scale_matches_own_pipeline()
corroboration (not blind inheritance) of a local checkpoint's native
translation scale.

Requires the local checkpoint produced by an earlier fine-tuning smoke
test: outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model
(skips the checkpoint-dependent sections with a clear message if absent).

Does not touch gripper decode logic or its own tests (see
test_gripper_native_range_semantics.py) -- covers translation/rotation
only, per this task's explicit scope.

Run: .venv-vla/bin/python -m benchmark.test_translation_rotation_scale_semantics
"""

import json
import shutil
import tempfile
from pathlib import Path

from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import MANIFEST_REGISTRY, get_manifest
from policy_semantics.native_policy_action import NativePolicyAction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT = str(
    PROJECT_ROOT / "outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model"
)
_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _decode(values7, manifest):
    adapter = SmolVLALiberoActionAdapter()
    native = NativePolicyAction(values=list(values7), source_policy="test", postprocessor_used=True, metadata={})
    return adapter.decode(native, manifest, context={"degraded_input": False})


def main() -> None:
    print("=== 1. Original LIBERO checkpoint: exact expected values (regression) ===")
    libero_manifest = MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"]
    check("LIBERO manifest declares native_translation_scale_m == 0.05", libero_manifest.native_translation_scale_m == 0.05)
    check("LIBERO manifest declares native_rotation_scale_rad == 0.5", libero_manifest.native_rotation_scale_rad == 0.5)
    check("LIBERO manifest declares native_action_clip_range == (-1.0, 1.0)", libero_manifest.native_action_clip_range == (-1.0, 1.0))

    # Real measured values from this task's diagnostic (far_center_right observation).
    cmd = _decode([-0.8501, -0.5970, -0.9375, 0.0, 0.0, 0.0, -0.87], libero_manifest)
    check(
        "LIBERO: postprocessed [-0.8501,-0.5970,-0.9375] -> translation_m == same * 0.05",
        cmd is not None and all(abs(a - b) < 1e-9 for a, b in zip(cmd.translation_m, (-0.8501 * 0.05, -0.5970 * 0.05, -0.9375 * 0.05))),
        str(cmd.translation_m) if cmd else "None",
    )

    # Clipping still applies for out-of-[-1,1] values (unchanged behavior).
    cmd = _decode([2.0, -2.0, 0.5, 0.0, 0.0, 0.0, -0.87], libero_manifest)
    check(
        "LIBERO: out-of-range translation clipped to [-1,1] BEFORE scaling (unchanged behavior)",
        cmd is not None and abs(cmd.translation_m[0] - 0.05) < 1e-9 and abs(cmd.translation_m[1] - (-0.05)) < 1e-9,
        str(cmd.translation_m) if cmd else "None",
    )
    print()

    print("=== 2. Fine-tuned recycling checkpoint: exact expected values (the confirmed fix) ===")
    if not Path(LOCAL_CHECKPOINT).is_dir():
        print(f"SKIPPED -- {LOCAL_CHECKPOINT} not present")
        ft_manifest = None
    else:
        ft_manifest = get_manifest(LOCAL_CHECKPOINT)
        check("fine-tuned manifest native_translation_scale_m == 1.0 (identity -- already real meters)", ft_manifest.native_translation_scale_m == 1.0)
        check("fine-tuned manifest native_rotation_scale_rad == 1.0 (identity)", ft_manifest.native_rotation_scale_rad == 1.0)
        check("fine-tuned manifest native_action_clip_range == (-inf, inf) (no native clip)", ft_manifest.native_action_clip_range == (float("-inf"), float("inf")))

        # Real measured values from this task's diagnostic (far_center_right observation).
        cmd = _decode([-0.0081, 0.0091, -0.0334, 0.0, 0.0, 0.0, 0.92], ft_manifest)
        check(
            "fine-tuned: postprocessed [-0.0081,+0.0091,-0.0334] -> translation_m UNCHANGED (no 20x shrink)",
            cmd is not None and all(abs(a - b) < 1e-9 for a, b in zip(cmd.translation_m, (-0.0081, 0.0091, -0.0334))),
            str(cmd.translation_m) if cmd else "None",
        )
        check(
            "fine-tuned: translation_m is NOT the old (buggy) 20x-too-small value",
            cmd is not None and abs(cmd.translation_m[0] - (-0.0081 * 0.05)) > 1e-6,
        )
    print()

    print("=== 3. Gripper dimension untouched by this fix (regression against the OTHER task's fix) ===")
    if ft_manifest is not None:
        cmd_a = _decode([-0.0081, 0.0091, -0.0334, 0.0, 0.0, 0.0, 0.0], ft_manifest)
        cmd_b = _decode([-0.0081, 0.0091, -0.0334, 0.0, 0.0, 0.0, 1.0], ft_manifest)
        check("fine-tuned: gripper=0.0 -> gripper_opening_01=1.0 (open, unchanged from prior fix)", cmd_a is not None and abs(cmd_a.gripper_opening_01 - 1.0) < 1e-9)
        check("fine-tuned: gripper=1.0 -> gripper_opening_01=0.0 (closed, unchanged from prior fix)", cmd_b is not None and abs(cmd_b.gripper_opening_01 - 0.0) < 1e-9)
    print()

    print("=== 4. NaN/Inf translation/rotation values rejected ===")
    check("translation NaN -> decode() returns None", _decode([float("nan"), 0, 0, 0, 0, 0, -0.87], libero_manifest) is None)
    check("rotation Inf -> decode() returns None", _decode([0, 0, 0, float("inf"), 0, 0, -0.87], libero_manifest) is None)
    print()

    print("=== 5. UNKNOWN/invalid native scale hard-fails (never guesses) ===")
    from dataclasses import replace as dc_replace

    unknown_manifest = dc_replace(libero_manifest, native_translation_scale_m=None, native_rotation_scale_rad=None)
    check("native_translation_scale_m=None -> decode() returns None", _decode([0.1, 0.1, 0.1, 0, 0, 0, -0.87], unknown_manifest) is None)
    check("CompatibilityGate fails for this manifest", CompatibilityGate.check(unknown_manifest).passed is False)

    invalid_clip_manifest = dc_replace(libero_manifest, native_action_clip_range=(1.0, -1.0))  # min >= max
    check("native_action_clip_range with min>=max -> decode() returns None", _decode([0.1, 0.1, 0.1, 0, 0, 0, -0.87], invalid_clip_manifest) is None)
    check("CompatibilityGate fails for invalid clip range", CompatibilityGate.check(invalid_clip_manifest).passed is False)
    print()

    print("=== 6. Original checkpoint's overall CompatibilityGate result: no regression ===")
    result = CompatibilityGate.check(libero_manifest)
    check("HuggingFaceVLA/smolvla_libero still passes CompatibilityGate overall", result.passed is True, str(result.reasons))
    check("translation_rotation_scale_known check present and passing", result.checks.get("translation_rotation_scale_known", {}).get("passed") is True)
    print()

    if ft_manifest is not None:
        print("=== 7. Fine-tuned checkpoint's overall CompatibilityGate result ===")
        result = CompatibilityGate.check(ft_manifest)
        check("local fine-tuned checkpoint passes CompatibilityGate overall", result.passed is True, str(result.reasons))
        print()

    print("=== 8. _verify_translation_scale_matches_own_pipeline(): corroboration, not blind inheritance ===")
    from policy_semantics.manifest import _verify_translation_scale_matches_own_pipeline

    if Path(LOCAL_CHECKPOINT).is_dir():
        check("real fine-tuned checkpoint's own postprocessor stats DO match DEFAULT_MAX_STEP_SIZE", _verify_translation_scale_matches_own_pipeline(Path(LOCAL_CHECKPOINT)) is True)

    scratch = Path(tempfile.mkdtemp(prefix="translation_scale_test_"))
    try:
        # Synthetic checkpoint whose postprocessor reports a LIBERO-like
        # native range (max ~0.9375) -- must NOT be treated as "our own
        # pipeline" despite pointing at a registered base.
        libero_scale_dir = scratch / "libero_scale_finetune"
        libero_scale_dir.mkdir()
        (libero_scale_dir / "train_config.json").write_text(
            json.dumps({"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero", "type": "smolvla"}})
        )
        (libero_scale_dir / "policy_preprocessor.json").write_text("{}")
        (libero_scale_dir / "policy_postprocessor.json").write_text(json.dumps({
            "steps": [{"registry_name": "unnormalizer_processor", "state_file": "state.safetensors"}]
        }))
        from safetensors.torch import save_file
        import torch

        save_file(
            {
                "action.mean": torch.zeros(7),
                "action.std": torch.ones(7),
                "action.min": torch.tensor([-0.9375, -0.9375, -0.9375, -0.25, -0.375, -0.375, -1.0]),
                "action.max": torch.tensor([0.9375, 0.9375, 0.9375, 0.25, 0.375, 0.375, 1.0]),
            },
            str(libero_scale_dir / "state.safetensors"),
        )
        check(
            "synthetic LIBERO-scale-native local checkpoint: correctly NOT verified as this project's own real-meter pipeline",
            _verify_translation_scale_matches_own_pipeline(libero_scale_dir) is False,
        )
        derived_manifest = get_manifest(str(libero_scale_dir))
        check(
            "synthetic LIBERO-scale-native checkpoint: native_translation_scale_m stays UNKNOWN (not silently inherited)",
            derived_manifest.native_translation_scale_m is None,
            str(derived_manifest.native_translation_scale_m),
        )
        check(
            "synthetic LIBERO-scale-native checkpoint: CompatibilityGate fails on translation_rotation_scale_known",
            CompatibilityGate.check(derived_manifest).checks["translation_rotation_scale_known"]["passed"] is False,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print()

    print("=== 9. No production file depends on this test module ===")
    hits = []
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy", "action_adapter"]
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "test_translation_rotation_scale_semantics" in text:
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    check("no production file imports/references this test module", len(hits) == 0, f"unexpected: {hits}")

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
