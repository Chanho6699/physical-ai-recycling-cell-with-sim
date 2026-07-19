"""Regression tests for the gripper native-range generalization fix (see
this task's chat report): PolicyManifest.native_gripper_range/
native_gripper_min_means/native_gripper_max_means,
SmolVLALiberoActionAdapter._decode_gripper()'s manifest-driven formula
(replacing the old hardcoded (1 - raw_gripper) / 2), and
_resolve_finetuned_manifest()'s fresh-per-checkpoint numeric-range
extraction from a local checkpoint's own postprocessor safetensors.

Requires the local checkpoint produced by an earlier fine-tuning smoke
test: outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model
(skips the checkpoint-dependent sections with a clear message if absent).

Run: .venv-vla/bin/python -m benchmark.test_gripper_native_range_semantics
"""

import json
import shutil
import tempfile
from pathlib import Path

from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import MANIFEST_REGISTRY, UNKNOWN, get_manifest
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


def _decode_gripper(raw_gripper, manifest):
    adapter = SmolVLALiberoActionAdapter()
    native = NativePolicyAction(
        values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, raw_gripper],
        source_policy="test", postprocessor_used=True, metadata={},
    )
    return adapter.decode(native, manifest, context={"degraded_input": False})


def main() -> None:
    print("=== 1. Original LIBERO checkpoint: exact expected values (regression) ===")
    libero_manifest = MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"]
    check("LIBERO manifest declares native_gripper_range == (-1.0, 1.0)", libero_manifest.native_gripper_range == (-1.0, 1.0))
    check("LIBERO manifest declares min_means=open, max_means=close", libero_manifest.native_gripper_min_means == "open" and libero_manifest.native_gripper_max_means == "close")

    cmd = _decode_gripper(-1.0, libero_manifest)
    check("LIBERO raw=-1.0 -> gripper_opening_01=1.0 (open)", cmd is not None and abs(cmd.gripper_opening_01 - 1.0) < 1e-9, str(cmd))
    cmd = _decode_gripper(0.0, libero_manifest)
    check("LIBERO raw=0.0 -> gripper_opening_01=0.5", cmd is not None and abs(cmd.gripper_opening_01 - 0.5) < 1e-9, str(cmd))
    cmd = _decode_gripper(1.0, libero_manifest)
    check("LIBERO raw=+1.0 -> gripper_opening_01=0.0 (closed)", cmd is not None and abs(cmd.gripper_opening_01 - 0.0) < 1e-9, str(cmd))
    print()

    print("=== 2. Fine-tuned recycling checkpoint manifest: exact expected values ===")
    if not Path(LOCAL_CHECKPOINT).is_dir():
        print(f"SKIPPED -- {LOCAL_CHECKPOINT} not present")
        ft_manifest = None
    else:
        ft_manifest = get_manifest(LOCAL_CHECKPOINT)
        check("fine-tuned manifest native_gripper_range == (0.0, 1.0) (extracted fresh, not inherited)", ft_manifest.native_gripper_range == (0.0, 1.0), str(ft_manifest.native_gripper_range))
        check("fine-tuned manifest min_means=open, max_means=close (inherited direction)", ft_manifest.native_gripper_min_means == "open" and ft_manifest.native_gripper_max_means == "close")

        cmd = _decode_gripper(0.0, ft_manifest)
        check("fine-tuned raw=0.0 -> gripper_opening_01=1.0 (open)", cmd is not None and abs(cmd.gripper_opening_01 - 1.0) < 1e-9, str(cmd))
        cmd = _decode_gripper(0.5, ft_manifest)
        check("fine-tuned raw=0.5 -> gripper_opening_01=0.5", cmd is not None and abs(cmd.gripper_opening_01 - 0.5) < 1e-9, str(cmd))
        cmd = _decode_gripper(1.0, ft_manifest)
        check("fine-tuned raw=1.0 -> gripper_opening_01=0.0 (closed)", cmd is not None and abs(cmd.gripper_opening_01 - 0.0) < 1e-9, str(cmd))

        # The exact real values measured in this task's chat report BEFORE this fix
        # (postprocessed gripper ~0.92-0.94, all decoding to "close" under the old
        # hardcoded formula) now decode correctly under the fine-tuned checkpoint's
        # OWN [0,1] range -- 0.93 is genuinely close to 1.0="closed" in this scale,
        # so it correctly STILL decodes close, but via the right mechanism (not by
        # accident of both scales agreeing at the closed end, as before).
        cmd = _decode_gripper(0.93, ft_manifest)
        check("fine-tuned raw=0.93 -> gripper_opening_01 near 0.07 (correctly still 'close', by the RIGHT mechanism)", cmd is not None and abs(cmd.gripper_opening_01 - 0.07) < 0.01, str(cmd))
    print()

    print("=== 3. Translation/rotation dims 0-5 are byte-identical before/after this fix ===")
    manifest_for_check = libero_manifest
    native = NativePolicyAction(
        values=[0.3, -0.2, 0.1, 0.05, -0.05, 0.02, -1.0],  # gripper=-1 (open) so we isolate translation/rotation
        source_policy="test", postprocessor_used=True, metadata={},
    )
    adapter = SmolVLALiberoActionAdapter()
    cmd = adapter.decode(native, manifest_for_check, context={"degraded_input": False})
    # Old formula: clip to [-1,1] (no-op here) * TRANSLATION_SCALE_M(0.05)/ROTATION_SCALE_RAD(0.5)
    expected_translation = (0.3 * 0.05, -0.2 * 0.05, 0.1 * 0.05)
    expected_rotation = (0.05 * 0.5, -0.05 * 0.5, 0.02 * 0.5)
    check("translation_m unchanged by this fix", all(abs(a - b) < 1e-9 for a, b in zip(cmd.translation_m, expected_translation)), str(cmd.translation_m))
    check("rotation_axis_angle_rad unchanged by this fix", all(abs(a - b) < 1e-9 for a, b in zip(cmd.rotation_axis_angle_rad, expected_rotation)), str(cmd.rotation_axis_angle_rad))
    print()

    print("=== 4. NaN/Inf gripper values rejected ===")
    check("raw=NaN -> decode() returns None", _decode_gripper(float("nan"), libero_manifest) is None)
    check("raw=Inf -> decode() returns None", _decode_gripper(float("inf"), libero_manifest) is None)
    check("raw=-Inf -> decode() returns None", _decode_gripper(float("-inf"), libero_manifest) is None)
    print()

    print("=== 5. UNKNOWN/invalid native range hard-fails (never guesses) ===")
    from dataclasses import replace as dc_replace

    unknown_manifest = dc_replace(libero_manifest, native_gripper_range=None, native_gripper_min_means=UNKNOWN, native_gripper_max_means=UNKNOWN)
    check("native_gripper_range=None -> decode() returns None", _decode_gripper(0.0, unknown_manifest) is None)

    invalid_range_manifest = dc_replace(libero_manifest, native_gripper_range=(1.0, -1.0))  # min >= max
    check("native_gripper_range with min>=max -> decode() returns None", _decode_gripper(0.0, invalid_range_manifest) is None)

    same_polarity_manifest = dc_replace(libero_manifest, native_gripper_min_means="open", native_gripper_max_means="open")
    check("min_means == max_means (ambiguous) -> decode() returns None", _decode_gripper(0.0, same_polarity_manifest) is None)

    check("CompatibilityGate fails for the None-range manifest", CompatibilityGate.check(unknown_manifest).passed is False)
    print()

    print("=== 6. Original checkpoint's overall CompatibilityGate result: no regression ===")
    result = CompatibilityGate.check(libero_manifest)
    check("HuggingFaceVLA/smolvla_libero still passes CompatibilityGate overall", result.passed is True, str(result.reasons))
    check("gripper_native_range_known check present and passing", result.checks.get("gripper_native_range_known", {}).get("passed") is True)
    print()

    if ft_manifest is not None:
        print("=== 7. Fine-tuned checkpoint's overall CompatibilityGate result ===")
        result = CompatibilityGate.check(ft_manifest)
        check("local fine-tuned checkpoint passes CompatibilityGate overall", result.passed is True, str(result.reasons))
        print()

    print("=== 8. Garbage/unregistered local path still hard-fails (regression) ===")
    garbage_manifest = get_manifest("/definitely/does/not/exist/anywhere")
    check("garbage path -> native_gripper_range is None", garbage_manifest.native_gripper_range is None)
    check("garbage path -> CompatibilityGate refuses it", CompatibilityGate.check(garbage_manifest).passed is False)
    print()

    print("=== 9. Local checkpoint gripper-range extraction: robust to missing postprocessor ===")
    scratch = Path(tempfile.mkdtemp(prefix="gripper_range_extraction_test_"))
    try:
        no_postprocessor_dir = scratch / "no_postprocessor"
        no_postprocessor_dir.mkdir()
        (no_postprocessor_dir / "train_config.json").write_text(
            json.dumps({"policy": {"pretrained_path": "HuggingFaceVLA/smolvla_libero", "type": "smolvla"}})
        )
        m = get_manifest(str(no_postprocessor_dir))
        check(
            "local checkpoint with train_config.json but NO postprocessor file -> native_gripper_range is None (UNKNOWN), not guessed/inherited",
            m.native_gripper_range is None and m.native_gripper_min_means == UNKNOWN,
            f"got range={m.native_gripper_range}, min_means={m.native_gripper_min_means}",
        )
        check("CompatibilityGate refuses this manifest specifically on the gripper check", CompatibilityGate.check(m).checks["gripper_native_range_known"]["passed"] is False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print()

    print("=== 10. No production file depends on this test module ===")
    hits = []
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy", "action_adapter"]
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "test_gripper_native_range_semantics" in text:
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
