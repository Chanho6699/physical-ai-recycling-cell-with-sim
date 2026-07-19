"""Regression tests for policy_semantics/manifest.py's local-checkpoint
ancestry-chain resolution (see this task's chat report): a --resume
continuation's own train_config.json records policy.pretrained_path as
the checkpoint it actually resumed from (a local path), not the
original registered Hub base -- so a single-hop lookup (the code before
this fix) correctly resolves a direct --policy.path fine-tune but fails
for any --resume'd checkpoint, however many generations deep.
_resolve_base_manifest_via_ancestry() now follows that chain (never
hardcoding a specific depth/checkpoint number/directory name) until it
reaches a registered MANIFEST_REGISTRY entry, with explicit,
diagnosable failure (never a silent guess) on cycles, missing/broken
parent configs, or exceeding the depth bound.

Builds real, synthetic checkpoint-directory trees under a temp
subdirectory of THIS project's own root (not /tmp) specifically so
relative policy.pretrained_path chains resolve exactly the way a real
checkpoint's would (see _normalize_checkpoint_path()'s own docstring:
relative paths resolve against the project root, matching every other
benchmark/*.py script's existing convention) -- deleted again at the
end of the run either way.

Run: .venv-vla/bin/python -m benchmark.test_manifest_ancestry_resolution
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import (
    MANIFEST_REGISTRY,
    _AncestryResolutionError,
    _resolve_base_manifest_via_ancestry,
    _resolve_finetuned_manifest,
    get_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DIR_NAME = "_test_manifest_ancestry_scratch"  # relative to PROJECT_ROOT -- cleaned up at the end
LOCAL_CHECKPOINT = str(
    PROJECT_ROOT / "outputs/train/smolvla_recycling_train80_v1/checkpoints/002000/pretrained_model"
)
LOCAL_CHECKPOINT_4000 = str(
    PROJECT_ROOT / "outputs/train/smolvla_recycling_train80_v1/checkpoints/004000/pretrained_model"
)

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _write_own_pipeline_postprocessor(directory: Path) -> None:
    """Own-pipeline-scale stats (action.max[0:3] near DEFAULT_MAX_STEP_SIZE,
    gripper dim (0,1)) -- matches this project's own real fine-tuned
    checkpoints, so _verify_translation_scale_matches_own_pipeline()
    genuinely corroborates (not just structurally passes) for these
    synthetic fixtures too."""
    (directory / "policy_postprocessor.json").write_text(json.dumps({
        "steps": [{"registry_name": "unnormalizer_processor", "state_file": "state.safetensors"}]
    }))
    save_file(
        {
            "action.mean": torch.zeros(7),
            "action.std": torch.ones(7) * 0.01,
            "action.min": torch.tensor([-0.03, -0.03, -0.03, 0.0, 0.0, 0.0, 0.0]),
            "action.max": torch.tensor([0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 1.0]),
        },
        str(directory / "state.safetensors"),
    )
    (directory / "policy_preprocessor.json").write_text("{}")


def _make_checkpoint(root: Path, name: str, pretrained_path) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": pretrained_path, "type": "smolvla"}}))
    _write_own_pipeline_postprocessor(directory)
    return directory


def main() -> None:
    scratch = PROJECT_ROOT / SCRATCH_DIR_NAME
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    try:
        print("=== A. Base model directly referenced (chain length 1) ===")
        ckpt_a = _make_checkpoint(scratch, "a_direct", "HuggingFaceVLA/smolvla_libero")
        manifest_a, reason_a = _resolve_finetuned_manifest(str(ckpt_a))
        check("A: resolves successfully", manifest_a is not None, str(reason_a))
        check("A: reason is None on success", reason_a is None)
        if manifest_a is not None:
            check("A: native_translation_scale_m == 1.0", manifest_a.native_translation_scale_m == 1.0)
            check("A: CompatibilityGate passes", CompatibilityGate.check(manifest_a).passed is True)
        print()

        print("=== B. base -> 2000 -> 4000 (2-hop chain, relative paths) ===")
        ckpt_b2000 = _make_checkpoint(scratch, "b_2000", "HuggingFaceVLA/smolvla_libero")
        rel_b2000 = str(ckpt_b2000.relative_to(PROJECT_ROOT))
        ckpt_b4000 = _make_checkpoint(scratch, "b_4000", rel_b2000)
        manifest_b, reason_b = _resolve_finetuned_manifest(str(ckpt_b4000))
        check("B: resolves successfully", manifest_b is not None, str(reason_b))
        if manifest_b is not None:
            check("B: native_translation_scale_m == 1.0", manifest_b.native_translation_scale_m == 1.0)
            check("B: CompatibilityGate passes", CompatibilityGate.check(manifest_b).passed is True)
            check("B: notes mention 2-hop ancestry chain", str(ckpt_b2000) in manifest_b.notes or rel_b2000 in manifest_b.notes)
        print()

        print("=== C. base -> 1000 -> 2000 -> 4000 (3-hop chain) ===")
        ckpt_c1000 = _make_checkpoint(scratch, "c_1000", "HuggingFaceVLA/smolvla_libero")
        ckpt_c2000 = _make_checkpoint(scratch, "c_2000", str(ckpt_c1000))
        ckpt_c4000 = _make_checkpoint(scratch, "c_4000", str(ckpt_c2000))
        manifest_c, reason_c = _resolve_finetuned_manifest(str(ckpt_c4000))
        check("C: resolves successfully", manifest_c is not None, str(reason_c))
        if manifest_c is not None:
            check("C: CompatibilityGate passes", CompatibilityGate.check(manifest_c).passed is True)
        print()

        print("=== D. Relative-path chain (explicit) ===")
        ckpt_d_base = _make_checkpoint(scratch, "d_base", "HuggingFaceVLA/smolvla_libero")
        rel_d_base = str(ckpt_d_base.relative_to(PROJECT_ROOT))
        ckpt_d_child = _make_checkpoint(scratch, "d_child", rel_d_base)
        _, chain_d = _resolve_base_manifest_via_ancestry(str(ckpt_d_child))
        check("D: relative-path parent resolved correctly", chain_d[-1] == "HuggingFaceVLA/smolvla_libero", str(chain_d))
        print()

        print("=== E. Absolute-path chain (explicit) ===")
        ckpt_e_base = _make_checkpoint(scratch, "e_base", "HuggingFaceVLA/smolvla_libero")
        ckpt_e_child = _make_checkpoint(scratch, "e_child", str(ckpt_e_base.resolve()))
        _, chain_e = _resolve_base_manifest_via_ancestry(str(ckpt_e_child))
        check("E: absolute-path parent resolved correctly", chain_e[-1] == "HuggingFaceVLA/smolvla_libero", str(chain_e))
        print()

        print("=== F. Cycle A -> B -> A ===")
        ckpt_f_a = scratch / "f_a"
        ckpt_f_b = scratch / "f_b"
        ckpt_f_a.mkdir()
        ckpt_f_b.mkdir()
        (ckpt_f_a / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": str(ckpt_f_b)}}))
        (ckpt_f_b / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": str(ckpt_f_a)}}))
        raised = False
        message = ""
        try:
            _resolve_base_manifest_via_ancestry(str(ckpt_f_a))
        except _AncestryResolutionError as exc:
            raised = True
            message = str(exc)
        check("F: cycle raises _AncestryResolutionError", raised, message)
        check("F: error message mentions 'cycle'", "cycle" in message.lower())
        manifest_f, reason_f = _resolve_finetuned_manifest(str(ckpt_f_a))
        check("F: _resolve_finetuned_manifest returns (None, reason) on cycle", manifest_f is None and reason_f is not None)
        gate_f = CompatibilityGate.check(get_manifest(str(ckpt_f_a)))
        check("F: get_manifest() never raises, CompatibilityGate hard-fails", gate_f.passed is False)
        check("F: get_manifest().notes surfaces the cycle reason", "cycle" in get_manifest(str(ckpt_f_a)).notes.lower())
        print()

        print("=== G. Missing parent config ===")
        ckpt_g_child = scratch / "g_child"
        ckpt_g_child.mkdir()
        missing_parent = scratch / "g_parent_does_not_exist"
        (ckpt_g_child / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": str(missing_parent)}}))
        raised_g = False
        try:
            _resolve_base_manifest_via_ancestry(str(ckpt_g_child))
        except _AncestryResolutionError as exc:
            raised_g = True
            message_g = str(exc)
        check("G: missing parent directory raises with diagnosable message", raised_g, message_g if raised_g else "")
        if raised_g:
            check("G: message names the missing path", str(missing_parent) in message_g)
        print()

        print("=== H. Non-existent parent path (never a directory at all) ===")
        ckpt_h_child = scratch / "h_child"
        ckpt_h_child.mkdir()
        (ckpt_h_child / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": "totally/bogus/nonexistent/path"}}))
        manifest_h, reason_h = _resolve_finetuned_manifest(str(ckpt_h_child))
        check("H: resolves to (None, reason) for a bogus parent path", manifest_h is None and reason_h is not None, str(reason_h))
        print()

        print("=== I. Max ancestry depth exceeded ===")
        prev = "HuggingFaceVLA/smolvla_libero"
        chain_dirs = []
        # _MAX_ANCESTRY_DEPTH is 8 -- build 10 hops so the walk must exceed it.
        for i in range(10):
            d = _make_checkpoint(scratch, f"i_hop_{i}", prev)
            chain_dirs.append(d)
            prev = str(d)
        raised_i = False
        try:
            _resolve_base_manifest_via_ancestry(str(chain_dirs[-1]))
        except _AncestryResolutionError as exc:
            raised_i = True
            message_i = str(exc)
        check("I: exceeding max depth raises with diagnosable message", raised_i, message_i if raised_i else "")
        if raised_i:
            check("I: message mentions depth/exceeded", "depth" in message_i.lower())
        print()

        print("=== J. Current checkpoint's own metadata takes priority over parent's ===")
        ckpt_j_base = _make_checkpoint(scratch, "j_base", "HuggingFaceVLA/smolvla_libero")
        ckpt_j_child_dir = scratch / "j_child"
        ckpt_j_child_dir.mkdir()
        (ckpt_j_child_dir / "train_config.json").write_text(json.dumps({"policy": {"pretrained_path": str(ckpt_j_base)}}))
        (ckpt_j_child_dir / "policy_preprocessor.json").write_text("{}")
        (ckpt_j_child_dir / "policy_postprocessor.json").write_text(json.dumps({
            "steps": [{"registry_name": "unnormalizer_processor", "state_file": "state.safetensors"}]
        }))
        # Child's OWN gripper native range differs from the parent's (0,1) --
        # e.g. (0, 2) -- to prove it's read from the child, not inherited.
        save_file(
            {
                "action.mean": torch.zeros(7),
                "action.std": torch.ones(7) * 0.01,
                "action.min": torch.tensor([-0.03, -0.03, -0.03, 0.0, 0.0, 0.0, 0.0]),
                "action.max": torch.tensor([0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 2.0]),
            },
            str(ckpt_j_child_dir / "state.safetensors"),
        )
        manifest_j_base, _ = _resolve_finetuned_manifest(str(ckpt_j_base))
        manifest_j_child, _ = _resolve_finetuned_manifest(str(ckpt_j_child_dir))
        check(
            "J: child's own gripper range (0,2) differs from what a blind parent-copy would give (0,1)",
            manifest_j_child is not None and manifest_j_child.native_gripper_range == (0.0, 2.0),
            str(manifest_j_child.native_gripper_range if manifest_j_child else None),
        )
        check(
            "J: parent (j_base) itself is unaffected, still (0,1)",
            manifest_j_base is not None and manifest_j_base.native_gripper_range == (0.0, 1.0),
        )
        print()

        print("=== K. Zero-shot LIBERO: no regression ===")
        libero_manifest = get_manifest("HuggingFaceVLA/smolvla_libero")
        check("K: still the exact registered manifest object", libero_manifest is MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"])
        check("K: CompatibilityGate still passes", CompatibilityGate.check(libero_manifest).passed is True)
        print()

        print("=== L. Existing 2000-step checkpoint: no regression ===")
        if Path(LOCAL_CHECKPOINT).is_dir():
            manifest_2000 = get_manifest(LOCAL_CHECKPOINT)
            check("L: 2000-step manifest still resolves", manifest_2000.native_translation_scale_m == 1.0)
            check("L: 2000-step CompatibilityGate still passes", CompatibilityGate.check(manifest_2000).passed is True)
        else:
            print("SKIPPED -- local 2000-step checkpoint not present")
        print()

        print("=== Bonus: real 4000-step checkpoint (the production blocker this fix addresses) ===")
        if Path(LOCAL_CHECKPOINT_4000).is_dir():
            manifest_4000 = get_manifest(LOCAL_CHECKPOINT_4000)
            check("4000-step: native_translation_scale_m == 1.0", manifest_4000.native_translation_scale_m == 1.0)
            check("4000-step: native_gripper_range == (0.0, 1.0)", manifest_4000.native_gripper_range == (0.0, 1.0))
            gate_4000 = CompatibilityGate.check(manifest_4000)
            check("4000-step: CompatibilityGate passes", gate_4000.passed is True, str(gate_4000.reasons))
        else:
            print("SKIPPED -- local 4000-step checkpoint not present")

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
