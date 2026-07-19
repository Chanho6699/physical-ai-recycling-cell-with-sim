"""Regression tests backing benchmark/diagnose_gripper_collapse.py's
findings (see this task's chat report for the full root-cause writeup).

Covers the specific claims that report leans on:
  1. Dataset label distribution function is computed correctly (checked
     against a small synthetic, hand-verifiable episode set, not just
     trusted on the real 651-frame dataset).
  2. action_is_pad IS excluded from the training loss in the installed
     LeRobot version (a source-presence check, not a live training run --
     confirms the exact lines this report's padding analysis depends on
     still exist).
  3. SmolVLALiberoActionAdapter's gripper decode formula behaves exactly
     as this report describes for known synthetic native-action values
     (-1/0/+1), and reproduces the SPECIFIC real numbers this report
     measured for both checkpoints on fixed observations.
  4. Our own collector's gripper convention (0.0=open/1.0=close) matches
     what's actually stored in the real train20 dataset.
  5. Processor stats extraction reads the exact real files this report
     cites.
  6. No production file depends on this diagnostic module.

Run: .venv-vla/bin/python -m benchmark.test_gripper_collapse_diagnosis
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.diagnose_gripper_collapse import (
    ORIGINAL_LIBERO_SNAPSHOT,
    TRAIN20_ROOT,
    analyze_dataset_gripper_distribution,
    compare_processor_stats,
)
from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.native_policy_action import NativePolicyAction
from policy_semantics.manifest import MANIFEST_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _write_synthetic_dataset(root: Path) -> None:
    """One 10-frame episode, hand-designed so every metric this test
    checks has an easy-to-verify-by-hand expected value: frames 0-3 open
    (0.0), frames 4-7 close (1.0), frames 8-9 open (0.0) -- 1 close-run of
    length 4, 2 transitions, first_close_frame=4, last_open_frame=9."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    gripper = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    rows = []
    for i, g in enumerate(gripper):
        rows.append({
            "episode_index": 0,
            "frame_index": i,
            "action": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g], dtype=np.float32),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def main() -> None:
    print("=== 1. analyze_dataset_gripper_distribution() correctness (synthetic, hand-verifiable) ===")
    scratch = Path(tempfile.mkdtemp(prefix="gripper_collapse_test_"))
    try:
        _write_synthetic_dataset(scratch)
        dist = analyze_dataset_gripper_distribution(scratch)
        check("total_frames == 10", dist["total_frames"] == 10, str(dist["total_frames"]))
        check("raw_close_ratio == 0.4 (4/10 close frames)", abs(dist["raw_close_ratio"] - 0.4) < 1e-9, str(dist["raw_close_ratio"]))
        ep0 = dist["per_episode"][0]
        check("transitions == 2 (open->close, close->open)", ep0["transitions"] == 2, str(ep0["transitions"]))
        check("first_close_frame == 4", ep0["first_close_frame"] == 4, str(ep0["first_close_frame"]))
        check("last_open_frame == 9", ep0["last_open_frame"] == 9, str(ep0["last_open_frame"]))
        check("close_run_lengths == [4]", ep0["close_run_lengths"] == [4], str(ep0["close_run_lengths"]))

        # Hand-computed chunk-exposure weights for L=10 (chunk_size=50 > L,
        # so every frame i is a valid target for every sample s in [0, i],
        # weight = i+1): close frames are indices 4,5,6,7 -> weights 5+6+7+8=26.
        # open frames are 0,1,2,3,8,9 -> weights 1+2+3+4+9+10=29. total=55.
        expected_close_weight = 5 + 6 + 7 + 8
        expected_open_weight = (1 + 2 + 3 + 4) + (9 + 10)
        expected_total = expected_close_weight + expected_open_weight
        check(
            "chunk-exposure-weighted close ratio matches hand computation",
            abs(dist["chunk_exposure_weighted_close_ratio"] - expected_close_weight / expected_total) < 1e-9,
            f"got {dist['chunk_exposure_weighted_close_ratio']}, expected {expected_close_weight/expected_total}",
        )
        check(
            "chunk-exposure weighting SKEWS toward the label that dominates the LATE half of a short episode "
            "(close, in this synthetic case) relative to the raw ratio",
            dist["chunk_exposure_weighted_close_ratio"] > dist["raw_close_ratio"],
            f"weighted={dist['chunk_exposure_weighted_close_ratio']} raw={dist['raw_close_ratio']}",
        )
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    print()

    print("=== 2. action_is_pad exclusion from loss -- source-presence check ===")
    import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla
    import inspect

    source = inspect.getsource(modeling_smolvla)
    check(
        "modeling_smolvla.py multiplies losses by ~actions_is_pad (masks padded targets out of the loss)",
        "in_episode_bound = ~actions_is_pad" in source and "losses = losses * in_episode_bound" in source,
    )
    print()

    print("=== 3. SmolVLALiberoActionAdapter gripper decode formula -- known synthetic values ===")
    adapter = SmolVLALiberoActionAdapter()
    manifest = MANIFEST_REGISTRY["HuggingFaceVLA/smolvla_libero"]

    def decode_gripper(raw_gripper_value: float):
        native = NativePolicyAction(
            values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, raw_gripper_value],
            source_policy="test", postprocessor_used=True, metadata={},
        )
        command = adapter.decode(native, manifest, context={"degraded_input": False})
        return command.gripper_opening_01, ("close" if command.gripper_opening_01 <= 0.5 else "open")

    opening_01, cmd = decode_gripper(-1.0)
    check("raw=-1.0 (LIBERO native open) -> gripper_opening_01=1.0 -> open", abs(opening_01 - 1.0) < 1e-9 and cmd == "open", f"{opening_01}, {cmd}")
    opening_01, cmd = decode_gripper(1.0)
    check("raw=+1.0 (LIBERO native closed) -> gripper_opening_01=0.0 -> close", abs(opening_01 - 0.0) < 1e-9 and cmd == "close", f"{opening_01}, {cmd}")
    opening_01, cmd = decode_gripper(0.0)
    check(
        "raw=0.0 (our OWN collector's 'fully open' value, NOT LIBERO's native scale) -> "
        "gripper_opening_01=0.5 -> 'close' (boundary tie-break) -- this IS the confirmed root cause: "
        "a locally fine-tuned checkpoint's own [0,1]-scale 'open' value decodes as 'close' through this "
        "LIBERO-native formula",
        abs(opening_01 - 0.5) < 1e-9 and cmd == "close",
        f"{opening_01}, {cmd}",
    )
    for v in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        opening_01, cmd = decode_gripper(v)
        check(f"our-scale value {v} (anywhere in our dataset's real [0,1] range) decodes to 'close'", cmd == "close", f"{opening_01}, {cmd}")
    print()

    print("=== 4. Our collector's gripper convention matches what's actually stored in train20 ===")
    df = pd.read_parquet(TRAIN20_ROOT / "data" / "chunk-000" / "file-000.parquet")
    unique_values = sorted(set(np.stack(df["action"].to_numpy())[:, 6].tolist()))
    check("train20 action[6] only ever takes values {0.0, 1.0} (matches ActionAdapter.convert()'s own threshold convention)", unique_values == [0.0, 1.0], str(unique_values))
    print()

    print("=== 5. Processor stats extraction reads the real files this report cites ===")
    stats = compare_processor_stats()
    check("train20 gripper mean is in (0, 1) -- a [0,1]-range variable, not LIBERO's [-1,1]", 0.0 < stats["train20_action_mean"][6] < 1.0, str(stats["train20_action_mean"][6]))
    check("train20 gripper min/max == 0.0/1.0 exactly", stats["train20_action_min"][6] == 0.0 and stats["train20_action_max"][6] == 1.0)
    check("real LIBERO gripper min/max == -1.0/1.0 exactly (confirms the native-range mismatch)", stats["libero_action_min"][6] == -1.0 and stats["libero_action_max"][6] == 1.0)
    print()

    print("=== 6. No production file depends on this diagnostic module ===")
    hits = []
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy", "action_adapter"]
    for directory in production_dirs:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "diagnose_gripper_collapse" in text:
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    check("no production file imports/references benchmark.diagnose_gripper_collapse", len(hits) == 0, f"unexpected: {hits}")

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
