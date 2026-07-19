"""Post-hoc cross-dataset loss computation for a saved SmolVLA checkpoint
(see this task's chat report). lerobot-train's own --eval_steps mechanism
only computes eval loss on a held-out SPLIT of the SAME training dataset
(--dataset.eval_split); it has no notion of a second, separate dataset
root. Since this task needs "validation loss" measured specifically
against datasets/recycling_validation20_v1 (a dataset the checkpoint
never trained on) rather than a random held-out slice of
recycling_train80_v1, this script instead reproduces exactly the same
forward-pass loss computation lerobot_train.py's own eval branch uses
(preprocessor(batch); loss, _ = policy.forward(batch)) against whichever
dataset root is passed in -- no training, no gradient step, no
production file touched.

Run:
  .venv-vla/bin/python -m benchmark.compute_validation_loss \\
    --checkpoint-dir outputs/train/smolvla_recycling_train80_v1/checkpoints/000500/pretrained_model \\
    --dataset-repo-id local/recycling_cell_train80_v1 --dataset-root datasets/recycling_train80_v1 \\
    --dataset-repo-id local/recycling_cell_validation20_v1 --dataset-root datasets/recycling_validation20_v1
"""

import argparse
import json
import math
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_loss(checkpoint_dir: str, repo_id: str, dataset_root: str, device: str = "cuda", max_samples: int = 0) -> dict:
    checkpoint_dir = str(resolve(checkpoint_dir))
    dataset_root = str(resolve(dataset_root))

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_cfg.pretrained_path = checkpoint_dir
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps, return_uint8=True)

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=checkpoint_dir)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    losses = []
    nan_inf_count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples and i >= max_samples:
                break
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            value = loss.item()
            if math.isnan(value) or math.isinf(value):
                nan_inf_count += 1
                continue
            losses.append(value)

    return {
        "checkpoint_dir": checkpoint_dir,
        "dataset_repo_id": repo_id,
        "dataset_root": dataset_root,
        "num_samples": len(losses),
        "nan_inf_count": nan_inf_count,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-repo-id", action="append", required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all frames")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    assert len(args.dataset_repo_id) == len(args.dataset_root), "Pass matching --dataset-repo-id/--dataset-root pairs"

    results = []
    for repo_id, root in zip(args.dataset_repo_id, args.dataset_root):
        print(f"=== Computing loss: checkpoint={args.checkpoint_dir}  dataset={root} ===")
        result = compute_loss(args.checkpoint_dir, repo_id, root, args.device, args.max_samples)
        print(json.dumps(result, indent=2))
        results.append(result)

    if args.output:
        output_path = resolve(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
