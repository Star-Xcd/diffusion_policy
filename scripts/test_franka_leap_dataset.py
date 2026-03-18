#!/usr/bin/env python3
if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.dataset.base_dataset import BaseImageDataset


def _describe_tensor(name: str, tensor: torch.Tensor):
    print(f"{name:<24} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Instantiate and sanity-check Franka+LEAP image dataset.")
    parser.add_argument(
        "--config-name",
        default="train_diffusion_unet_real_image_workspace",
        help="Top-level training config to compose.",
    )
    parser.add_argument(
        "--task-name",
        default="franka_leap_image",
        help="Task config name under diffusion_policy/config/task/.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional override for task.dataset_path.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Temporary DataLoader batch size for the smoke test.",
    )
    args = parser.parse_args()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = Path(__file__).resolve().parent.parent / "diffusion_policy" / "config"

    overrides = [f"task={args.task_name}"]
    if args.dataset_path is not None:
        dataset_path = str(Path(args.dataset_path).expanduser().resolve())
        overrides.append(f"task.dataset_path={dataset_path}")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    if not isinstance(dataset, BaseImageDataset):
        raise RuntimeError(f"Expected BaseImageDataset, got {type(dataset)}")

    val_dataset = dataset.get_validation_dataset()
    if not isinstance(val_dataset, BaseImageDataset):
        raise RuntimeError(f"Expected validation dataset to be BaseImageDataset, got {type(val_dataset)}")

    print("Dataset instantiated successfully.")
    print(f"dataset_path             : {cfg.task.dataset.dataset_path}")
    print(f"horizon                  : {cfg.task.dataset.horizon}")
    print(f"pad_before               : {cfg.task.dataset.pad_before}")
    print(f"pad_after                : {cfg.task.dataset.pad_after}")
    print(f"n_obs_steps              : {cfg.task.dataset.n_obs_steps}")
    print(f"train samples            : {len(dataset)}")
    print(f"validation samples       : {len(val_dataset)}")
    print(f"num episodes             : {dataset.replay_buffer.n_episodes}")
    print(f"num timesteps            : {dataset.replay_buffer.n_steps}")

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    sample_index = min(max(args.sample_index, 0), len(dataset) - 1)
    sample = dataset[sample_index]
    print(f"\nSample[{sample_index}]")
    for key, value in sample["obs"].items():
        _describe_tensor(f"obs.{key}", value)
    _describe_tensor("action", sample["action"])

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(dataloader))
    print(f"\nBatch[{args.batch_size}]")
    for key, value in batch["obs"].items():
        _describe_tensor(f"obs.{key}", value)
    _describe_tensor("action", batch["action"])


if __name__ == "__main__":
    main()
