from typing import Dict
import copy

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class FrankaLeapImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        load_to_memory: bool = False,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.load_to_memory = load_to_memory

        rgb_keys, lowdim_keys = self._parse_shape_meta(shape_meta)
        replay_keys = rgb_keys + lowdim_keys + ["action"]

        if load_to_memory:
            self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=replay_keys)
        else:
            # Keep image arrays on disk. Raw RGB frames are too large for eager in-memory copies.
            self.replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode="r")

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self._validate_replay_buffer_shapes()

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=replay_keys,
            key_first_k=key_first_k,
            episode_mask=train_mask,
        )

    @staticmethod
    def _parse_shape_meta(shape_meta: dict) -> tuple[list[str], list[str]]:
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type `{obs_type}` for key `{key}`")
        return sorted(rgb_keys), sorted(lowdim_keys)

    def _validate_replay_buffer_shapes(self):
        for key in self.rgb_keys:
            if key not in self.replay_buffer:
                raise RuntimeError(f"Replay buffer missing rgb key `{key}`")
            expected_chw = tuple(self.shape_meta["obs"][key]["shape"])
            if len(expected_chw) != 3:
                raise RuntimeError(f"RGB key `{key}` must have shape [C, H, W], got {expected_chw}")
            expected_hwc = (expected_chw[1], expected_chw[2], expected_chw[0])
            actual_hwc = tuple(self.replay_buffer[key].shape[1:])
            if actual_hwc != expected_hwc:
                raise RuntimeError(
                    f"Replay buffer key `{key}` has shape {actual_hwc}, expected {expected_hwc} "
                    f"from shape_meta {expected_chw}"
                )

        for key in self.lowdim_keys:
            if key not in self.replay_buffer:
                raise RuntimeError(f"Replay buffer missing low-dim key `{key}`")
            expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
            actual_shape = tuple(self.replay_buffer[key].shape[1:])
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Replay buffer key `{key}` has shape {actual_shape}, expected {expected_shape}"
                )

        action_shape = tuple(self.shape_meta["action"]["shape"])
        actual_action_shape = tuple(self.replay_buffer["action"].shape[1:])
        if actual_action_shape != action_shape:
            raise RuntimeError(
                f"Replay buffer key `action` has shape {actual_action_shape}, expected {action_shape}"
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.rgb_keys + self.lowdim_keys + ["action"],
            key_first_k={key: self.n_obs_steps for key in self.rgb_keys + self.lowdim_keys}
            if self.n_obs_steps is not None
            else {},
            episode_mask=self.val_mask,
        )
        val_set.train_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"], **kwargs)
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key], **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: dict[str, np.ndarray]) -> dict:
        if self.n_obs_steps is None:
            t_slice = slice(None)
        else:
            t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(sample[key][t_slice], -1, 1).astype(np.float32) / 255.0
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][t_slice].astype(np.float32)

        data = {
            "obs": obs_dict,
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
