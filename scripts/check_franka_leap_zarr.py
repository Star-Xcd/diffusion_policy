#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


def _require_key(group, key: str):
    if key not in group:
        raise RuntimeError(f"Missing key `{key}`")
    return group[key]


def _check_shape(name: str, array, expected_ndim: int, expected_last_dim: Optional[int] = None):
    if array.ndim != expected_ndim:
        raise RuntimeError(f"`{name}` must have ndim={expected_ndim}, got shape={array.shape}")
    if expected_last_dim is not None and array.shape[-1] != expected_last_dim:
        raise RuntimeError(
            f"`{name}` must have last_dim={expected_last_dim}, got shape={array.shape}"
        )


def _check_dtype(name: str, array, expected_dtype):
    if np.dtype(array.dtype) != np.dtype(expected_dtype):
        raise RuntimeError(f"`{name}` must have dtype={np.dtype(expected_dtype)}, got {array.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Validate Franka+LEAP Diffusion Policy zarr dataset.")
    parser.add_argument("--zarr-path", required=True, help="Path to the converted zarr dataset.")
    parser.add_argument(
        "--no-third-view",
        action="store_true",
        help="Do not require `data/third_view`.",
    )
    parser.add_argument(
        "--no-wrist",
        action="store_true",
        help="Do not require `data/wrist_view`.",
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path).expanduser().resolve()
    if not zarr_path.exists():
        raise RuntimeError(f"zarr path does not exist: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise RuntimeError("Expected root groups `data/` and `meta/`.")

    data_group = root["data"]
    meta_group = root["meta"]

    action = _require_key(data_group, "action")
    state = _require_key(data_group, "state")
    episode_ends = _require_key(meta_group, "episode_ends")

    _check_shape("data/action", action, expected_ndim=2, expected_last_dim=23)
    _check_shape("data/state", state, expected_ndim=2, expected_last_dim=23)
    _check_dtype("data/action", action, np.float32)
    _check_dtype("data/state", state, np.float32)
    _check_dtype("meta/episode_ends", episode_ends, np.int64)

    n_steps = int(action.shape[0])
    if int(state.shape[0]) != n_steps:
        raise RuntimeError(
            f"Length mismatch: `data/action` has {n_steps} steps, `data/state` has {state.shape[0]}"
        )

    third_view = None
    if not args.no_third_view:
        third_view = _require_key(data_group, "third_view")
        _check_shape("data/third_view", third_view, expected_ndim=4, expected_last_dim=3)
        _check_dtype("data/third_view", third_view, np.uint8)
        if int(third_view.shape[0]) != n_steps:
            raise RuntimeError(
                f"Length mismatch: `data/third_view` has {third_view.shape[0]} steps, expected {n_steps}"
            )

    wrist_view = None
    if not args.no_wrist:
        wrist_view = _require_key(data_group, "wrist_view")
        _check_shape("data/wrist_view", wrist_view, expected_ndim=4, expected_last_dim=3)
        _check_dtype("data/wrist_view", wrist_view, np.uint8)
        if int(wrist_view.shape[0]) != n_steps:
            raise RuntimeError(
                f"Length mismatch: `data/wrist_view` has {wrist_view.shape[0]} steps, expected {n_steps}"
            )

    episode_ends_np = np.asarray(episode_ends[:], dtype=np.int64)
    if episode_ends_np.ndim != 1:
        raise RuntimeError(f"`meta/episode_ends` must be 1D, got shape={episode_ends_np.shape}")
    if len(episode_ends_np) == 0:
        raise RuntimeError("`meta/episode_ends` is empty.")
    if np.any(episode_ends_np <= 0):
        raise RuntimeError(f"`meta/episode_ends` must be strictly positive, got {episode_ends_np}")
    if np.any(np.diff(episode_ends_np) <= 0):
        raise RuntimeError(f"`meta/episode_ends` must be strictly increasing, got {episode_ends_np}")
    if int(episode_ends_np[-1]) != n_steps:
        raise RuntimeError(
            f"`meta/episode_ends[-1]` must equal total steps {n_steps}, got {episode_ends_np[-1]}"
        )

    episode_starts = np.concatenate([[0], episode_ends_np[:-1]])
    episode_lengths = episode_ends_np - episode_starts

    print("Zarr validation passed.")
    print(f"zarr_path      : {zarr_path}")
    print(f"action         : shape={action.shape}, dtype={action.dtype}")
    print(f"state          : shape={state.shape}, dtype={state.dtype}")
    if third_view is not None:
        print(f"third_view     : shape={third_view.shape}, dtype={third_view.dtype}")
    if wrist_view is not None:
        print(f"wrist_view     : shape={wrist_view.shape}, dtype={wrist_view.dtype}")
    print(f"episode_ends   : shape={episode_ends.shape}, dtype={episode_ends.dtype}")
    print(f"num_episodes   : {len(episode_ends_np)}")
    print(f"total_steps    : {n_steps}")
    print(
        "episode_length : "
        f"min={int(episode_lengths.min())}, "
        f"max={int(episode_lengths.max())}, "
        f"mean={float(episode_lengths.mean()):.2f}"
    )
    print(f"last_5_ends    : {episode_ends_np[-5:]}")


if __name__ == "__main__":
    main()
