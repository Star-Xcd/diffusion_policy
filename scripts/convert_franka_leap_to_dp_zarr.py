#!/usr/bin/env python3
if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


THIRD_VIEW_PATH_KEY = "observation_image_third_view_path"
WRIST_PATH_KEY = "observation_image_wrist_path"
STATE_KEY = "observation_q"
ACTION_KEY = "actions_delta_q"
ABS_ACTION_KEY = "actions_q_target_abs"
META_FILENAME = "episode_meta.json"
NPZ_FILENAME = "data.npz"
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 16
HAND_PRESET_METADATA_SUFFIX = ".hand_preset.npz"


@dataclass(frozen=True)
class HandPresetMetadata:
    source_path: Path
    hand_presets: np.ndarray

    @property
    def preset_count(self) -> int:
        return int(self.hand_presets.shape[0])

    @classmethod
    def from_npz(cls, metadata_path: Path) -> "HandPresetMetadata":
        metadata_path = metadata_path.expanduser().resolve()
        with np.load(metadata_path, allow_pickle=False) as metadata:
            representation = metadata.get("representation")
            if representation is not None:
                representation_value = str(np.asarray(representation).item())
                if representation_value != "hand_preset":
                    raise RuntimeError(
                        f"Expected hand_preset metadata in {metadata_path}, found representation={representation_value!r}"
                    )
            hand_presets = np.asarray(metadata["hand_presets"], dtype=np.float32)
            arm_action_dim = int(np.asarray(metadata.get("arm_action_dim", ARM_ACTION_DIM)).item())
            hand_action_dim = int(np.asarray(metadata.get("hand_action_dim", HAND_ACTION_DIM)).item())

        if arm_action_dim != ARM_ACTION_DIM:
            raise RuntimeError(
                f"Expected arm_action_dim={ARM_ACTION_DIM} in {metadata_path}, got {arm_action_dim}"
            )
        if hand_action_dim != HAND_ACTION_DIM:
            raise RuntimeError(
                f"Expected hand_action_dim={HAND_ACTION_DIM} in {metadata_path}, got {hand_action_dim}"
            )
        if hand_presets.ndim != 2 or hand_presets.shape[1] != HAND_ACTION_DIM:
            raise RuntimeError(
                f"Expected hand_presets with shape [K, {HAND_ACTION_DIM}], got {hand_presets.shape} in {metadata_path}"
            )
        return cls(source_path=metadata_path, hand_presets=hand_presets)

    def assign_many(self, hand_targets_q: np.ndarray) -> np.ndarray:
        hand_targets_q = np.asarray(hand_targets_q, dtype=np.float32)
        if hand_targets_q.ndim != 2 or hand_targets_q.shape[1] != HAND_ACTION_DIM:
            raise RuntimeError(
                f"Expected hand target array with shape [T, {HAND_ACTION_DIM}], got {hand_targets_q.shape}"
            )
        distances = np.sum(np.square(hand_targets_q[:, None, :] - self.hand_presets[None, :, :]), axis=-1)
        return np.argmin(distances, axis=1).astype(np.int32)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_dir: Path
    state: np.ndarray
    action: np.ndarray
    third_view: Optional[np.ndarray]
    wrist_view: Optional[np.ndarray]
    success: Optional[bool]
    prompt: str


def _decode_path_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _find_episode_dirs(input_dir: Path) -> list[Path]:
    if (input_dir / NPZ_FILENAME).exists():
        return [input_dir]

    episode_dirs: list[Path] = []
    for npz_path in input_dir.rglob(NPZ_FILENAME):
        episode_dir = npz_path.parent
        if episode_dir.name.startswith("episode_"):
            episode_dirs.append(episode_dir)

    episode_dirs.sort()
    return episode_dirs


def _load_episode_meta(episode_dir: Path) -> dict:
    meta_path = episode_dir / META_FILENAME
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_episode_arrays(
    data: np.lib.npyio.NpzFile,
    episode_dir: Path,
    include_third_view: bool,
    include_wrist: bool,
    action_representation: str,
) -> int:
    if STATE_KEY not in data.files:
        raise RuntimeError(f"Missing key `{STATE_KEY}` in {episode_dir / NPZ_FILENAME}")
    if ACTION_KEY not in data.files:
        raise RuntimeError(f"Missing key `{ACTION_KEY}` in {episode_dir / NPZ_FILENAME}")

    state = data[STATE_KEY]
    raw_action = data[ACTION_KEY]
    if state.ndim != 2 or state.shape[1] != 23:
        raise RuntimeError(
            f"Expected `{STATE_KEY}` shape (T, 23), got {state.shape} in {episode_dir / NPZ_FILENAME}"
        )
    if raw_action.ndim != 2 or raw_action.shape[1] != 23:
        raise RuntimeError(
            f"Expected `{ACTION_KEY}` shape (T, 23), got {raw_action.shape} in {episode_dir / NPZ_FILENAME}"
        )

    num_steps = int(raw_action.shape[0])
    if state.shape[0] != num_steps:
        raise RuntimeError(
            f"Length mismatch in {episode_dir / NPZ_FILENAME}: "
            f"`{STATE_KEY}` has {state.shape[0]}, `{ACTION_KEY}` has {num_steps}"
        )

    if action_representation == "hand_preset":
        if ABS_ACTION_KEY not in data.files:
            raise RuntimeError(
                f"Missing key `{ABS_ACTION_KEY}` required for action_representation='hand_preset' "
                f"in {episode_dir / NPZ_FILENAME}"
            )
        abs_action = data[ABS_ACTION_KEY]
        if abs_action.ndim != 2 or abs_action.shape[1] != 23:
            raise RuntimeError(
                f"Expected `{ABS_ACTION_KEY}` shape (T, 23), got {abs_action.shape} in {episode_dir / NPZ_FILENAME}"
            )
        if abs_action.shape[0] != num_steps:
            raise RuntimeError(
                f"Length mismatch in {episode_dir / NPZ_FILENAME}: "
                f"`{ABS_ACTION_KEY}` has {abs_action.shape[0]}, `{ACTION_KEY}` has {num_steps}"
            )

    if include_third_view:
        if THIRD_VIEW_PATH_KEY not in data.files:
            raise RuntimeError(f"Missing key `{THIRD_VIEW_PATH_KEY}` in {episode_dir / NPZ_FILENAME}")
        if len(data[THIRD_VIEW_PATH_KEY]) != num_steps:
            raise RuntimeError(
                f"Length mismatch in {episode_dir / NPZ_FILENAME}: "
                f"`{THIRD_VIEW_PATH_KEY}` has {len(data[THIRD_VIEW_PATH_KEY])}, `{ACTION_KEY}` has {num_steps}"
            )

    if include_wrist:
        if WRIST_PATH_KEY not in data.files:
            raise RuntimeError(f"Missing key `{WRIST_PATH_KEY}` in {episode_dir / NPZ_FILENAME}")
        if len(data[WRIST_PATH_KEY]) != num_steps:
            raise RuntimeError(
                f"Length mismatch in {episode_dir / NPZ_FILENAME}: "
                f"`{WRIST_PATH_KEY}` has {len(data[WRIST_PATH_KEY])}, `{ACTION_KEY}` has {num_steps}"
            )

    return num_steps


def _load_image_sequence(
    *,
    episode_dir: Path,
    data: np.lib.npyio.NpzFile,
    path_key: str,
    num_steps: int,
    view_name: str,
    expected_shape: Optional[tuple[int, int, int]],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    rel_paths = [_decode_path_value(value) for value in data[path_key]]
    if len(rel_paths) != num_steps:
        raise RuntimeError(
            f"Expected {num_steps} image paths for {view_name}, got {len(rel_paths)} in {episode_dir / NPZ_FILENAME}"
        )

    frames = []
    local_shape = expected_shape
    for rel_path in rel_paths:
        image = _load_rgb_image(episode_dir / rel_path)
        if local_shape is None:
            local_shape = tuple(int(x) for x in image.shape)
        elif tuple(image.shape) != local_shape:
            raise RuntimeError(
                f"Inconsistent {view_name} image shape in {episode_dir}: "
                f"expected {local_shape}, got {tuple(image.shape)} for {rel_path}"
            )
        frames.append(image)

    assert local_shape is not None
    return np.stack(frames, axis=0).astype(np.uint8, copy=False), local_shape


def _encode_actions(
    *,
    raw_action: np.ndarray,
    abs_action: Optional[np.ndarray],
    action_representation: str,
    hand_preset_metadata: Optional[HandPresetMetadata],
    episode_dir: Path,
) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if action_representation == "delta_q":
        return raw_action

    if action_representation != "hand_preset":
        raise RuntimeError(f"Unsupported action_representation={action_representation!r}")
    if hand_preset_metadata is None:
        raise RuntimeError("hand_preset conversion requires preset metadata.")
    if abs_action is None:
        raise RuntimeError(
            f"Expected `{ABS_ACTION_KEY}` to encode hand_preset actions in {episode_dir / NPZ_FILENAME}"
        )

    abs_action = np.asarray(abs_action, dtype=np.float32)
    hand_targets = abs_action[:, ARM_ACTION_DIM:]
    preset_ids = hand_preset_metadata.assign_many(hand_targets)
    preset_onehot = np.eye(hand_preset_metadata.preset_count, dtype=np.float32)[preset_ids]
    encoded = np.concatenate([raw_action[:, :ARM_ACTION_DIM], preset_onehot], axis=1)
    return encoded.astype(np.float32, copy=False)


def _load_episode_record(
    episode_dir: Path,
    *,
    include_third_view: bool,
    include_wrist: bool,
    expected_shapes: dict[str, Optional[tuple[int, int, int]]],
    action_representation: str,
    hand_preset_metadata: Optional[HandPresetMetadata],
) -> EpisodeRecord:
    meta = _load_episode_meta(episode_dir)

    with np.load(episode_dir / NPZ_FILENAME, allow_pickle=False) as data:
        num_steps = _validate_episode_arrays(
            data=data,
            episode_dir=episode_dir,
            include_third_view=include_third_view,
            include_wrist=include_wrist,
            action_representation=action_representation,
        )

        state = np.asarray(data[STATE_KEY], dtype=np.float32)
        raw_action = np.asarray(data[ACTION_KEY], dtype=np.float32)
        abs_action = np.asarray(data[ABS_ACTION_KEY], dtype=np.float32) if action_representation == "hand_preset" else None
        action = _encode_actions(
            raw_action=raw_action,
            abs_action=abs_action,
            action_representation=action_representation,
            hand_preset_metadata=hand_preset_metadata,
            episode_dir=episode_dir,
        )

        third_view = None
        if include_third_view:
            third_view, third_shape = _load_image_sequence(
                episode_dir=episode_dir,
                data=data,
                path_key=THIRD_VIEW_PATH_KEY,
                num_steps=num_steps,
                view_name="third_view",
                expected_shape=expected_shapes["third_view"],
            )
            expected_shapes["third_view"] = third_shape

        wrist_view = None
        if include_wrist:
            wrist_view, wrist_shape = _load_image_sequence(
                episode_dir=episode_dir,
                data=data,
                path_key=WRIST_PATH_KEY,
                num_steps=num_steps,
                view_name="wrist_view",
                expected_shape=expected_shapes["wrist_view"],
            )
            expected_shapes["wrist_view"] = wrist_shape

    return EpisodeRecord(
        episode_dir=episode_dir,
        state=state,
        action=action,
        third_view=third_view,
        wrist_view=wrist_view,
        success=meta.get("success"),
        prompt=str(meta.get("prompt", "")),
    )


def _remove_output_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _copy_hand_preset_metadata(
    *,
    output_path: Path,
    hand_preset_metadata: Optional[HandPresetMetadata],
) -> Optional[Path]:
    if hand_preset_metadata is None:
        return None
    metadata_copy_path = output_path.parent / f"{output_path.name}{HAND_PRESET_METADATA_SUFFIX}"
    shutil.copy2(hand_preset_metadata.source_path, metadata_copy_path)
    return metadata_copy_path


def _write_manifest(
    *,
    output_path: Path,
    kept_records: list[EpisodeRecord],
    skipped: list[dict],
    include_third_view: bool,
    include_wrist: bool,
    action_representation: str,
    action_shape: tuple[int, ...],
    preset_metadata_source: Optional[Path],
    preset_metadata_copy: Optional[Path],
):
    manifest = {
        "num_kept_episodes": len(kept_records),
        "num_skipped_episodes": len(skipped),
        "include_third_view": include_third_view,
        "include_wrist": include_wrist,
        "action_representation": action_representation,
        "action_shape": list(action_shape),
        "preset_metadata_source": None if preset_metadata_source is None else str(preset_metadata_source),
        "preset_metadata_copy": None if preset_metadata_copy is None else str(preset_metadata_copy),
        "kept_episodes": [
            {
                "episode_dir": str(record.episode_dir),
                "num_steps": int(record.action.shape[0]),
                "success": record.success,
                "prompt": record.prompt,
            }
            for record in kept_records
        ],
        "skipped_episodes": skipped,
    }

    manifest_path = output_path.parent / f"{output_path.name}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


@click.command()
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Overwrite the output zarr directory if it already exists.")
@click.option("--success-only/--include-failures", default=True, show_default=True)
@click.option("--include-third-view/--no-third-view", default=True, show_default=True)
@click.option("--include-wrist/--no-wrist", default=True, show_default=True)
@click.option(
    "--allow-missing-cameras",
    is_flag=True,
    help="Skip episodes with missing requested cameras instead of failing.",
)
@click.option(
    "--compressor",
    type=click.Choice(["default", "disk", "none"]),
    default="disk",
    show_default=True,
    help="Compression preset passed to ReplayBuffer when arrays are first created.",
)
@click.option(
    "--action-representation",
    type=click.Choice(["delta_q", "hand_preset"]),
    default="delta_q",
    show_default=True,
    help="Action representation to write into zarr.",
)
@click.option(
    "--preset-metadata",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to franka_leap_hand_action_repr.npz. Required for action_representation=hand_preset.",
)
def main(
    input_dir: Path,
    output: Path,
    overwrite: bool,
    success_only: bool,
    include_third_view: bool,
    include_wrist: bool,
    allow_missing_cameras: bool,
    compressor: str,
    action_representation: str,
    preset_metadata: Optional[Path],
):
    if not include_third_view and not include_wrist:
        raise click.ClickException("At least one image stream must be enabled for the image-based DP baseline.")

    input_dir = input_dir.expanduser().resolve()
    output = output.expanduser().resolve()

    hand_preset_metadata = None
    if action_representation == "hand_preset":
        if preset_metadata is None:
            raise click.ClickException(
                "--preset-metadata is required when --action-representation=hand_preset."
            )
        hand_preset_metadata = HandPresetMetadata.from_npz(preset_metadata)
    elif preset_metadata is not None:
        raise click.ClickException(
            "--preset-metadata is only valid when --action-representation=hand_preset."
        )

    episode_dirs = _find_episode_dirs(input_dir)
    if not episode_dirs:
        raise click.ClickException(f"No episode directories found under: {input_dir}")

    metadata_copy_path = output.parent / f"{output.name}{HAND_PRESET_METADATA_SUFFIX}"
    if output.exists():
        if not overwrite:
            raise click.ClickException(f"Output already exists: {output}. Pass --overwrite to replace it.")
        _remove_output_path(output)
    if metadata_copy_path.exists():
        _remove_output_path(metadata_copy_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(output)))
    compressor_arg = None if compressor == "none" else compressor

    expected_shapes: dict[str, Optional[tuple[int, int, int]]] = {
        "third_view": None,
        "wrist_view": None,
    }
    kept_records: list[EpisodeRecord] = []
    skipped: list[dict] = []
    total_steps = 0

    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        meta = _load_episode_meta(episode_dir)
        success = meta.get("success")
        if success_only and success is not True:
            skipped.append({"episode_dir": str(episode_dir), "reason": f"success={success!r}"})
            continue

        try:
            record = _load_episode_record(
                episode_dir,
                include_third_view=include_third_view,
                include_wrist=include_wrist,
                expected_shapes=expected_shapes,
                action_representation=action_representation,
                hand_preset_metadata=hand_preset_metadata,
            )
        except RuntimeError as exc:
            if allow_missing_cameras and (
                THIRD_VIEW_PATH_KEY in str(exc) or WRIST_PATH_KEY in str(exc) or "Failed to read image" in str(exc)
            ):
                skipped.append({"episode_dir": str(episode_dir), "reason": str(exc)})
                continue
            raise click.ClickException(str(exc)) from exc

        episode_payload = {
            "action": record.action,
            "state": record.state,
        }
        if include_third_view:
            assert record.third_view is not None
            episode_payload["third_view"] = record.third_view
        if include_wrist:
            assert record.wrist_view is not None
            episode_payload["wrist_view"] = record.wrist_view

        replay_buffer.add_episode(
            episode_payload,
            compressors=compressor_arg,
        )
        kept_records.append(record)
        total_steps += int(record.action.shape[0])

    if not kept_records:
        _remove_output_path(output)
        raise click.ClickException("No episodes were converted. Nothing written.")

    metadata_copy = _copy_hand_preset_metadata(
        output_path=output,
        hand_preset_metadata=hand_preset_metadata,
    )

    manifest_path = output.parent / f"{output.name}.manifest.json"
    action_shape = tuple(int(x) for x in kept_records[0].action.shape[1:])
    _write_manifest(
        output_path=output,
        kept_records=kept_records,
        skipped=skipped,
        include_third_view=include_third_view,
        include_wrist=include_wrist,
        action_representation=action_representation,
        action_shape=action_shape,
        preset_metadata_source=None if hand_preset_metadata is None else hand_preset_metadata.source_path,
        preset_metadata_copy=metadata_copy,
    )

    click.echo(f"Converted episodes: {len(kept_records)}")
    click.echo(f"Skipped episodes: {len(skipped)}")
    click.echo(f"Total timesteps: {total_steps}")
    click.echo(f"Action representation: {action_representation}")
    click.echo(f"Action shape: {action_shape}")
    click.echo(f"Output zarr: {output}")
    click.echo(f"Manifest: {manifest_path}")
    if metadata_copy is not None:
        click.echo(f"Preset metadata copy: {metadata_copy}")
    if include_third_view:
        click.echo(f"third_view shape: {expected_shapes['third_view']}")
    if include_wrist:
        click.echo(f"wrist_view shape: {expected_shapes['wrist_view']}")


if __name__ == "__main__":
    main()
