"""
Train a 3D MuViT MAE on volumetric microscopy crops.

Supported crop formats:
    .npy            (D, H, W) or (C, D, H, W)
    .tif / .tiff    (D, H, W) or (C, D, H, W)
    .h5 / .hdf5     dataset containing (D, H, W) or (C, D, H, W)

Each crop is treated as the finest resolution level. Coarser levels are
synthesized by block-average downsampling, then nearest-neighbor upsampling
back to the original shape -- so every level shares the same tensor shape
and the same physical field of view, but differs in effective voxel size.

Example: --levels 1,4,16 with --crop_size 128 128 128 produces three
(C, 128, 128, 128) tensors representing native, 4x-coarser, and 16x-coarser
resolution -- useful for multi-resolution / multi-FOV pretraining when only
one native resolution is available.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from muvit.data import MuViTDataset
from muvit.mae import MuViTMAE3d

IMG_EXTS = {".tif", ".tiff", ".npy", ".h5", ".hdf5"}
Size3D = tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def as_3tuple(value: Union[int, Sequence[int]]) -> Size3D:
    """Convert an int or 3-element sequence to a (D, H, W) tuple."""
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected an integer or 3 values, got {value}")
    return tuple(int(x) for x in value)


def load_image(path: Path, h5_key: str = "raw") -> np.ndarray:
    """Load a single 3D crop as float32 (C, D, H, W).

    Accepts (D,H,W), (C,D,H,W), or (D,H,W,C) source arrays.
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    elif suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            if h5_key not in f:
                raise KeyError(
                    f"Dataset key {h5_key!r} not found in {path}. "
                    f"Available top-level keys: {list(f.keys())}"
                )
            arr = f[h5_key][...]
    else:
        raise ValueError(f"Unsupported image extension: {suffix}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[None]  # (D,H,W) -> (1,D,H,W)
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 2, 3, 4):
            arr = np.moveaxis(arr, -1, 0)  # (D,H,W,C) -> (C,D,H,W)
        # else assume already (C,D,H,W)
    else:
        raise ValueError(
            f"Unsupported 3D array shape {arr.shape} for {path}; "
            "expected (D,H,W), (C,D,H,W), or (D,H,W,C)."
        )

    if any(s <= 0 for s in arr.shape):
        raise ValueError(f"Invalid image shape {arr.shape} for {path}")

    return np.ascontiguousarray(arr, dtype=np.float32)


def pyramid_level(img: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample by `factor`, then nearest-neighbor upsample
    back to the original (C, D, H, W) shape -- same shape, coarser content.
    """
    if factor == 1:
        return img.copy()
    if factor <= 0:
        raise ValueError(f"Resolution factor must be positive, got {factor}")

    C, D, H, W = img.shape
    if any(dim % factor for dim in (D, H, W)):
        raise ValueError(f"Volume shape {(D, H, W)} is not divisible by level factor {factor}")

    Dc, Hc, Wc = D // factor, H // factor, W // factor
    down = img.reshape(C, Dc, factor, Hc, factor, Wc, factor).mean(axis=(2, 4, 6))

    up = down
    for axis in (1, 2, 3):
        up = np.repeat(up, factor, axis=axis)

    assert up.shape == img.shape, f"Pyramid level shape mismatch: {up.shape} vs {img.shape}"
    return np.ascontiguousarray(up)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CropDataset(MuViTDataset):
    """3D dataset for MuViT MAE pretraining.

    Each sample is a native-resolution crop; `levels` (e.g. (1, 4, 16))
    synthesizes coarser resolution levels from that same crop, all sharing
    one physical field of view.

    Returns {"img": (L, C, D, H, W), "bbox": (L, 2, 3)}.
    """

    def __init__(
        self,
        files: list[Path],
        n_channels: int,
        levels: tuple[int, ...] = (1, 4, 16),
        crop_size: Optional[Union[int, Size3D]] = None,
        augment: bool = False,
        normalize: Optional[str] = "percentile",
    ):
        self._files = files
        self._n_channels = n_channels
        self._levels = tuple(sorted(int(x) for x in levels))
        self._crop_size = None if crop_size is None else as_3tuple(crop_size)
        self._augment = augment
        self._normalize = normalize
        super().__init__()

    def __len__(self) -> int:
        return len(self._files)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def levels(self) -> tuple[int, ...]:
        return self._levels

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        if self._normalize == "percentile":
            lo, hi = np.percentile(img, [1, 99.8])
            img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        elif self._normalize == "minmax":
            lo, hi = img.min(), img.max()
            img = (img - lo) / max(hi - lo, 1e-6)
        elif self._normalize is not None:
            raise ValueError(f"Unknown normalize mode {self._normalize!r}")
        return img.astype(np.float32)

    @staticmethod
    def _random_crop(img: np.ndarray, size: Size3D) -> np.ndarray:
        _, D, H, W = img.shape
        crop_d, crop_h, crop_w = size
        for dim, crop_dim, name in ((D, crop_d, "depth"), (H, crop_h, "height"), (W, crop_w, "width")):
            if dim < crop_dim:
                raise ValueError(f"Volume {name} {dim} is smaller than requested crop {name} {crop_dim}")

        z0 = random.randint(0, D - crop_d)
        y0 = random.randint(0, H - crop_h)
        x0 = random.randint(0, W - crop_w)
        return img[:, z0 : z0 + crop_d, y0 : y0 + crop_h, x0 : x0 + crop_w]

    @staticmethod
    def _random_flip_rot(img: np.ndarray) -> np.ndarray:
        """Random flips along Z/Y/X plus a random 90-degree rotation about one axis pair."""
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                img = np.flip(img, axis=axis)

        plane = random.choice([(1, 2), (1, 3), (2, 3)])  # Z-Y, Z-X, Y-X
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k=k, axes=plane)

        return np.ascontiguousarray(img)

    def __getitem__(self, idx: int) -> dict:
        path = self._files[idx]
        img = load_image(path)

        if img.shape[0] != self._n_channels:
            raise ValueError(f"{path} has {img.shape[0]} channel(s), expected {self._n_channels}.")

        if self._crop_size is not None:
            img = self._random_crop(img, self._crop_size)

        img = self._normalize_img(img)  # normalize before building the pyramid

        if self._augment:
            img = self._random_flip_rot(img)

        _, D, H, W = img.shape
        pyramid = np.stack([pyramid_level(img, factor) for factor in self._levels], axis=0)
        img_t = torch.from_numpy(np.ascontiguousarray(pyramid, dtype=np.float32))

        # All synthetic levels share the same physical FOV -> identical bboxes.
        bbox = torch.tensor([[[0, 0, 0], [D, H, W]]] * len(self._levels), dtype=torch.float32)

        return {"img": img_t, "bbox": bbox}


# --------------------------------------------------------------------------- #
# Dataset construction / split
# --------------------------------------------------------------------------- #
def build_datasets(args: argparse.Namespace) -> tuple[CropDataset, CropDataset]:
    files = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No supported 3D images found under {args.data_dir}.\nSupported extensions: {sorted(IMG_EXTS)}"
        )

    random.seed(args.seed)
    shuffled = random.sample(files, len(files))
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_files, train_files = shuffled[:n_val], shuffled[n_val:]
    print(f"Found {len(files)} volumes -> {len(train_files)} train / {len(val_files)} val")

    common = dict(n_channels=args.n_channels, levels=args.levels, crop_size=args.crop_size, normalize=args.normalize)
    train_ds = CropDataset(train_files, augment=True, **common)
    val_ds = CropDataset(val_files, augment=False, **common)
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Argument parsing / validation
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 3D MuViT MAE on volumetric microscopy crops.")

    p.add_argument("-c", type=str, default=None, help="YAML file of arguments; CLI flags override it.")

    # Data
    p.add_argument("--data_dir", required=False, help="Folder containing 3D crops (searched recursively).")
    p.add_argument("--output", required=False, help="Folder for checkpoints/logs.")
    p.add_argument("--n_channels", type=int, default=1, help="Number of channels in the volume.")
    p.add_argument(
        "--crop_size", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
        help="Random 3D crop size, e.g. --crop_size 128 128 128",
    )
    p.add_argument(
        "--patch_size", type=int, nargs="+", default=[8],
        help="3D patch size: one value or three (D H W), e.g. --patch_size 8 or --patch_size 4 8 8",
    )
    p.add_argument(
        "--levels", type=str, default="1,4,16",
        help="Comma-separated resolution factors, finest first, e.g. '1,4,16'.",
    )

    # Training
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=2, help="3D volumes usually need a much smaller batch size than 2D.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=None, help="Default: let MuViTMAE.fit determine it.")

    # Model
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_layers_decoder", type=int, default=4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--dim_decoder", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--masking_ratio", type=float, default=0.75)

    # Normalization / logging / misc
    p.add_argument("--normalize", choices=["percentile", "minmax", "none"], default="percentile")
    p.add_argument("--logger", choices=["tensorboard", "wandb", "none"], default="tensorboard")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry", action="store_true", help="Fast sanity-check run.")

    # Two-pass parse: read --config first (if given), use its values as the
    # parser's defaults, then do the real parse so any CLI flag still wins.
    config_path = _peek_config_path()
    if config_path is not None:
        p.set_defaults(**_load_yaml_config(config_path))

    args = p.parse_args()
    if not args.data_dir or not args.output:
        p.error("--data_dir and --output are required (via CLI or --config).")
    return args


def _peek_config_path() -> Optional[str]:
    """Look for --config in sys.argv without disturbing the main parser."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", type=str, default=None)
    known, _ = peek.parse_known_args()
    return known.config


def _load_yaml_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    # levels/crop_size/patch_size come through as native YAML types already;
    # argparse's own parsing (nargs, type=str for --levels) is bypassed for
    # anything set via set_defaults, so normalize the couple of fields whose
    # CLI form differs from their natural YAML form.
    if "levels" in config and not isinstance(config["levels"], str):
        config["levels"] = ",".join(str(x) for x in config["levels"])
    return config


def _parse_levels(spec: str) -> tuple[int, ...]:
    levels = tuple(sorted(int(x.strip()) for x in spec.split(",")))
    if 1 not in levels:
        print(f"Warning: --levels {levels} does not include 1; 1 is recommended as the finest level.")
    return levels


def _parse_patch_size(values: list[int]) -> Size3D:
    if len(values) == 1:
        return (values[0],) * 3
    if len(values) == 3:
        return tuple(values)
    raise ValueError("--patch_size must contain either 1 or 3 integers.")


def _validate_crop_size(crop_size: Size3D, patch_size: Size3D, levels: tuple[int, ...]) -> None:
    for dim, patch_dim in zip(crop_size, patch_size):
        if dim % patch_dim:
            raise ValueError(f"Crop dimension {dim} is not divisible by patch dimension {patch_dim}.")

    for level in levels:
        for dim in crop_size:
            if dim % level:
                raise ValueError(
                    f"Crop size {crop_size} must be divisible by every resolution level; "
                    f"dimension {dim} is not divisible by level {level}."
                )


def _print_config(args: argparse.Namespace) -> None:
    fields = {
        "Data directory": args.data_dir,
        "Output": args.output,
        "Channels": args.n_channels,
        "Crop size": args.crop_size,
        "Patch size": args.patch_size,
        "Levels": args.levels,
        "Model dim": args.dim,
        "Decoder dim": args.dim_decoder,
        "Encoder layers": args.num_layers,
        "Decoder layers": args.num_layers_decoder,
        "Attention heads": args.heads,
        "Mask ratio": args.masking_ratio,
    }
    width = 70
    print(f"\n{'=' * width}\n3D MuViT MAE\n{'=' * width}")
    for name, value in fields.items():
        print(f"{name:<15}: {value}")
    print(f"{'=' * width}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    args.normalize = None if args.normalize == "none" else args.normalize
    args.levels = _parse_levels(args.levels)
    args.patch_size = _parse_patch_size(args.patch_size)

    if args.crop_size is not None:
        args.crop_size = tuple(args.crop_size)
        _validate_crop_size(args.crop_size, args.patch_size, args.levels)

    _print_config(args)

    train_ds, val_ds = build_datasets(args)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = MuViTMAE3d(
        in_channels=train_ds.n_channels,
        levels=train_ds.levels,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        dim=args.dim,
        num_layers_decoder=args.num_layers_decoder,
        dim_decoder=args.dim_decoder,
        heads=args.heads,
        decoder_mode="multi",
        loss="mse",
        masking_ratio=args.masking_ratio,
        use_level_embed=True,
        rotary_mode="per_layer",
        rotary_base=10000,
        attention_mode="all",
        masking_mode="dirichlet",
        input_space="real",
        dropout=0.0,
    )

    model.fit(
        train_dl,
        val_dl,
        output=args.output,
        num_epochs=args.epochs,
        lr=args.lr,
        logger=None if args.logger == "none" else args.logger,
        run_name=args.run_name,
        dry=args.dry,
    )


if __name__ == "__main__":
    main()