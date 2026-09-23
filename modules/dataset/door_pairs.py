"""Paired real/render door data for domain-specific XFeat fine-tuning.

The expected directory layout is::

    root/
      image_real/000000.png
      image_render/000000.png
      mask_real/000000_mask.png

The real image and Gaussian render are assumed to use the same camera and pixel
coordinates.  The mask is used both to reject invalid background supervision
and, optionally, to make the real input look more like the black-background
render at training/inference time.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DoorPairSample:
    stem: str
    real_path: Path
    render_path: Path
    mask_path: Path
    common_pixels: int


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return mask


def discover_door_pairs(
    root: str | Path,
    mask_threshold: int = 64,
    render_threshold: int = 3,
    min_common_pixels: int = 256,
) -> list[DoorPairSample]:
    """Find valid same-name triplets and discard empty/misaligned pairs."""

    root = Path(root)
    real_dir = root / "image_real"
    render_dir = root / "image_render"
    mask_dir = root / "mask_real"
    for directory in (real_dir, render_dir, mask_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {directory}")

    samples: list[DoorPairSample] = []
    for real_path in sorted(real_dir.glob("*.png")):
        stem = real_path.stem
        render_path = render_dir / real_path.name
        mask_path = mask_dir / f"{stem}.png"
        if not render_path.is_file() or not mask_path.is_file():
            continue

        render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if render is None or mask is None or render.shape[:2] != mask.shape:
            continue

        render_support = render.max(axis=2) > render_threshold
        mask_support = mask >= mask_threshold
        common_pixels = int(np.logical_and(render_support, mask_support).sum())
        if common_pixels < min_common_pixels:
            continue

        samples.append(
            DoorPairSample(
                stem=stem,
                real_path=real_path,
                render_path=render_path,
                mask_path=mask_path,
                common_pixels=common_pixels,
            )
        )

    if not samples:
        raise RuntimeError(
            f"No valid real/render/mask triplets found below {root}. "
            "Check the directory layout and thresholds."
        )
    return samples


def split_door_pairs(
    samples: Sequence[DoorPairSample],
    val_fraction: float = 0.2,
    block_size: int = 10,
    seed: int = 42,
) -> tuple[list[DoorPairSample], list[DoorPairSample]]:
    """Split temporal data by blocks instead of leaking adjacent frames.

    Numeric stems are grouped into blocks (for example, 000100--000109).  A
    whole block is assigned to either train or validation.  Non-numeric stems
    are grouped by their full stem.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    grouped = {}
    for sample in samples:
        if sample.stem.isdigit():
            group = f"numeric-{int(sample.stem) // block_size}"
        else:
            group = f"stem-{sample.stem}"
        grouped.setdefault(group, []).append(sample)

    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    num_val_groups = min(
        len(groups) - 1,
        max(1, round(len(groups) * val_fraction)),
    )
    val_groups = set(groups[:num_val_groups])

    train, val = [], []
    for group, group_samples in grouped.items():
        (val if group in val_groups else train).extend(group_samples)
    train.sort(key=lambda sample: sample.stem)
    val.sort(key=lambda sample: sample.stem)
    return train, val


def _resize_triplet(
    real: np.ndarray,
    render: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = size
    if real.shape[1] == width and real.shape[0] == height:
        return real, render, mask
    real = cv2.resize(real, (width, height), interpolation=cv2.INTER_AREA)
    render = cv2.resize(render, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return real, render, mask


def _mask_centered_crop(
    real: np.ndarray,
    render: np.ndarray,
    mask: np.ndarray,
    common: np.ndarray,
    output_size: tuple[int, int],
    rng: np.random.Generator,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    scale = float(rng.uniform(min_scale, 1.0))
    crop_width = max(32, round(width * scale))
    crop_height = max(32, round(height * scale))

    target_aspect = output_size[0] / output_size[1]
    if crop_width / crop_height > target_aspect:
        crop_width = round(crop_height * target_aspect)
    else:
        crop_height = round(crop_width / target_aspect)
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    ys, xs = np.nonzero(common)
    if len(xs):
        selected = int(rng.integers(0, len(xs)))
        center_x, center_y = int(xs[selected]), int(ys[selected])
    else:
        center_x, center_y = width // 2, height // 2

    jitter_x = int(rng.uniform(-0.15, 0.15) * crop_width)
    jitter_y = int(rng.uniform(-0.15, 0.15) * crop_height)
    x0 = int(np.clip(center_x + jitter_x - crop_width // 2, 0, width - crop_width))
    y0 = int(np.clip(center_y + jitter_y - crop_height // 2, 0, height - crop_height))
    x1, y1 = x0 + crop_width, y0 + crop_height

    return _resize_triplet(
        real[y0:y1, x0:x1],
        render[y0:y1, x0:x1],
        mask[y0:y1, x0:x1],
        output_size,
    )


def _random_gray(
    image: np.ndarray, rng: np.random.Generator, augment: bool
) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    if not augment:
        weights = np.full(3, 1.0 / 3.0, dtype=np.float32)
    else:
        draw = float(rng.random())
        if draw < 0.25:
            weights = np.eye(3, dtype=np.float32)[int(rng.integers(0, 3))]
        elif draw < 0.75:
            weights = rng.dirichlet(np.full(3, 0.7)).astype(np.float32)
        else:
            weights = np.full(3, 1.0 / 3.0, dtype=np.float32)
    return np.tensordot(image, weights, axes=([-1], [0])).astype(np.float32)


def _motion_blur(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    kernel_size = int(rng.choice([5, 7, 9, 11, 13, 15]))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    angle = float(rng.uniform(0.0, np.pi))
    radius = (kernel_size - 1) / 2.0
    center = (kernel_size - 1) / 2.0
    dx, dy = np.cos(angle) * radius, np.sin(angle) * radius
    p0 = (round(center - dx), round(center - dy))
    p1 = (round(center + dx), round(center + dy))
    cv2.line(kernel, p0, p1, 1.0, 1)
    kernel /= max(float(kernel.sum()), 1.0)
    return cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _photometric_augment(
    gray: np.ndarray,
    rng: np.random.Generator,
    domain: str,
) -> np.ndarray:
    gamma = float(rng.uniform(0.45, 2.2))
    gray = np.power(np.clip(gray, 0.0, 1.0), gamma)

    contrast = float(rng.uniform(0.45, 1.65))
    brightness = float(rng.uniform(-0.18, 0.18))
    gray = (gray - gray.mean()) * contrast + gray.mean() + brightness

    # Rendered text/edges are much sharper than the moving real camera, so the
    # render branch receives blur more often.  Real images still get blur/noise
    # to cover motion and exposure changes.
    blur_probability = 0.70 if domain == "render" else 0.45
    if rng.random() < blur_probability:
        if rng.random() < 0.55:
            sigma = float(rng.uniform(0.3, 2.8 if domain == "render" else 2.0))
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            gray = _motion_blur(gray, rng)

    height, width = gray.shape
    if rng.random() < 0.35:
        # Smooth multiplicative lighting gradient/shadow.
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        field = xx * np.cos(angle) + yy * np.sin(angle)
        gray *= 1.0 + float(rng.uniform(-0.45, 0.45)) * field

    if domain == "real" and rng.random() < 0.25:
        # A soft saturated blob approximates lamp glare/reflection on the door.
        glare = np.zeros_like(gray)
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (
            max(5, int(rng.uniform(0.03, 0.18) * width)),
            max(5, int(rng.uniform(0.03, 0.18) * height)),
        )
        cv2.ellipse(glare, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        sigma = max(3.0, float(max(axes)) * 0.45)
        glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=sigma)
        gray += glare * float(rng.uniform(0.15, 0.65))

    if rng.random() < 0.55:
        noise_sigma = float(rng.uniform(0.0, 0.055 if domain == "real" else 0.035))
        gray += rng.normal(0.0, noise_sigma, size=gray.shape).astype(np.float32)

    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def apply_real_mask(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = "hard",
    threshold: int = 64,
) -> np.ndarray:
    """Mask an RGB/BGR real image while preserving its dtype."""

    if mode == "hard":
        alpha = (mask >= threshold).astype(np.float32)
    elif mode == "soft":
        alpha = mask.astype(np.float32) / 255.0
    elif mode == "none":
        return image
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return (image.astype(np.float32) * alpha[..., None]).astype(image.dtype)


class DoorPairDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[DoorPairSample],
        output_size: tuple[int, int] = (640, 512),
        augment: bool = True,
        background_mode: str = "mixed",
        mask_threshold: int = 64,
        render_threshold: int = 3,
        crop_probability: float = 0.75,
        min_crop_scale: float = 0.65,
        seed: int = 42,
    ) -> None:
        if background_mode not in {"original", "masked", "soft", "mixed"}:
            raise ValueError(f"Unknown background_mode: {background_mode}")
        self.samples = list(samples)
        self.output_size = output_size
        self.augment = augment
        self.background_mode = background_mode
        self.mask_threshold = mask_threshold
        self.render_threshold = render_threshold
        self.crop_probability = crop_probability
        self.min_crop_scale = min_crop_scale
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        # seed_worker initializes NumPy independently in every DataLoader
        # worker.  Drawing a fresh seed here keeps augmentation changing when
        # the same frame is revisited in later epochs.
        seed = (
            int(np.random.randint(0, 2**32 - 1)) + self.seed * 1_000_003 + index
        ) % (2**32)
        return np.random.default_rng(seed)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rng = self._rng(index)
        real = _read_color(sample.real_path)
        render = _read_color(sample.render_path)
        mask_u8 = _read_mask(sample.mask_path)

        if real.shape != render.shape or real.shape[:2] != mask_u8.shape:
            raise RuntimeError(f"Shape mismatch for pair {sample.stem}")

        render_support = render.max(axis=2) > self.render_threshold
        mask_support = mask_u8 >= self.mask_threshold
        common = np.logical_and(render_support, mask_support)

        if self.augment and rng.random() < self.crop_probability:
            real, render, mask_u8 = _mask_centered_crop(
                real,
                render,
                mask_u8,
                common,
                self.output_size,
                rng,
                self.min_crop_scale,
            )
        else:
            real, render, mask_u8 = _resize_triplet(
                real, render, mask_u8, self.output_size
            )

        render_support = render.max(axis=2) > self.render_threshold
        mask_support = mask_u8 >= self.mask_threshold
        common = np.logical_and(render_support, mask_support).astype(np.float32)

        selected_background = self.background_mode
        if selected_background == "mixed":
            draw = float(rng.random()) if self.augment else 0.0
            selected_background = (
                "masked" if draw < 0.65 else ("soft" if draw < 0.85 else "original")
            )

        if selected_background == "masked":
            real_base = apply_real_mask(real, mask_u8, "hard", self.mask_threshold)
        elif selected_background == "soft":
            real_base = apply_real_mask(real, mask_u8, "soft", self.mask_threshold)
        else:
            real_base = real

        teacher_real = _random_gray(real_base, rng, augment=False)
        teacher_render = _random_gray(render, rng, augment=False)
        real_gray = _random_gray(real_base, rng, augment=self.augment)
        render_gray = _random_gray(render, rng, augment=self.augment)
        if self.augment:
            real_gray = _photometric_augment(real_gray, rng, "real")
            render_gray = _photometric_augment(render_gray, rng, "render")

        # Keep the render background black.  When the real branch is masked,
        # enforce the same convention after additive noise/exposure jitter.
        render_gray *= render_support.astype(np.float32)
        if selected_background == "masked":
            real_gray *= mask_support.astype(np.float32)
        elif selected_background == "soft":
            real_gray *= mask_u8.astype(np.float32) / 255.0

        def as_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array))[None].float()

        return {
            "real": as_tensor(real_gray),
            "render": as_tensor(render_gray),
            "teacher_real": as_tensor(teacher_real),
            "teacher_render": as_tensor(teacher_render),
            "common_mask": as_tensor(common),
            "mask": as_tensor(mask_u8.astype(np.float32) / 255.0),
            "stem": sample.stem,
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def stems(samples: Iterable[DoorPairSample]) -> list[str]:
    return [sample.stem for sample in samples]
