"""Fine-tune XFeat on aligned real/Gaussian-rendered door pairs.

This training path is intentionally separate from the original MegaDepth/COCO
trainer.  It uses exact pixel alignment, a soft foreground mask, independent
appearance augmentation, and a frozen pretrained teacher so descriptors remain
compatible with the released XFeat LighterGlue weights.
CUDA_VISIBLE_DEVICES=7 python3 -m modules.training.finetune_door \
  --data_root data \
  --pretrained_weights checkpoints_xfeat/xfeat_synthetic_50000.pt \
  --output_dir checkpoints_door \
  --device cuda:0 \
  --steps 1200 \
  --batch_size 4 \
  --workers 4 \
  --train_scope late \
  --lr 2e-5 \
  --background_mode masked \
  --eval_background_mode masked \
  --save_every 100 \
  --val_every 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.dataset.door_pairs import (
    DoorPairDataset,
    discover_door_pairs,
    seed_worker,
    split_door_pairs,
    stems,
)
from modules.model import XFeatModel


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must be WIDTH,HEIGHT") from exc
    if width < 32 or height < 32 or width % 8 or height % 8:
        raise argparse.ArgumentTypeError(
            "width and height must be >=32 and divisible by 8"
        )
    return width, height


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask-aware XFeat fine-tuning for aligned real/render door pairs."
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--pretrained_weights", default="weights/xfeat.pt")
    parser.add_argument("--output_dir", default="checkpoints_door")
    parser.add_argument(
        "--resume", default=None, help="Training-state checkpoint to resume."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=parse_size, default=(640, 512))
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head_lr_scale", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument(
        "--train_scope",
        choices=("fusion", "late", "all"),
        default="late",
        help="fusion: fusion+reliability; late: block3-5+fusion+reliability; all: full descriptor backbone.",
    )
    parser.add_argument(
        "--train_keypoint_head",
        action="store_true",
        help="Also adapt keypoint offsets with teacher-anchored pair consistency.",
    )
    parser.add_argument(
        "--update_batchnorm",
        action="store_true",
        help="Update BatchNorm running statistics (normally unsafe for this small dataset).",
    )

    parser.add_argument("--mask_threshold", type=int, default=64)
    parser.add_argument("--render_threshold", type=int, default=3)
    parser.add_argument("--min_common_pixels", type=int, default=256)
    parser.add_argument("--coarse_mask_threshold", type=float, default=0.25)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_block_size", type=int, default=10)
    parser.add_argument(
        "--background_mode",
        choices=("original", "masked", "soft", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--eval_background_mode",
        choices=("original", "masked", "soft"),
        default="masked",
    )
    parser.add_argument("--crop_probability", type=float, default=0.75)
    parser.add_argument("--min_crop_scale", type=float, default=0.65)

    parser.add_argument("--points_per_pair", type=int, default=256)
    parser.add_argument("--min_points", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument(
        "--negative_radius",
        type=float,
        default=1.0,
        help="Do not treat nearby coarse cells as negatives (units: 8-pixel cells).",
    )
    parser.add_argument("--descriptor_weight", type=float, default=1.0)
    parser.add_argument("--positive_weight", type=float, default=0.20)
    parser.add_argument(
        "--reliability_weight",
        type=float,
        default=0.0,
        help="Optional door-vs-background reliability classification weight.",
    )
    parser.add_argument(
        "--reliability_anchor_weight",
        type=float,
        default=0.10,
        help="Preserve the pretrained keypoint ranking used by LighterGlue.",
    )
    parser.add_argument("--descriptor_anchor_weight", type=float, default=0.25)
    parser.add_argument("--keypoint_consistency_weight", type=float, default=0.10)
    parser.add_argument("--keypoint_anchor_weight", type=float, default=0.10)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_model_state(checkpoint) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dict")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not all(isinstance(key, str) for key in checkpoint):
        raise RuntimeError("Could not find a model state dict in checkpoint")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
    return checkpoint


def load_model_weights(model: nn.Module, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(extract_model_state(checkpoint), strict=True)


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()


def configure_trainable(
    model: XFeatModel,
    train_scope: str,
    train_keypoint_head: bool,
) -> tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    scope_modules = {
        "fusion": (model.block_fusion,),
        "late": (model.block3, model.block4, model.block5, model.block_fusion),
        "all": (
            model.skip1,
            model.block1,
            model.block2,
            model.block3,
            model.block4,
            model.block5,
            model.block_fusion,
        ),
    }[train_scope]
    for module in scope_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    head_modules = [model.heatmap_head]
    if train_keypoint_head:
        head_modules.append(model.keypoint_head)
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    backbone_parameters = [
        parameter
        for module in scope_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for module in head_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return backbone_parameters, head_parameters


def coarse_mask(mask: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=output_hw, mode="area")


def gradient_score(image: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    dx = F.pad((image[..., :, 1:] - image[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((image[..., 1:, :] - image[..., :-1, :]).abs(), (0, 0, 0, 1))
    return F.interpolate(dx + dy, size=output_hw, mode="area")


def masked_descriptor_loss(
    features_real: torch.Tensor,
    features_render: torch.Tensor,
    common_mask: torch.Tensor,
    real_image: torch.Tensor,
    render_image: torch.Tensor,
    points_per_pair: int,
    min_points: int,
    mask_threshold: float,
    temperature: float,
    negative_radius: float,
    random_sample: bool,
) -> dict[str, torch.Tensor]:
    _, _, height, width = features_real.shape
    valid_map = coarse_mask(common_mask, (height, width)) >= mask_threshold
    scores = (
        gradient_score(real_image, (height, width))
        + gradient_score(render_image, (height, width))
        + 0.02
    )

    losses = []
    positive_losses = []
    accuracies = []
    pck1_values = []
    errors = []
    used_points = []
    for batch_index in range(features_real.shape[0]):
        coordinates_yx = valid_map[batch_index, 0].nonzero(as_tuple=False)
        if len(coordinates_yx) < min_points:
            continue

        count = min(points_per_pair, len(coordinates_yx))
        point_scores = scores[
            batch_index, 0, coordinates_yx[:, 0], coordinates_yx[:, 1]
        ].float()
        if random_sample and len(coordinates_yx) > count:
            point_scores = point_scores.clamp_min(1e-4)
            selected = torch.multinomial(point_scores, count, replacement=False)
        elif len(coordinates_yx) > count:
            selected = torch.topk(point_scores, count, sorted=False).indices
        else:
            selected = torch.arange(len(coordinates_yx), device=coordinates_yx.device)
        coordinates_yx = coordinates_yx[selected]

        y, x = coordinates_yx[:, 0], coordinates_yx[:, 1]
        descriptor_real = F.normalize(
            features_real[batch_index, :, y, x].transpose(0, 1).float(), dim=-1
        )
        descriptor_render = F.normalize(
            features_render[batch_index, :, y, x].transpose(0, 1).float(), dim=-1
        )
        logits = descriptor_real @ descriptor_render.transpose(0, 1)
        logits = logits / temperature
        metric_logits = logits.detach()

        labels = torch.arange(count, device=logits.device)
        if negative_radius > 0:
            coordinates_xy = coordinates_yx[:, [1, 0]].float()
            distance = torch.cdist(coordinates_xy, coordinates_xy, p=float("inf"))
            nearby = (distance <= negative_radius) & ~torch.eye(
                count, dtype=torch.bool, device=logits.device
            )
            logits = logits.masked_fill(nearby, -1e4)

        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        )
        positive_losses.append(
            1.0 - (descriptor_real * descriptor_render).sum(-1).mean()
        )

        with torch.no_grad():
            prediction = metric_logits.argmax(dim=1)
            prediction_xy = coordinates_yx[prediction][:, [1, 0]].float()
            target_xy = coordinates_yx[:, [1, 0]].float()
            error = torch.linalg.vector_norm(prediction_xy - target_xy, dim=-1)
            accuracies.append((prediction == labels).float().mean())
            pck1_values.append((error <= 1.0).float().mean())
            errors.append(error.mean())
            used_points.append(torch.tensor(float(count), device=logits.device))

    if not losses:
        zero = features_real.sum() * 0.0
        return {
            "descriptor": zero,
            "positive": zero,
            "accuracy": zero.detach(),
            "pck1": zero.detach(),
            "error_cells": zero.detach(),
            "points": zero.detach(),
            "valid_pairs": zero.detach(),
        }

    return {
        "descriptor": torch.stack(losses).mean(),
        "positive": torch.stack(positive_losses).mean(),
        "accuracy": torch.stack(accuracies).mean(),
        "pck1": torch.stack(pck1_values).mean(),
        "error_cells": torch.stack(errors).mean(),
        "points": torch.stack(used_points).mean(),
        "valid_pairs": torch.tensor(float(len(losses)), device=features_real.device),
    }


def balanced_reliability_loss(
    reliability_real: torch.Tensor,
    reliability_render: torch.Tensor,
    common_mask: torch.Tensor,
    mask_threshold: float,
) -> torch.Tensor:
    target = coarse_mask(common_mask, reliability_real.shape[-2:]) >= mask_threshold
    side_losses = []
    for prediction in (reliability_real.float(), reliability_render.float()):
        for batch_index in range(prediction.shape[0]):
            positive = prediction[batch_index, 0][target[batch_index, 0]]
            negative = prediction[batch_index, 0][~target[batch_index, 0]]
            if not len(positive) or not len(negative):
                continue
            side_losses.append(
                0.5 * (-torch.log(positive.clamp_min(1e-6))).mean()
                + 0.5 * (-torch.log((1.0 - negative).clamp_min(1e-6))).mean()
            )
    if not side_losses:
        return reliability_real.sum() * 0.0
    return torch.stack(side_losses).mean()


def descriptor_anchor_loss(
    student_real: torch.Tensor,
    student_render: torch.Tensor,
    teacher_real: torch.Tensor,
    teacher_render: torch.Tensor,
    common_mask: torch.Tensor,
    mask_threshold: float,
) -> torch.Tensor:
    target = coarse_mask(common_mask, student_real.shape[-2:]) >= mask_threshold
    losses = []
    # Keep the real branch in the pretrained descriptor coordinate system and
    # explicitly make the render branch imitate the *real* teacher descriptor.
    # Anchoring render to teacher_render would preserve the domain gap that this
    # fine-tuning is supposed to remove.  The argument remains available for a
    # symmetric API and future diagnostics.
    del teacher_render
    for student, teacher in (
        (student_real, teacher_real),
        (student_render, teacher_real),
    ):
        cosine = (
            F.normalize(student.float(), dim=1) * F.normalize(teacher.float(), dim=1)
        ).sum(dim=1, keepdim=True)
        if target.any():
            losses.append((1.0 - cosine[target]).mean())
    if not losses:
        return student_real.sum() * 0.0
    return torch.stack(losses).mean()


def reliability_anchor_loss(
    student_real: torch.Tensor,
    student_render: torch.Tensor,
    teacher_real: torch.Tensor,
    teacher_render: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        F.l1_loss(student_real.float(), teacher_real.float())
        + F.l1_loss(student_render.float(), teacher_render.float())
    )


def keypoint_losses(
    student_real: torch.Tensor,
    student_render: torch.Tensor,
    teacher_real: torch.Tensor,
    teacher_render: torch.Tensor,
    common_mask: torch.Tensor,
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = coarse_mask(common_mask, student_real.shape[-2:]) >= mask_threshold
    if not target.any():
        zero = student_real.sum() * 0.0
        return zero, zero

    p_real = F.softmax(student_real.float(), dim=1).clamp_min(1e-7)
    p_render = F.softmax(student_render.float(), dim=1).clamp_min(1e-7)
    mean_probability = 0.5 * (p_real + p_render)
    js_map = 0.5 * (
        (p_real * (p_real.log() - mean_probability.log())).sum(dim=1, keepdim=True)
        + (p_render * (p_render.log() - mean_probability.log())).sum(
            dim=1, keepdim=True
        )
    )
    consistency = js_map[target].mean()

    teacher_probability_real = F.softmax(teacher_real.float(), dim=1)
    teacher_probability_render = F.softmax(teacher_render.float(), dim=1)
    anchor_real = (
        teacher_probability_real
        * (teacher_probability_real.clamp_min(1e-7).log() - p_real.log())
    ).sum(dim=1, keepdim=True)
    anchor_render = (
        teacher_probability_render
        * (teacher_probability_render.clamp_min(1e-7).log() - p_render.log())
    ).sum(dim=1, keepdim=True)
    anchor = 0.5 * (anchor_real[target].mean() + anchor_render[target].mean())
    return consistency, anchor


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_losses(
    model: XFeatModel,
    teacher: XFeatModel,
    batch: dict,
    args: argparse.Namespace,
    random_sample: bool,
) -> dict[str, torch.Tensor]:
    features_real, keypoints_real, reliability_real = model(batch["real"])
    features_render, keypoints_render, reliability_render = model(batch["render"])

    with torch.no_grad():
        teacher_features_real, teacher_keypoints_real, teacher_reliability_real = (
            teacher(batch["teacher_real"])
        )
        (
            teacher_features_render,
            teacher_keypoints_render,
            teacher_reliability_render,
        ) = teacher(batch["teacher_render"])

    losses = masked_descriptor_loss(
        features_real,
        features_render,
        batch["common_mask"],
        batch["real"],
        batch["render"],
        points_per_pair=args.points_per_pair,
        min_points=args.min_points,
        mask_threshold=args.coarse_mask_threshold,
        temperature=args.temperature,
        negative_radius=args.negative_radius,
        random_sample=random_sample,
    )
    losses["reliability"] = balanced_reliability_loss(
        reliability_real,
        reliability_render,
        batch["common_mask"],
        args.coarse_mask_threshold,
    )
    losses["descriptor_anchor"] = descriptor_anchor_loss(
        features_real,
        features_render,
        teacher_features_real,
        teacher_features_render,
        batch["common_mask"],
        args.coarse_mask_threshold,
    )
    losses["reliability_anchor"] = reliability_anchor_loss(
        reliability_real,
        reliability_render,
        teacher_reliability_real,
        teacher_reliability_render,
    )
    if args.train_keypoint_head:
        consistency, anchor = keypoint_losses(
            keypoints_real,
            keypoints_render,
            teacher_keypoints_real,
            teacher_keypoints_render,
            batch["common_mask"],
            args.coarse_mask_threshold,
        )
    else:
        consistency = features_real.sum() * 0.0
        anchor = features_real.sum() * 0.0
    losses["keypoint_consistency"] = consistency
    losses["keypoint_anchor"] = anchor
    losses["total"] = (
        args.descriptor_weight * losses["descriptor"]
        + args.positive_weight * losses["positive"]
        + args.reliability_weight * losses["reliability"]
        + args.reliability_anchor_weight * losses["reliability_anchor"]
        + args.descriptor_anchor_weight * losses["descriptor_anchor"]
        + args.keypoint_consistency_weight * losses["keypoint_consistency"]
        + args.keypoint_anchor_weight * losses["keypoint_anchor"]
    )
    return losses


@torch.inference_mode()
def evaluate_dense(
    model: XFeatModel,
    teacher: XFeatModel,
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    teacher.eval()
    totals = defaultdict(float)
    batches = 0
    for batch_index, batch in enumerate(data_loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            losses = forward_losses(model, teacher, batch, args, random_sample=False)
        for key, value in losses.items():
            totals[key] += float(value.detach())
        batches += 1
    if not batches:
        return {}
    return {key: value / batches for key, value in totals.items()}


def save_raw_model(model: XFeatModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_training_state(
    model: XFeatModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    step: int,
    best_pck1: float,
    args: argparse.Namespace,
    path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_pck1": best_pck1,
            "args": vars(args),
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = device.type == "cuda" and not args.no_amp
    if args.dry_run:
        args.steps = min(args.steps, 3)
        args.workers = 0
        args.val_every = 1
        args.save_every = max(args.steps, 1)

    samples = discover_door_pairs(
        args.data_root,
        mask_threshold=args.mask_threshold,
        render_threshold=args.render_threshold,
        min_common_pixels=args.min_common_pixels,
    )
    train_samples, val_samples = split_door_pairs(
        samples,
        val_fraction=args.val_fraction,
        block_size=args.split_block_size,
        seed=args.seed,
    )
    print(
        f"Found {len(samples)} valid pairs: {len(train_samples)} train / "
        f"{len(val_samples)} validation"
    )

    train_dataset = DoorPairDataset(
        train_samples,
        output_size=args.image_size,
        augment=True,
        background_mode=args.background_mode,
        mask_threshold=args.mask_threshold,
        render_threshold=args.render_threshold,
        crop_probability=args.crop_probability,
        min_crop_scale=args.min_crop_scale,
        seed=args.seed,
    )
    val_dataset = DoorPairDataset(
        val_samples,
        output_size=args.image_size,
        augment=False,
        background_mode=args.eval_background_mode,
        mask_threshold=args.mask_threshold,
        render_threshold=args.render_threshold,
        crop_probability=0.0,
        min_crop_scale=1.0,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )

    model = XFeatModel().to(device)
    teacher = XFeatModel().to(device)
    load_model_weights(model, args.pretrained_weights)
    load_model_weights(teacher, args.pretrained_weights)
    teacher.eval()
    teacher.requires_grad_(False)

    backbone_parameters, head_parameters = configure_trainable(
        model, args.train_scope, args.train_keypoint_head
    )
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.lr})
    if head_parameters:
        parameter_groups.append(
            {"params": head_parameters, "lr": args.lr * args.head_lr_scale}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def learning_rate_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-3, (step + 1) / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_step = 0
    best_pck1 = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(extract_model_state(state), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step = int(state.get("step", 0))
        best_pck1 = float(state.get("best_pck1", -1.0))
        print(f"Resumed at step {start_step}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "train": stems(train_samples),
                "validation": stems(val_samples),
                "args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    writer = SummaryWriter(
        output_dir / "logdir" / time.strftime("door_%Y_%m_%d-%H_%M_%S")
    )

    num_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Device: {device}; trainable parameters: {num_trainable:,}; AMP: {amp_enabled}"
    )
    print(f"Training output: {output_dir.resolve()}")

    train_iterator = iter(train_loader)
    running = defaultdict(float)
    model.train()
    if not args.update_batchnorm:
        freeze_batch_norm(model)

    progress_bar = tqdm.trange(
        start_step + 1, args.steps + 1, initial=start_step, total=args.steps
    )
    for step in progress_bar:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            losses = forward_losses(model, teacher, batch, args, random_sample=True)

        if float(losses["valid_pairs"].detach()) < 1:
            continue
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for key, value in losses.items():
            running[key] += float(value.detach())
        if step % args.log_every == 0 or step == start_step + 1:
            divisor = 1 if step == start_step + 1 else args.log_every
            averaged = {key: value / divisor for key, value in running.items()}
            running.clear()
            for key, value in averaged.items():
                writer.add_scalar(f"train/{key}", value, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            progress_bar.set_postfix(
                loss=f"{averaged.get('total', 0):.3f}",
                exact=f"{averaged.get('accuracy', 0):.3f}",
                pck1=f"{averaged.get('pck1', 0):.3f}",
            )

        if step % args.val_every == 0 or step == args.steps:
            validation = evaluate_dense(
                model, teacher, val_loader, device, args, amp_enabled
            )
            for key, value in validation.items():
                writer.add_scalar(f"validation/{key}", value, step)
            validation_pck1 = validation.get("pck1", -1.0)
            print(
                f"\nvalidation step={step}: loss={validation.get('total', float('nan')):.4f} "
                f"exact={validation.get('accuracy', float('nan')):.3f} "
                f"pck@8px={validation_pck1:.3f} "
                f"error={validation.get('error_cells', float('nan')) * 8.0:.2f}px"
            )
            if validation_pck1 > best_pck1:
                best_pck1 = validation_pck1
                save_raw_model(model, output_dir / "best.pt")
            model.train()
            if not args.update_batchnorm:
                freeze_batch_norm(model)

        if step % args.save_every == 0 or step == args.steps:
            save_raw_model(model, output_dir / f"door_step_{step:06d}.pt")
            save_raw_model(model, output_dir / "last.pt")
            save_training_state(
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                best_pck1,
                args,
                output_dir / "last_state.pt",
            )

    writer.close()
    print(f"Finished. Best dense validation PCK@8px: {best_pck1:.4f}")


if __name__ == "__main__":
    main(parse_arguments())
