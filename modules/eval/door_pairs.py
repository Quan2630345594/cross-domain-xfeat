"""Evaluate XFeat or XFeat+LighterGlue on aligned door pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from modules.dataset.door_pairs import discover_door_pairs, split_door_pairs
from modules.xfeat import XFeat


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers"
        ) from exc
    if not thresholds or any(threshold <= 0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be positive")
    return thresholds


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Door-pair XFeat evaluation")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--weights", default="weights/xfeat.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_block_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_threshold", type=int, default=64)
    parser.add_argument("--render_threshold", type=int, default=3)
    parser.add_argument("--min_common_pixels", type=int, default=256)
    parser.add_argument(
        "--real_input",
        choices=("original", "masked", "soft"),
        default="masked",
    )
    parser.add_argument(
        "--filter_keypoints",
        choices=("none", "mask", "intersection", "separate"),
        default="separate",
        help=(
            "Feature filtering mode. 'separate' uses mask_real for real features "
            "and the dilated non-black render foreground for render features."
        ),
    )
    parser.add_argument("--render_mask_dilation", type=int, default=5)
    parser.add_argument(
        "--matcher", choices=("mnn", "lighterglue", "both"), default="both"
    )
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--detection_threshold", type=float, default=0.01)
    parser.add_argument("--mnn_min_cossim", type=float, default=0.0)
    parser.add_argument("--lighterglue_min_conf", type=float, default=0.05)
    parser.add_argument(
        "--pixel_thresholds", type=parse_thresholds, default=[5.0, 8.0, 12.0]
    )
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def prepare_real(
    image: np.ndarray, mask: np.ndarray, mode: str, threshold: int
) -> np.ndarray:
    if mode == "original":
        return image
    if mode == "masked":
        alpha = (mask >= threshold).astype(np.float32)
    else:
        alpha = mask.astype(np.float32) / 255.0
    return (image.astype(np.float32) * alpha[..., None]).astype(np.uint8)


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )


def filter_features(features: dict, support: np.ndarray, threshold: int = 1) -> dict:
    if not len(features["keypoints"]):
        return features
    coordinates = features["keypoints"].detach().round().long()
    coordinates[:, 0].clamp_(0, support.shape[1] - 1)
    coordinates[:, 1].clamp_(0, support.shape[0] - 1)
    values = support[coordinates[:, 1].cpu().numpy(), coordinates[:, 0].cpu().numpy()]
    keep = torch.from_numpy(values >= threshold).to(coordinates.device)
    return {
        key: value[keep] if key in {"keypoints", "scores", "descriptors"} else value
        for key, value in features.items()
    }


def match_mnn(xfeat: XFeat, features0: dict, features1: dict, min_cossim: float):
    if not len(features0["keypoints"]) or not len(features1["keypoints"]):
        return np.empty((0, 2)), np.empty((0, 2))
    index0, index1 = xfeat.match(
        features0["descriptors"], features1["descriptors"], min_cossim=min_cossim
    )
    return (
        features0["keypoints"][index0].detach().cpu().numpy(),
        features1["keypoints"][index1].detach().cpu().numpy(),
    )


def match_lighterglue(
    xfeat: XFeat,
    features0: dict,
    features1: dict,
    min_confidence: float,
):
    if len(features0["keypoints"]) < 2 or len(features1["keypoints"]) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    try:
        points0, points1, _ = xfeat.match_lighterglue(
            features0, features1, min_conf=min_confidence
        )
        return points0, points1
    except (IndexError, RuntimeError):
        return np.empty((0, 2)), np.empty((0, 2))


def summarize(rows: Sequence[dict], thresholds: Sequence[float]) -> dict:
    num_pairs = len(rows)
    total_matches = sum(row["matches"] for row in rows)
    summary = {
        "pairs": num_pairs,
        "empty_pairs": sum(row["matches"] == 0 for row in rows),
        "matches_per_pair": total_matches / max(num_pairs, 1),
        "median_matches": float(np.median([row["matches"] for row in rows]))
        if rows
        else 0.0,
    }
    for threshold in thresholds:
        key = f"correct@{threshold:g}px"
        total_correct = sum(row[key] for row in rows)
        per_pair = np.asarray([row[key] for row in rows], dtype=np.float32)
        summary[f"correct_per_pair@{threshold:g}px"] = total_correct / max(num_pairs, 1)
        summary[f"precision@{threshold:g}px"] = total_correct / max(total_matches, 1)
        summary[f"median_correct@{threshold:g}px"] = (
            float(np.median(per_pair)) if len(per_pair) else 0.0
        )
        summary[f"pair_success_5matches@{threshold:g}px"] = (
            float(np.mean(per_pair >= 5)) if len(per_pair) else 0.0
        )
    return summary


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
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
    if args.split == "train":
        samples = train_samples
    elif args.split == "val":
        samples = val_samples
    if args.max_pairs:
        samples = samples[: args.max_pairs]

    xfeat = XFeat(
        weights=args.weights,
        top_k=args.top_k,
        detection_threshold=args.detection_threshold,
    )
    # XFeat selects its own CUDA device; keep image tensors on that same device.
    device = xfeat.dev

    matchers = [args.matcher] if args.matcher != "both" else ["mnn", "lighterglue"]
    if "lighterglue" in matchers:
        from modules.lighterglue import LighterGlue

        xfeat.lighterglue = LighterGlue()
        # Pruning can reduce a tiny, mask-filtered set to zero in Kornia 0.8.x.
        xfeat.lighterglue.net.conf.width_confidence = -1
        xfeat.lighterglue.net.conf.depth_confidence = -1

    all_rows = {matcher: [] for matcher in matchers}
    for sample in tqdm.tqdm(samples, desc=f"door-{args.split}"):
        real = cv2.imread(str(sample.real_path), cv2.IMREAD_COLOR)
        render = cv2.imread(str(sample.render_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if real is None or render is None or mask is None:
            continue
        real = prepare_real(real, mask, args.real_input, args.mask_threshold)

        features_real = xfeat.detectAndCompute(
            image_tensor(real, device),
            top_k=args.top_k,
            detection_threshold=args.detection_threshold,
        )[0]
        features_render = xfeat.detectAndCompute(
            image_tensor(render, device),
            top_k=args.top_k,
            detection_threshold=args.detection_threshold,
        )[0]

        if args.filter_keypoints != "none":
            real_support = (mask >= args.mask_threshold).astype(np.uint8)
            render_support = (render.max(axis=2) > args.render_threshold).astype(
                np.uint8
            )
            if args.filter_keypoints == "intersection":
                common_support = np.logical_and(
                    real_support, render_support
                ).astype(np.uint8)
                real_support = common_support
                render_support = common_support
            elif args.filter_keypoints == "mask":
                render_support = real_support
            else:
                radius = max(args.render_mask_dilation, 0)
                if radius:
                    kernel_size = 2 * radius + 1
                    render_support = cv2.dilate(
                        render_support,
                        np.ones((kernel_size, kernel_size), dtype=np.uint8),
                    )
            features_real = filter_features(features_real, real_support)
            features_render = filter_features(features_render, render_support)

        image_size = (real.shape[1], real.shape[0])
        features_real["image_size"] = image_size
        features_render["image_size"] = image_size

        for matcher in matchers:
            if matcher == "mnn":
                points0, points1 = match_mnn(
                    xfeat, features_real, features_render, args.mnn_min_cossim
                )
            else:
                points0, points1 = match_lighterglue(
                    xfeat,
                    features_real,
                    features_render,
                    args.lighterglue_min_conf,
                )
            errors = (
                np.linalg.norm(points0 - points1, axis=1)
                if len(points0)
                else np.empty(0, dtype=np.float32)
            )
            row = {
                "stem": sample.stem,
                "keypoints_real": len(features_real["keypoints"]),
                "keypoints_render": len(features_render["keypoints"]),
                "matches": len(errors),
                "mean_error_px": float(errors.mean()) if len(errors) else None,
                "median_error_px": float(np.median(errors)) if len(errors) else None,
            }
            for threshold in args.pixel_thresholds:
                row[f"correct@{threshold:g}px"] = int((errors <= threshold).sum())
            all_rows[matcher].append(row)

    result = {
        "weights": str(Path(args.weights).resolve()),
        "split": args.split,
        "real_input": args.real_input,
        "filter_keypoints": args.filter_keypoints,
        "render_mask_dilation": args.render_mask_dilation,
        "thresholds_px": args.pixel_thresholds,
        "matchers": {
            matcher: {
                "summary": summarize(rows, args.pixel_thresholds),
                "pairs": rows,
            }
            for matcher, rows in all_rows.items()
        },
    }
    for matcher, matcher_result in result["matchers"].items():
        print(f"\n{matcher}:")
        print(json.dumps(matcher_result["summary"], indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main(parse_arguments())
