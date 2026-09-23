"""Render strict XFeat + LighterGlue matches for every real/render door pair."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tqdm

from modules.eval.door_pairs import filter_features, image_tensor
from modules.lighterglue import LighterGlue
from modules.xfeat import XFeat


@dataclass
class Supports:
    real: np.ndarray | None
    render: np.ndarray | None
    render_raw: np.ndarray
    source: str
    mask_pixels: int
    render_pixels: int
    common_pixels: int


@dataclass
class GeometryResult:
    matrix: np.ndarray | None
    candidate_inliers: np.ndarray
    verified_inliers: np.ndarray
    reprojection_errors: np.ndarray
    status: str
    method: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render strictly masked and geometrically verified LighterGlue matches "
            "for all image_real/image_render pairs"
        )
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--weights", default="checkpoints_door_formal/xfeat_door_final.pt"
    )
    parser.add_argument(
        "--output_dir", default="checkpoints_door_formal/all_frame_matches"
    )
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--detection_threshold", type=float, default=0.01)
    parser.add_argument("--lighterglue_min_conf", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=int, default=64)
    parser.add_argument("--render_threshold", type=int, default=3)
    parser.add_argument("--render_mask_dilation", type=int, default=5)
    parser.add_argument("--min_support_pixels", type=int, default=256)
    parser.add_argument("--ransac_threshold", type=float, default=5.0)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)
    parser.add_argument("--ransac_max_iters", type=int, default=10000)
    parser.add_argument("--min_ransac_inliers", type=int, default=8)
    parser.add_argument("--max_draw_inliers", type=int, default=180)
    parser.add_argument("--max_draw_outliers", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--contact_sheet_columns", type=int, default=5)
    parser.add_argument("--contact_sheet_rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_mask.png", f"{stem}.png"):
        path = mask_dir / name
        if path.exists():
            return path
    return None


def dilate_support(support: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return support.astype(np.uint8)
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(support.astype(np.uint8), kernel)


def select_supports(
    mask: np.ndarray | None,
    render: np.ndarray,
    mask_threshold: int,
    render_threshold: int,
    render_mask_dilation: int,
    min_support_pixels: int,
) -> Supports:
    render_raw = (render.max(axis=2) > render_threshold).astype(np.uint8)
    render_support = dilate_support(render_raw, render_mask_dilation)
    if mask is None:
        mask_support = np.zeros(render.shape[:2], dtype=np.uint8)
    else:
        mask_support = (mask >= mask_threshold).astype(np.uint8)

    mask_pixels = int(mask_support.sum())
    render_pixels = int(render_raw.sum())
    common_pixels = int(np.logical_and(mask_support, render_raw).sum())
    if render_pixels < min_support_pixels:
        return Supports(
            real=None,
            render=None,
            render_raw=render_raw,
            source="none",
            mask_pixels=mask_pixels,
            render_pixels=render_pixels,
            common_pixels=common_pixels,
        )
    if mask_pixels >= min_support_pixels:
        return Supports(
            real=mask_support,
            render=render_support,
            render_raw=render_raw,
            source="separate_masks",
            mask_pixels=mask_pixels,
            render_pixels=render_pixels,
            common_pixels=common_pixels,
        )
    return Supports(
        real=render_support,
        render=render_support,
        render_raw=render_raw,
        source="render_fallback",
        mask_pixels=mask_pixels,
        render_pixels=render_pixels,
        common_pixels=common_pixels,
    )


def extract_features(
    xfeat: XFeat,
    real: np.ndarray,
    render: np.ndarray,
    real_support: np.ndarray,
    render_support: np.ndarray,
    top_k: int,
    detection_threshold: float,
) -> tuple[dict, dict]:
    real_input = (real.astype(np.float32) * real_support[..., None]).astype(
        np.uint8
    )
    features_real = xfeat.detectAndCompute(
        image_tensor(real_input, xfeat.dev),
        top_k=top_k,
        detection_threshold=detection_threshold,
    )[0]
    features_render = xfeat.detectAndCompute(
        image_tensor(render, xfeat.dev),
        top_k=top_k,
        detection_threshold=detection_threshold,
    )[0]
    features_real = filter_features(features_real, real_support)
    features_render = filter_features(features_render, render_support)
    image_size = (real.shape[1], real.shape[0])
    features_real["image_size"] = image_size
    features_render["image_size"] = image_size
    return features_real, features_render


def run_matcher(
    xfeat: XFeat,
    features_real: dict,
    features_render: dict,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(features_real["keypoints"]) < 2
        or len(features_render["keypoints"]) < 2
    ):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    points_real, points_render, _ = xfeat.match_lighterglue(
        features_real,
        features_render,
        min_conf=min_confidence,
    )
    return points_real.astype(np.float32), points_render.astype(np.float32)


def empty_geometry(num_matches: int, status: str, method: str) -> GeometryResult:
    return GeometryResult(
        matrix=None,
        candidate_inliers=np.zeros(num_matches, dtype=bool),
        verified_inliers=np.zeros(num_matches, dtype=bool),
        reprojection_errors=np.full(num_matches, np.inf, dtype=np.float32),
        status=status,
        method=method,
    )


def verify_homography(
    points_real: np.ndarray,
    points_render: np.ndarray,
    threshold: float,
    confidence: float,
    max_iters: int,
    min_inliers: int,
) -> GeometryResult:
    num_matches = len(points_real)
    method_name = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC"
    if num_matches < 4:
        return empty_geometry(num_matches, "insufficient_matches", method_name)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        matrix, mask = cv2.findHomography(
            points_real,
            points_render,
            method=method,
            ransacReprojThreshold=threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    except cv2.error:
        return empty_geometry(num_matches, "model_error", method_name)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return empty_geometry(num_matches, "model_failed", method_name)

    homogeneous = np.concatenate(
        [points_real.astype(np.float64), np.ones((num_matches, 1))], axis=1
    )
    projected = (matrix @ homogeneous.T).T
    valid = np.abs(projected[:, 2]) > 1e-8
    projected_xy = np.full_like(points_real, np.nan, dtype=np.float64)
    projected_xy[valid] = projected[valid, :2] / projected[valid, 2:3]
    errors = np.linalg.norm(projected_xy - points_render, axis=1).astype(np.float32)
    errors[~np.isfinite(errors)] = np.inf
    candidate_inliers = mask.reshape(-1).astype(bool) & np.isfinite(errors)
    if int(candidate_inliers.sum()) < min_inliers:
        return GeometryResult(
            matrix=matrix,
            candidate_inliers=candidate_inliers,
            verified_inliers=np.zeros(num_matches, dtype=bool),
            reprojection_errors=errors,
            status="weak_model",
            method=method_name,
        )
    return GeometryResult(
        matrix=matrix,
        candidate_inliers=candidate_inliers,
        verified_inliers=candidate_inliers.copy(),
        reprojection_errors=errors,
        status="verified",
        method=method_name,
    )


def points_inside(points: np.ndarray, support: np.ndarray | None) -> np.ndarray:
    if support is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    coordinates = np.rint(points).astype(np.int64)
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, support.shape[1] - 1)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, support.shape[0] - 1)
    return support[coordinates[:, 1], coordinates[:, 0]] > 0


def sample_indices(
    indices: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    if maximum <= 0 or not len(indices):
        return np.empty(0, dtype=np.int64)
    if len(indices) <= maximum:
        return indices
    return np.sort(rng.choice(indices, maximum, replace=False))


def draw_correspondences(
    image: np.ndarray,
    points_real: np.ndarray,
    points_render: np.ndarray,
    indices: np.ndarray,
    width: int,
    header: int,
    colors: list[tuple[int, int, int]] | tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    for index in indices:
        point_real = tuple(
            np.rint(points_real[index] + np.array([0, header])).astype(int)
        )
        point_render = tuple(
            np.rint(points_render[index] + np.array([width, header])).astype(int)
        )
        color = colors[index] if isinstance(colors, list) else colors
        cv2.line(image, point_real, point_render, color, thickness, cv2.LINE_AA)
        cv2.circle(image, point_real, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, point_render, radius, color, -1, cv2.LINE_AA)


def draw_support_contour(
    canvas: np.ndarray,
    support: np.ndarray | None,
    x_offset: int,
    y_offset: int,
    color: tuple[int, int, int],
) -> None:
    if support is None:
        return
    contours, _ = cv2.findContours(
        support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shifted = [
        contour + np.array([[[x_offset, y_offset]]], dtype=contour.dtype)
        for contour in contours
    ]
    cv2.drawContours(canvas, shifted, -1, color, 1, cv2.LINE_AA)


def render_matches(
    real: np.ndarray,
    render: np.ndarray,
    supports: Supports,
    points_real: np.ndarray,
    points_render: np.ndarray,
    geometry: GeometryResult,
    stem: str,
    status: str,
    max_draw_inliers: int,
    max_draw_outliers: int,
    ransac_threshold: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    height, width = real.shape[:2]
    header = 82
    canvas = np.zeros((height + header, width * 2, 3), dtype=np.uint8)
    canvas[header:, :width] = real
    canvas[header:, width:] = render
    draw_support_contour(canvas, supports.real, 0, header, (0, 215, 255))
    draw_support_contour(canvas, supports.render, width, header, (255, 210, 0))

    rng = np.random.default_rng(seed + int(stem))
    inside_render = points_inside(points_render, supports.render)
    outside_indices = np.flatnonzero(~inside_render)
    inlier_indices = np.flatnonzero(geometry.verified_inliers & inside_render)
    outlier_indices = np.flatnonzero(
        ~geometry.verified_inliers & inside_render
    )
    drawn_outside = sample_indices(outside_indices, max_draw_outliers, rng)
    drawn_outliers = sample_indices(outlier_indices, max_draw_outliers, rng)
    drawn_inliers = sample_indices(inlier_indices, max_draw_inliers, rng)

    diagnostic_overlay = canvas.copy()
    draw_correspondences(
        diagnostic_overlay,
        points_real,
        points_render,
        drawn_outliers,
        width,
        header,
        (45, 45, 210),
        1,
        2,
    )
    draw_correspondences(
        diagnostic_overlay,
        points_real,
        points_render,
        drawn_outside,
        width,
        header,
        (220, 0, 220),
        1,
        3,
    )
    canvas = cv2.addWeighted(diagnostic_overlay, 0.42, canvas, 0.58, 0)

    inlier_colors = [(40, 220, 40)] * len(points_real)
    for index in inlier_indices:
        if geometry.reprojection_errors[index] > min(3.0, ransac_threshold):
            inlier_colors[index] = (0, 215, 255)
    verified_overlay = canvas.copy()
    draw_correspondences(
        verified_overlay,
        points_real,
        points_render,
        drawn_inliers,
        width,
        header,
        inlier_colors,
        1,
        3,
    )
    canvas = cv2.addWeighted(verified_overlay, 0.88, canvas, 0.12, 0)

    raw_matches = len(points_real)
    verified_matches = int(geometry.verified_inliers.sum())
    candidate_inliers = int(geometry.candidate_inliers.sum())
    inlier_ratio = verified_matches / max(raw_matches, 1)
    identity_errors = (
        np.linalg.norm(points_real - points_render, axis=1)
        if raw_matches
        else np.empty(0, dtype=np.float32)
    )
    verified_errors = geometry.reprojection_errors[geometry.verified_inliers]
    summary = (
        f"frame={stem}  status={status}  geometry={geometry.status}  "
        f"raw={raw_matches}  verified={verified_matches} ({inlier_ratio:.1%})  "
        f"outside_render={len(outside_indices)}"
    )
    cv2.putText(
        canvas,
        summary,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "real: yellow contour | render: cyan contour",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green <=3px MAGSAC | yellow 3-5px | red sampled outlier | magenta outside",
        (width + 12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )

    metrics = {
        "raw_matches": raw_matches,
        "ransac_candidate_inliers": candidate_inliers,
        "verified_matches": verified_matches,
        "verified_inlier_ratio": inlier_ratio,
        "outside_render_support": int((~inside_render).sum()),
        "outside_exact_render_foreground": int(
            (~points_inside(points_render, supports.render_raw)).sum()
        ),
        "identity_correct_5px": int((identity_errors <= 5).sum()),
        "identity_correct_8px": int((identity_errors <= 8).sum()),
        "median_verified_reprojection_error_px": (
            float(np.median(verified_errors)) if len(verified_errors) else None
        ),
        "mean_verified_reprojection_error_px": (
            float(verified_errors.mean()) if len(verified_errors) else None
        ),
        "drawn_inliers": len(drawn_inliers),
        "drawn_outliers": len(drawn_outliers),
        "drawn_outside": len(drawn_outside),
    }
    return canvas, metrics


def make_contact_sheets(
    image_paths: list[Path], output_dir: Path, columns: int, rows: int
) -> list[Path]:
    page_size = columns * rows
    cell_width = 400
    cell_height = 190
    pages = []
    for page_index in range(math.ceil(len(image_paths) / page_size)):
        page_paths = image_paths[page_index * page_size : (page_index + 1) * page_size]
        sheet = np.zeros(
            (rows * cell_height, columns * cell_width, 3), dtype=np.uint8
        )
        for index, image_path in enumerate(page_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            resized = cv2.resize(
                image, (cell_width, cell_height), interpolation=cv2.INTER_AREA
            )
            row, column = divmod(index, columns)
            y0, x0 = row * cell_height, column * cell_width
            sheet[y0 : y0 + cell_height, x0 : x0 + cell_width] = resized
        page_path = output_dir / f"contact_sheet_{page_index + 1:03d}.jpg"
        cv2.imwrite(str(page_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(page_path)
    return pages


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gallery(rows: list[dict], contact_sheets: list[Path], path: Path) -> None:
    contacts = "\n".join(
        f'<a href="{html.escape(sheet.name)}">{html.escape(sheet.name)}</a>'
        for sheet in contact_sheets
    )
    cards = []
    for row in rows:
        visual = html.escape(row["visualization"])
        caption = (
            f"frame {html.escape(row['stem'])} | {html.escape(row['geometry_status'])} "
            f"| raw={row['raw_matches']} | verified={row['verified_matches']} "
            f"| ratio={float(row['verified_inlier_ratio']):.1%}"
        )
        cards.append(
            f'<figure><a href="{visual}"><img loading="lazy" src="{visual}"></a>'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Strict door matches</title>
<style>
body {{ background:#111; color:#eee; font:14px sans-serif; margin:20px; }}
a {{ color:#7ec8ff; margin-right:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:16px; }}
figure {{ margin:0; background:#1d1d1d; padding:8px; }}
img {{ width:100%; height:auto; }} figcaption {{ padding-top:7px; }}
</style></head><body><h1>Strict real/render door matches</h1>
<p>Green/yellow are verified homography inliers. Red lines are sampled raw outliers.</p>
<p>Contact sheets: {contacts}</p><div class="grid">{''.join(cards)}</div></body></html>"""
    path.write_text(document, encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    processed_rows = [row for row in rows if row["status"] == "ok"]
    total_raw = sum(int(row["raw_matches"]) for row in processed_rows)
    total_verified = sum(int(row["verified_matches"]) for row in processed_rows)
    return {
        "total_pairs": len(rows),
        "processed_pairs": len(processed_rows),
        "separate_mask_pairs": sum(
            row["support_source"] == "separate_masks" for row in rows
        ),
        "render_fallback_pairs": sum(
            row["support_source"] == "render_fallback" for row in rows
        ),
        "no_support_pairs": sum(row["support_source"] == "none" for row in rows),
        "pairs_with_raw_matches": sum(int(row["raw_matches"]) > 0 for row in rows),
        "geometrically_verified_pairs": sum(
            row["geometry_status"] == "verified" for row in rows
        ),
        "pairs_with_at_least_5_verified_matches": sum(
            int(row["verified_matches"]) >= 5 for row in rows
        ),
        "pairs_with_at_least_8_verified_matches": sum(
            int(row["verified_matches"]) >= 8 for row in rows
        ),
        "raw_matches_per_processed_pair": total_raw / max(len(processed_rows), 1),
        "verified_matches_per_processed_pair": total_verified
        / max(len(processed_rows), 1),
        "global_verified_inlier_ratio": total_verified / max(total_raw, 1),
        "matches_outside_dilated_render_support": sum(
            int(row["outside_render_support"]) for row in rows
        ),
        "errors": sum(row["status"] == "error" for row in rows),
    }


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    real_dir = data_root / "image_real"
    render_dir = data_root / "image_render"
    mask_dir = data_root / "mask_real"
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    match_dir = output_dir / "matches"
    image_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        path.stem
        for path in real_dir.glob("*.png")
        if (render_dir / path.name).exists()
    )
    if not stems:
        raise RuntimeError(f"No paired PNG images found under {data_root}")

    cv2.setRNGSeed(args.seed)
    xfeat = XFeat(
        weights=args.weights,
        top_k=args.top_k,
        detection_threshold=args.detection_threshold,
    )
    xfeat.lighterglue = LighterGlue()
    xfeat.lighterglue.net.conf.width_confidence = -1
    xfeat.lighterglue.net.conf.depth_confidence = -1

    rows = []
    visualization_paths = []
    for stem in tqdm.tqdm(stems, desc="render-strict-door-matches"):
        real_path = real_dir / f"{stem}.png"
        render_path = render_dir / f"{stem}.png"
        mask_path = find_mask(mask_dir, stem)
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        mask = (
            cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_path is not None
            else None
        )
        if real is None or render is None:
            raise RuntimeError(f"Failed to read pair {stem}")

        supports = select_supports(
            mask,
            render,
            args.mask_threshold,
            args.render_threshold,
            args.render_mask_dilation,
            args.min_support_pixels,
        )
        points_real = np.empty((0, 2), dtype=np.float32)
        points_render = np.empty((0, 2), dtype=np.float32)
        keypoints_real = 0
        keypoints_render = 0
        error_message = ""
        status = "no_support"
        geometry = empty_geometry(0, "not_run", "USAC_MAGSAC")
        if supports.real is not None and supports.render is not None:
            try:
                features_real, features_render = extract_features(
                    xfeat,
                    real,
                    render,
                    supports.real,
                    supports.render,
                    args.top_k,
                    args.detection_threshold,
                )
                keypoints_real = len(features_real["keypoints"])
                keypoints_render = len(features_render["keypoints"])
                points_real, points_render = run_matcher(
                    xfeat,
                    features_real,
                    features_render,
                    args.lighterglue_min_conf,
                )
                geometry = verify_homography(
                    points_real,
                    points_render,
                    args.ransac_threshold,
                    args.ransac_confidence,
                    args.ransac_max_iters,
                    args.min_ransac_inliers,
                )
                status = "ok"
            except (IndexError, RuntimeError) as error:
                status = "error"
                error_message = f"{type(error).__name__}: {error}"
                geometry = empty_geometry(len(points_real), "not_run", "USAC_MAGSAC")

        visualization, metrics = render_matches(
            real,
            render,
            supports,
            points_real,
            points_render,
            geometry,
            stem,
            status,
            args.max_draw_inliers,
            args.max_draw_outliers,
            args.ransac_threshold,
            args.seed,
        )
        visualization_path = image_dir / f"{stem}.jpg"
        written = cv2.imwrite(
            str(visualization_path),
            visualization,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )
        if not written:
            raise RuntimeError(f"Failed to write {visualization_path}")
        visualization_paths.append(visualization_path)

        match_path = match_dir / f"{stem}.npz"
        np.savez_compressed(
            match_path,
            points_real=points_real,
            points_render=points_render,
            verified_inliers=geometry.verified_inliers,
            candidate_inliers=geometry.candidate_inliers,
            reprojection_errors=geometry.reprojection_errors,
            homography=(
                geometry.matrix
                if geometry.matrix is not None
                else np.empty((0, 0), dtype=np.float64)
            ),
        )
        rows.append(
            {
                "stem": stem,
                "status": status,
                "support_source": supports.source,
                "geometry_status": geometry.status,
                "geometry_method": geometry.method,
                "mask_pixels": supports.mask_pixels,
                "render_pixels": supports.render_pixels,
                "common_pixels": supports.common_pixels,
                "mask_render_iou": supports.common_pixels
                / max(
                    supports.mask_pixels
                    + supports.render_pixels
                    - supports.common_pixels,
                    1,
                ),
                "keypoints_real": keypoints_real,
                "keypoints_render": keypoints_render,
                **metrics,
                "error": error_message,
                "visualization": str(visualization_path.relative_to(output_dir)),
                "match_file": str(match_path.relative_to(output_dir)),
            }
        )

    contact_sheets = make_contact_sheets(
        visualization_paths,
        output_dir,
        args.contact_sheet_columns,
        args.contact_sheet_rows,
    )
    summary = summarize(rows)
    result = {
        "weights": str(Path(args.weights).resolve()),
        "data_root": str(data_root.resolve()),
        "parameters": vars(args),
        "summary": summary,
        "frames": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(rows, output_dir / "metrics.csv")
    write_gallery(rows, contact_sheets, output_dir / "index.html")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {len(rows)} visualizations to {image_dir.resolve()}")
    print(f"Saved per-frame verified matches to {match_dir.resolve()}")


if __name__ == "__main__":
    main(parse_arguments())
