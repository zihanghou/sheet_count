from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "output_height_pitch"
LABEL_FILE = IMG_DIR / "num.txt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TIE_BREAK_PRIORITY = ("global", "left", "right")


@dataclass(frozen=True)
class PitchCounterConfig:
    roi_x1_ratio: float = 0.00
    roi_x2_ratio: float = 0.93
    roi_y1_ratio: float = 0.05
    roi_y2_ratio: float = 0.80
    bright_threshold: int = 150
    row_bright_fraction: float = 0.20
    gaussian_blur: int = 3
    row_smooth_window: int = 15
    vertical_close_kernel: int = 25
    sheet_pitch_px: int = 110
    min_region_threshold: int = 80
    max_region_threshold: int = 245


@dataclass(frozen=True)
class RegionCountResult:
    region: str
    count: int
    total_thickness: int
    all_runs: list[tuple[int, int]]
    roi: tuple[int, int, int, int]
    threshold_used: float
    gray: np.ndarray
    row_profile: np.ndarray
    row_mask_before_close: np.ndarray
    row_mask_after_close: np.ndarray


@dataclass(frozen=True)
class VoteResult:
    final_count: int
    vote_reason: str
    results: dict[str, RegionCountResult]


def image_paths() -> list[Path]:
    return sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_expected_counts() -> list[int]:
    if not LABEL_FILE.exists():
        return []
    return [int(line.strip()) for line in LABEL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def all_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile, kernel, mode="same")


def count_region(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float,
    config: PitchCounterConfig,
    region_name: str,
) -> RegionCountResult:
    x1, y1, x2, y2 = roi
    region_img = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)

    if config.gaussian_blur > 1:
        gray = cv2.GaussianBlur(gray, (config.gaussian_blur, config.gaussian_blur), 0)

    bright_pixels_by_row = (gray >= threshold).mean(axis=1)
    row_profile = smooth_profile(bright_pixels_by_row, config.row_smooth_window)
    row_mask_before_close = row_profile >= config.row_bright_fraction

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.vertical_close_kernel))
    row_mask_after_close = (
        cv2.morphologyEx((row_mask_before_close.astype(np.uint8) * 255)[:, None], cv2.MORPH_CLOSE, close_kernel)[:, 0] > 0
    )

    runs = all_true_runs(row_mask_after_close)
    total_thickness = sum(end - start + 1 for start, end in runs)
    count = max(0, int(round(total_thickness / config.sheet_pitch_px)))
    abs_runs = [(start + y1, end + y1) for start, end in runs]

    return RegionCountResult(
        region=region_name,
        count=count,
        total_thickness=total_thickness,
        all_runs=abs_runs,
        roi=roi,
        threshold_used=float(threshold),
        gray=gray,
        row_profile=row_profile,
        row_mask_before_close=row_mask_before_close,
        row_mask_after_close=row_mask_after_close,
    )


def threshold_percentile(global_gray: np.ndarray, base_threshold: float) -> float:
    return float((global_gray <= base_threshold).mean() * 100.0)


def region_threshold(image: np.ndarray, roi: tuple[int, int, int, int], percentile: float, config: PitchCounterConfig) -> float:
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if config.gaussian_blur > 1:
        gray = cv2.GaussianBlur(gray, (config.gaussian_blur, config.gaussian_blur), 0)
    threshold = float(np.percentile(gray, percentile))
    return float(np.clip(threshold, config.min_region_threshold, config.max_region_threshold))


def choose_by_priority(candidates: list[int], results: dict[str, RegionCountResult]) -> int:
    for region in TIE_BREAK_PRIORITY:
        if results[region].count in candidates:
            return results[region].count
    return candidates[0]


def vote_counts(results: dict[str, RegionCountResult]) -> VoteResult:
    counts = [results["left"].count, results["right"].count, results["global"].count]
    freq = Counter(counts)
    mode_freq = max(freq.values())
    mode_candidates = sorted([value for value, count in freq.items() if count == mode_freq])

    if len(mode_candidates) == 1:
        return VoteResult(mode_candidates[0], f"unique_mode_{mode_candidates[0]}", results)

    median_value = float(np.median(np.array(counts, dtype=np.float32)))
    distances = {candidate: abs(candidate - median_value) for candidate in mode_candidates}
    min_distance = min(distances.values())
    nearest_candidates = sorted([candidate for candidate, distance in distances.items() if distance == min_distance])
    if len(nearest_candidates) == 1:
        return VoteResult(nearest_candidates[0], f"mode_tie_median_nearest_{nearest_candidates[0]}", results)

    chosen = choose_by_priority(nearest_candidates, results)
    return VoteResult(chosen, f"mode_tie_priority_{chosen}", results)


def base_roi(image: np.ndarray, config: PitchCounterConfig) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1 = int(width * config.roi_x1_ratio)
    x2 = int(width * config.roi_x2_ratio)
    y1 = int(height * config.roi_y1_ratio)
    y2 = int(height * config.roi_y2_ratio)
    return x1, y1, x2, y2


def split_rois(roi: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    mid = x1 + width // 2
    return {"left": (x1, y1, mid, y2), "right": (mid, y1, x2, y2), "global": (x1, y1, x2, y2)}


def count_with_vote(image: np.ndarray, config: PitchCounterConfig) -> VoteResult:
    roi = base_roi(image, config)
    rois = split_rois(roi)
    gx1, gy1, gx2, gy2 = rois["global"]
    global_gray = cv2.cvtColor(image[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
    if config.gaussian_blur > 1:
        global_gray = cv2.GaussianBlur(global_gray, (config.gaussian_blur, config.gaussian_blur), 0)

    percentile = threshold_percentile(global_gray, config.bright_threshold)
    thresholds = {name: region_threshold(image, region_roi, percentile, config) for name, region_roi in rois.items()}
    results = {name: count_region(image, rois[name], thresholds[name], config, name) for name in ("left", "right", "global")}
    return vote_counts(results)


def draw_overview(image: np.ndarray, vote: VoteResult, expected: int | None, output_path: Path) -> None:
    debug = image.copy()
    x1, y1, x2, y2 = vote.results["global"].roi
    mid = x1 + (x2 - x1) // 2
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.line(debug, (mid, y1), (mid, y2), (255, 200, 0), 2)

    for start, end in vote.results["global"].all_runs:
        cv2.line(debug, (x1, start), (x2, start), (0, 255, 0), 1)
        cv2.line(debug, (x1, end), (x2, end), (0, 0, 255), 1)

    row1 = (
        f"L={vote.results['left'].count} R={vote.results['right'].count} G={vote.results['global'].count} "
        f"TL={vote.results['left'].total_thickness} TR={vote.results['right'].total_thickness} TG={vote.results['global'].total_thickness}"
    )
    row2 = (
        f"final={vote.final_count} reason={vote.vote_reason} "
        f"thr(L/R/G)={vote.results['left'].threshold_used:.1f}/{vote.results['right'].threshold_used:.1f}/{vote.results['global'].threshold_used:.1f}"
    )
    if expected is not None:
        row2 += f" expected={expected} err={abs(vote.final_count - expected)}"

    cv2.putText(debug, row1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(debug, row2, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def render_profile_panel(result: RegionCountResult, config: PitchCounterConfig, width: int = 460, height: int = 260) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    x0, y0, x1, y1 = 40, 28, width - 20, height - 38
    cv2.rectangle(panel, (x0, y0), (x1, y1), (90, 90, 90), 1)
    frac_y = int(y1 - config.row_bright_fraction * (y1 - y0))
    cv2.line(panel, (x0, frac_y), (x1, frac_y), (0, 170, 255), 1)

    profile = np.clip(result.row_profile, 0.0, 1.0)
    n = len(profile)
    if n > 1:
        points = []
        for i, value in enumerate(profile):
            px = int(x0 + i * (x1 - x0) / (n - 1))
            py = int(y1 - value * (y1 - y0))
            points.append((px, py))
        cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (80, 255, 80), 1)

    title = f"{result.region} cnt={result.count} total={result.total_thickness} thr={result.threshold_used:.1f}"
    cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
    cv2.putText(panel, f"runs={len(result.all_runs)}", (x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    return panel


def draw_runs_overlay(image: np.ndarray, result: RegionCountResult, output_path: Path) -> None:
    x1, y1, x2, y2 = result.roi
    patch = image[y1:y2, x1:x2].copy()
    local_runs = [(start - y1, end - y1) for start, end in result.all_runs]
    for idx, (start, end) in enumerate(local_runs, start=1):
        cv2.line(patch, (0, start), (patch.shape[1] - 1, start), (0, 255, 0), 1)
        cv2.line(patch, (0, end), (patch.shape[1] - 1, end), (0, 0, 255), 1)
        cy = max(12, min(patch.shape[0] - 5, (start + end) // 2))
        cv2.putText(patch, str(idx), (4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)
    cv2.imwrite(str(output_path), patch)


def write_reports(
    summary_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    params: PitchCounterConfig,
    total_abs_error: int,
    hit_count: int,
    total_count: int,
) -> None:
    reports_dir = OUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_headers = [
        "image",
        "expected",
        "left_count",
        "right_count",
        "global_count",
        "left_total",
        "right_total",
        "global_total",
        "left_thr",
        "right_thr",
        "global_thr",
        "final_count",
        "vote_reason",
        "error",
    ]
    with open(reports_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    region_headers = [
        "image",
        "region",
        "count",
        "total_thickness",
        "threshold",
        "run_count",
        "runs",
    ]
    with open(reports_dir / "regions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=region_headers)
        writer.writeheader()
        writer.writerows(region_rows)

    with open(reports_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("height_pitch_export summary\n")
        f.write(f"images={total_count}\n")
        f.write(f"hit_count={hit_count}\n")
        f.write(f"accuracy={hit_count/total_count:.4f}\n")
        f.write(f"total_abs_error={total_abs_error}\n")
        f.write(f"params={params}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export-rich height pitch counting")
    parser.add_argument("--sheet-pitch", type=int, default=None, help="Override sheet pitch parameter")
    parser.add_argument("--bright-threshold", type=int, default=None, help="Override bright threshold")
    parser.add_argument("--row-bright-fraction", type=float, default=None, help="Override row bright fraction")
    parser.add_argument("--vertical-close-kernel", type=int, default=None, help="Override close kernel size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PitchCounterConfig()
    if args.sheet_pitch is not None:
        config = replace(config, sheet_pitch_px=args.sheet_pitch)
    if args.bright_threshold is not None:
        config = replace(config, bright_threshold=args.bright_threshold)
    if args.row_bright_fraction is not None:
        config = replace(config, row_bright_fraction=args.row_bright_fraction)
    if args.vertical_close_kernel is not None:
        config = replace(config, vertical_close_kernel=args.vertical_close_kernel)

    paths = image_paths()
    labels = read_expected_counts()
    if not paths:
        raise SystemExit("No image files found in images/.")
    if labels and len(labels) != len(paths):
        raise SystemExit(f"Expected {len(paths)} labels in {LABEL_FILE}, found {len(labels)}.")

    overview_dir = OUT_DIR / "overview"
    inter_dir = OUT_DIR / "intermediate"
    reports_dir = OUT_DIR / "reports"
    for d in (overview_dir, inter_dir, reports_dir):
        if d.exists():
            shutil.rmtree(d)
    overview_dir.mkdir(parents=True, exist_ok=True)
    inter_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[int] = []
    total_abs_error = 0
    hit_count = 0
    summary_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []

    print("Algorithm: total bright thickness with left/right/global voting (export mode)")
    print(f"Parameters: {config}")
    print()

    for idx, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        vote = count_with_vote(image, config)
        expected = labels[idx] if labels else None
        predictions.append(vote.final_count)

        if expected is not None:
            error = abs(vote.final_count - expected)
            total_abs_error += error
            if error == 0:
                hit_count += 1
        else:
            error = None

        draw_overview(image, vote, expected, overview_dir / f"{path.stem}_overview.jpg")

        image_dir = inter_dir / path.stem
        for region_name in ("left", "right", "global"):
            region = vote.results[region_name]
            region_dir = image_dir / region_name
            region_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(region_dir / "01_gray.jpg"), region.gray)
            cv2.imwrite(str(region_dir / "02_row_mask_before_close.jpg"), (region.row_mask_before_close.astype(np.uint8) * 255)[:, None])
            cv2.imwrite(str(region_dir / "03_row_mask_after_close.jpg"), (region.row_mask_after_close.astype(np.uint8) * 255)[:, None])
            cv2.imwrite(str(region_dir / "04_profile_panel.jpg"), render_profile_panel(region, config))
            draw_runs_overlay(image, region, region_dir / "05_runs_overlay.jpg")
            with open(region_dir / "06_runs.txt", "w", encoding="utf-8") as f:
                f.write(f"region={region_name}\n")
                f.write(f"count={region.count}\n")
                f.write(f"total_thickness={region.total_thickness}\n")
                f.write(f"threshold_used={region.threshold_used:.4f}\n")
                f.write(f"run_count={len(region.all_runs)}\n")
                f.write(f"runs={region.all_runs}\n")

            region_rows.append(
                {
                    "image": path.name,
                    "region": region_name,
                    "count": region.count,
                    "total_thickness": region.total_thickness,
                    "threshold": round(region.threshold_used, 4),
                    "run_count": len(region.all_runs),
                    "runs": str(region.all_runs),
                }
            )

        summary_rows.append(
            {
                "image": path.name,
                "expected": expected if expected is not None else "",
                "left_count": vote.results["left"].count,
                "right_count": vote.results["right"].count,
                "global_count": vote.results["global"].count,
                "left_total": vote.results["left"].total_thickness,
                "right_total": vote.results["right"].total_thickness,
                "global_total": vote.results["global"].total_thickness,
                "left_thr": round(vote.results["left"].threshold_used, 3),
                "right_thr": round(vote.results["right"].threshold_used, 3),
                "global_thr": round(vote.results["global"].threshold_used, 3),
                "final_count": vote.final_count,
                "vote_reason": vote.vote_reason,
                "error": error if error is not None else "",
            }
        )

        print(
            f"{path.name}: "
            f"L/R/G={vote.results['left'].count}/{vote.results['right'].count}/{vote.results['global'].count}, "
            f"TL/TR/TG={vote.results['left'].total_thickness}/{vote.results['right'].total_thickness}/{vote.results['global'].total_thickness}, "
            f"T(L/R/G)={vote.results['left'].threshold_used:.1f}/{vote.results['right'].threshold_used:.1f}/{vote.results['global'].threshold_used:.1f}, "
            f"final={vote.final_count}, reason={vote.vote_reason}, error={error}"
        )

    print()
    print(f"Predictions: {predictions}")
    if labels:
        print(f"Expected:    {labels}")
        print(f"Total absolute error: {total_abs_error}")
        print(f"Hit rate: {hit_count}/{len(labels)}")

    write_reports(summary_rows, region_rows, config, total_abs_error, hit_count, len(paths))
    print(f"Overview images: {overview_dir}")
    print(f"Intermediate images: {inter_dir}")
    print(f"Reports: {OUT_DIR / 'reports'}")


if __name__ == "__main__":
    main()
