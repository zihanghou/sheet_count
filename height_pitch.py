from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
        region_name,
        count,
        total_thickness,
        abs_runs,
        roi,
        float(threshold),
        gray,
        row_profile,
        row_mask_before_close,
        row_mask_after_close,
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
    results = {name: count_region(image, region_roi, thresholds[name], config, name) for name, region_roi in rois.items()}
    return vote_counts(results)


def draw_debug(image: np.ndarray, vote: VoteResult, expected: int | None, output_path: Path) -> None:
    debug = image.copy()
    x1, y1, x2, y2 = vote.results["global"].roi
    width = x2 - x1
    mid = x1 + width // 2
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


def render_profile_panel(
    result: RegionCountResult,
    config: PitchCounterConfig,
    width: int = 420,
    height: int = 220,
) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    x0, y0, x1, y1 = 40, 20, width - 20, height - 30
    cv2.rectangle(panel, (x0, y0), (x1, y1), (90, 90, 90), 1)

    threshold_line_y = int(y1 - config.row_bright_fraction * (y1 - y0))
    cv2.line(panel, (x0, threshold_line_y), (x1, threshold_line_y), (0, 170, 255), 1)

    profile = np.clip(result.row_profile, 0.0, 1.0)
    n = len(profile)
    if n > 1:
        points = []
        for i, value in enumerate(profile):
            px = int(x0 + i * (x1 - x0) / (n - 1))
            py = int(y1 - value * (y1 - y0))
            points.append((px, py))
        cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (80, 255, 80), 1)

    title = (
        f"{result.region}: cnt={result.count} total={result.total_thickness} "
        f"thr={result.threshold_used:.1f} runs={len(result.all_runs)}"
    )
    cv2.putText(panel, title, (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.putText(panel, "line: row_profile  orange: row_bright_fraction", (10, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    return panel


def write_intermediate_outputs(
    image: np.ndarray,
    vote: VoteResult,
    expected: int | None,
    image_stem: str,
    output_root: Path,
    config: PitchCounterConfig,
) -> None:
    image_dir = output_root / image_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    draw_debug(image, vote, expected, image_dir / "00_overview_total.jpg")

    for name in ("left", "right", "global"):
        result = vote.results[name]
        region_dir = image_dir / name
        region_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(region_dir / "01_gray.jpg"), result.gray)
        cv2.imwrite(str(region_dir / "02_row_mask_before_close.jpg"), (result.row_mask_before_close.astype(np.uint8) * 255)[:, None])
        cv2.imwrite(str(region_dir / "03_row_mask_after_close.jpg"), (result.row_mask_after_close.astype(np.uint8) * 255)[:, None])
        cv2.imwrite(str(region_dir / "04_profile_panel.jpg"), render_profile_panel(result, config))

        with open(region_dir / "05_runs.txt", "w", encoding="utf-8") as f:
            f.write(f"region={name}\n")
            f.write(f"count={result.count}\n")
            f.write(f"total_thickness={result.total_thickness}\n")
            f.write(f"threshold_used={result.threshold_used:.4f}\n")
            f.write(f"runs={result.all_runs}\n")


def main() -> None:
    config = PitchCounterConfig()
    paths = image_paths()
    labels = read_expected_counts()

    if not paths:
        raise SystemExit("No image files found in images/.")
    if labels and len(labels) != len(paths):
        raise SystemExit(f"Expected {len(paths)} labels in {LABEL_FILE}, found {len(labels)}.")

    predictions: list[int] = []
    total_abs_error = 0
    summary_rows: list[dict[str, object]] = []

    print("Algorithm: total bright thickness with left/right/global voting")
    print(f"Parameters: {config}")
    print()

    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")

        vote = count_with_vote(image, config)
        expected = labels[index] if labels else None
        predictions.append(vote.final_count)

        if expected is not None:
            error = abs(vote.final_count - expected)
            total_abs_error += error
            status = "OK" if error == 0 else "FAIL"
        else:
            status = "n/a"

        draw_debug(image, vote, expected, OUT_DIR / f"{path.stem}_total_debug.jpg")
        write_intermediate_outputs(image, vote, expected, path.stem, OUT_DIR / "intermediate", config)
        print(
            f"{path.name}: "
            f"left_count={vote.results['left'].count}, right_count={vote.results['right'].count}, global_count={vote.results['global'].count}, "
            f"left_total={vote.results['left'].total_thickness}, right_total={vote.results['right'].total_thickness}, global_total={vote.results['global'].total_thickness}, "
            f"final_count={vote.final_count}, vote_reason={vote.vote_reason}, expected={expected}, status={status}"
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
                "error": abs(vote.final_count - expected) if expected is not None else "",
            }
        )

    print()
    print(f"Predictions: {predictions}")
    if labels:
        print(f"Expected:    {labels}")
        print(f"Total absolute error: {total_abs_error}")

    summary_path = OUT_DIR / "summary_total.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
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
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to: {summary_path}")
    print(f"Debug images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
