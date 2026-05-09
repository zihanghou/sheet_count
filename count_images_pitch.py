from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs" / "pitch_debug"
LABEL_FILE = IMG_DIR / "num.txt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


@dataclass(frozen=True)
class CountResult:
    count: int
    stack_height: int
    stack_bounds_y: tuple[int, int] | None
    roi: tuple[int, int, int, int]


def image_paths() -> list[Path]:
    return sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_expected_counts() -> list[int]:
    if not LABEL_FILE.exists():
        return []
    return [int(line.strip()) for line in LABEL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
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

    if not runs:
        return None
    return max(runs, key=lambda item: item[1] - item[0])


def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile, kernel, mode="same")


def count_sheets_by_stack_pitch(image: np.ndarray, config: PitchCounterConfig) -> CountResult:
    height, width = image.shape[:2]
    x1 = int(width * config.roi_x1_ratio)
    x2 = int(width * config.roi_x2_ratio)
    y1 = int(height * config.roi_y1_ratio)
    y2 = int(height * config.roi_y2_ratio)

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if config.gaussian_blur > 1:
        gray = cv2.GaussianBlur(gray, (config.gaussian_blur, config.gaussian_blur), 0)

    bright_pixels_by_row = (gray >= config.bright_threshold).mean(axis=1)
    row_profile = smooth_profile(bright_pixels_by_row, config.row_smooth_window)
    row_mask = row_profile >= config.row_bright_fraction

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.vertical_close_kernel))
    row_mask = cv2.morphologyEx((row_mask.astype(np.uint8) * 255)[:, None], cv2.MORPH_CLOSE, close_kernel)
    row_mask = row_mask[:, 0] > 0

    run = longest_true_run(row_mask)
    if run is None:
        return CountResult(count=0, stack_height=0, stack_bounds_y=None, roi=(x1, y1, x2, y2))

    top, bottom = run
    stack_height = bottom - top + 1
    count = max(0, int(round(stack_height / config.sheet_pitch_px)))
    return CountResult(
        count=count,
        stack_height=stack_height,
        stack_bounds_y=(top + y1, bottom + y1),
        roi=(x1, y1, x2, y2),
    )


def draw_debug(image: np.ndarray, result: CountResult, expected: int | None, output_path: Path) -> None:
    debug = image.copy()
    x1, y1, x2, y2 = result.roi
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if result.stack_bounds_y is not None:
        top, bottom = result.stack_bounds_y
        cv2.line(debug, (x1, top), (x2, top), (0, 255, 0), 3)
        cv2.line(debug, (x1, bottom), (x2, bottom), (0, 0, 255), 3)

    label = f"pred={result.count}"
    if expected is not None:
        label += f" expected={expected}"
    label += f" height={result.stack_height}"
    cv2.putText(debug, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def main() -> None:
    config = PitchCounterConfig()
    paths = image_paths()
    expected_counts = read_expected_counts()

    if not paths:
        raise SystemExit("No image files found in images/.")

    if expected_counts and len(expected_counts) != len(paths):
        raise SystemExit(f"Expected {len(paths)} labels in {LABEL_FILE}, found {len(expected_counts)}.")

    total_abs_error = 0
    predictions: list[int] = []

    print("Algorithm: bright stack height divided by calibrated sheet pitch")
    print(f"Parameters: {config}")
    print()

    for idx, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")

        result = count_sheets_by_stack_pitch(image, config)
        expected = expected_counts[idx] if expected_counts else None
        predictions.append(result.count)

        if expected is None:
            status = "n/a"
            error = 0
        else:
            error = abs(result.count - expected)
            total_abs_error += error
            status = "OK" if error == 0 else "FAIL"

        draw_debug(image, result, expected, OUT_DIR / f"{path.stem}_pitch_debug.jpg")
        print(
            f"{path.name}: pred={result.count}, expected={expected}, "
            f"height={result.stack_height}, bounds={result.stack_bounds_y}, status={status}"
        )

    print()
    print(f"Predictions: {predictions}")
    if expected_counts:
        print(f"Expected:    {expected_counts}")
        print(f"Total absolute error: {total_abs_error}")
    print(f"Debug images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
