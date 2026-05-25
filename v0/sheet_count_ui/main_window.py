import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ops.device.hikvision_camera.camera import CameraWrapper
from sheet_count_ui.camera_debug_widget import CameraDebugWidget
from sheet_count_ui.detection_widget import DetectionWidget
from sheet_count_ui.pitch_calibration_widget import PitchCalibrationWidget


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "log"
IMAGE_DIR = ROOT / "images"
SCHEMA_VERSION = 1

PITCH_PX = 83.5
SEGMENTS = 6
VOTE_WEIGHTS = {
    "total": 10,
    "seg1": 2,
    "seg2": 2,
    "seg3": 3,
    "seg4": 4,
    "seg5": 4,
    "seg6": 2,
}
MIN_ROW_FRACTION = 0.01
MIN_COL_FRACTION = 0.005
MIN_COMPONENT_AREA = 80
MIN_COL_BRIGHT_PIXELS = 3

HAND_PROCESS_SCALE = 0.5
USE_HAND_COLUMN_EXCLUSION = False
USE_FAST_BRIGHT_CHANNEL = False
HAND_DILATE = 9
HAND_MIN_COMPONENT_AREA = 1800
HAND_KEEP_TOP_K = 3
HAND_TOP_RATIO = 0.34
NAIL_TOP_RATIO = 0.26
NAIL_NEAR_SKIN_DILATE = 25
HAND_COL_OCCLUSION_MAX = 0.22

PLATE_MIN_WIDTH_RATIO = 0.16
PLATE_MIN_ASPECT = 1.5
PLATE_MIN_AREA = 250
PLATE_BAND_MARGIN = 35

HEIGHT_OUTLIER_MAD_SCALE = 3.5
HEIGHT_PERCENTILE = 75

cv2.setUseOptimized(True)

# ====== auto detect params ======
AUTO_DETECT_INTERVAL_MS = 250
AUTO_PRESENT_FRAMES = 3
AUTO_EMPTY_FRAMES = 5
AUTO_PRESENCE_SCALE = 0.45
AUTO_STATE_IDLE = "IDLE"
AUTO_STATE_COUNTED = "COUNTED"



def post_process(mask):
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask


def remove_small_components(mask, min_area=80):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    out = np.zeros_like(mask, dtype=np.uint8)
    out[keep[labels]] = 255
    return out


def keep_plate_like_components(mask, img_shape):
    h, w = img_shape[:2]
    min_width = int(w * PLATE_MIN_WIDTH_RATIO)
    src = mask.astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, 8)
    if num <= 1:
        return np.zeros_like(src, dtype=np.uint8)

    areas = stats[:, cv2.CC_STAT_AREA]
    xs = stats[:, cv2.CC_STAT_LEFT]
    ys = stats[:, cv2.CC_STAT_TOP]
    bws = stats[:, cv2.CC_STAT_WIDTH]
    bhs = stats[:, cv2.CC_STAT_HEIGHT]
    x2s = xs + bws - 1
    y2s = ys + bhs - 1
    aspects = bws / np.maximum(1, bhs)

    valid_area = areas >= MIN_COMPONENT_AREA
    valid_area[0] = False

    anchor_ids = np.where(
        (areas >= PLATE_MIN_AREA)
        & (bws >= min_width)
        & (aspects >= PLATE_MIN_ASPECT)
    )[0]
    anchor_ids = anchor_ids[anchor_ids != 0]

    if len(anchor_ids) == 0:
        out = np.zeros_like(src, dtype=np.uint8)
        out[valid_area[labels]] = 255
        return out

    ax1 = max(0, int(xs[anchor_ids].min()) - 20)
    ay1 = max(0, int(ys[anchor_ids].min()) - PLATE_BAND_MARGIN)
    ax2 = min(w - 1, int(x2s[anchor_ids].max()) + 20)
    ay2 = min(h - 1, int(y2s[anchor_ids].max()) + PLATE_BAND_MARGIN)

    in_band = (
        (x2s >= ax1) & (xs <= ax2)
        & (y2s >= ay1) & (ys <= ay2)
        & valid_area
    )

    out = np.zeros_like(src, dtype=np.uint8)
    out[in_band[labels]] = 255
    return out


def _odd_kernel_size(value, min_value=3):
    value = max(min_value, int(round(value)))
    if value % 2 == 0:
        value += 1
    return value


def _raw_hand_mask_core(img, scale=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h, _ = img.shape[:2]

    hsv_skin = cv2.inRange(
        hsv,
        np.array([0, 20, 50], dtype=np.uint8),
        np.array([25, 190, 255], dtype=np.uint8),
    )
    ycrcb_skin = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 176, 127], dtype=np.uint8),
    )

    skin = cv2.bitwise_and(hsv_skin, ycrcb_skin)
    skin[int(h * HAND_TOP_RATIO):, :] = 0

    k_small_size = _odd_kernel_size(7 * scale)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small_size, k_small_size))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k_small)

    low_s_high_v = cv2.inRange(
        hsv,
        np.array([0, 0, 120], dtype=np.uint8),
        np.array([179, 85, 255], dtype=np.uint8),
    )
    nail_size = _odd_kernel_size(NAIL_NEAR_SKIN_DILATE * scale)
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nail_size, nail_size))
    near_skin = cv2.dilate(skin, near_kernel)
    nail = cv2.bitwise_and(low_s_high_v, near_skin)
    nail[int(h * NAIL_TOP_RATIO):, :] = 0

    mask = cv2.bitwise_or(skin, nail)
    hand_size = _odd_kernel_size(HAND_DILATE * scale)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hand_size, hand_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, k)
    return mask


def raw_hand_mask(img):
    scale = float(HAND_PROCESS_SCALE)
    if scale >= 0.999:
        return _raw_hand_mask_core(img, 1.0)

    h, w = img.shape[:2]
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = _raw_hand_mask_core(small, scale)
    return cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def keep_main_hand_components(mask, min_area=None):
    if min_area is None:
        min_area = HAND_MIN_COMPONENT_AREA
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    out = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        y = int(stats[i, cv2.CC_STAT_TOP])
        if area >= min_area:
            components.append((area - y * 2, i))

    components.sort(reverse=True)
    kept = components[:HAND_KEEP_TOP_K]

    if kept:
        keep_ids = np.zeros(num, dtype=bool)
        for _, i in kept:
            keep_ids[i] = True
        out[keep_ids[labels]] = 255

    return out, {
        "raw_components": num - 1,
        "kept_components": len(kept),
        "removed_components": max(0, num - 1 - len(kept)),
    }


def hand_masks(img):
    scale = float(HAND_PROCESS_SCALE)
    if scale < 0.999:
        h, w = img.shape[:2]
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        raw_small = _raw_hand_mask_core(small, scale)
        min_area = max(50, int(round(HAND_MIN_COMPONENT_AREA * scale * scale)))
        main_small, stats = keep_main_hand_components(raw_small, min_area=min_area)
        raw = cv2.resize(raw_small, (w, h), interpolation=cv2.INTER_NEAREST)
        main = cv2.resize(main_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return raw, main, stats

    raw = raw_hand_mask(img)
    main, stats = keep_main_hand_components(raw)
    return raw, main, stats


def bright_otsu_mask(img):
    if img.ndim == 2:
        v = img
    elif USE_FAST_BRIGHT_CHANNEL:
        v = img[:, :, 0]
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

    v = cv2.GaussianBlur(v, (5, 5), 0)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return post_process(mask)


def hsv_otsu_mask(img):
    return bright_otsu_mask(img)


def calc_bright_bbox(mask):
    fg = mask > 0
    h, w = fg.shape[:2]
    row_min = max(1, int(round(w * MIN_ROW_FRACTION)))
    col_min = max(1, int(round(h * MIN_COL_FRACTION)))

    ys = np.flatnonzero(np.count_nonzero(fg, axis=1) >= row_min)
    xs = np.flatnonzero(np.count_nonzero(fg, axis=0) >= col_min)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def empty_height_info():
    info = {
        "bright_x1": "",
        "bright_y1": "",
        "bright_x2": "",
        "bright_y2": "",
        "total_top_y": "",
        "total_bottom_y": "",
        "total_height_px": 0,
        "valid_cols": 0,
        "raw_valid_cols": 0,
        "excluded_hand_cols": 0,
        "outlier_cols": 0,
        "height_stat": f"p{HEIGHT_PERCENTILE}",
    }

    for i in range(SEGMENTS):
        info[f"seg{i + 1}_x1"] = ""
        info[f"seg{i + 1}_x2"] = ""
        info[f"seg{i + 1}_top_y"] = ""
        info[f"seg{i + 1}_bottom_y"] = ""
        info[f"seg{i + 1}_height_px"] = 0
        info[f"seg{i + 1}_valid_cols"] = 0

    return info


def column_extent_points(mask, hand_mask, x1, x2, min_col_pixels=MIN_COL_BRIGHT_PIXELS):
    h, w = mask.shape[:2]
    x1 = max(0, int(x1))
    x2 = min(w - 1, int(x2))
    if x1 > x2:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, np.array([], dtype=np.int32)

    fg = mask[:, x1:x2 + 1] > 0
    col_counts = np.count_nonzero(fg, axis=0)
    valid = col_counts >= min_col_pixels
    local_xs = np.flatnonzero(valid)

    if len(local_xs) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, np.array([], dtype=np.int32)

    tops_all = np.argmax(fg, axis=0)
    bottoms_all = h - 1 - np.argmax(fg[::-1, :], axis=0)

    xs = (local_xs + x1).astype(np.float32)
    tops = tops_all[local_xs].astype(np.float32)
    bottoms = bottoms_all[local_xs].astype(np.float32)
    excluded_hand_cols = np.array([], dtype=np.int32)

    if hand_mask is not None:
        hand = hand_mask[:, x1:x2 + 1] > 0
        prefix = np.zeros((h + 1, hand.shape[1]), dtype=np.int32)
        prefix[1:] = np.cumsum(hand, axis=0, dtype=np.int32)

        y1 = tops.astype(np.int32)
        y2 = bottoms.astype(np.int32)
        hand_pixels = prefix[y2 + 1, local_xs] - prefix[y1, local_xs]
        heights = np.maximum(1, y2 - y1 + 1)
        hand_ratio = hand_pixels / heights

        keep = hand_ratio <= HAND_COL_OCCLUSION_MAX
        excluded_hand_cols = xs[~keep].astype(np.int32)
        xs = xs[keep]
        tops = tops[keep]
        bottoms = bottoms[keep]

    return xs, tops, bottoms, excluded_hand_cols


def robust_line_angle(xs, ys):
    if len(xs) < 8:
        return 0.0

    try:
        slope, _ = np.polyfit(xs, ys, 1)
    except Exception:
        return 0.0

    return float(np.degrees(np.arctan(float(slope))))


def robust_height_filter(heights):
    if len(heights) < 8:
        return np.ones(len(heights), dtype=bool)

    median = float(np.median(heights))
    mad = float(np.median(np.abs(heights - median)))
    if mad < 1.0:
        return np.ones(len(heights), dtype=bool)

    return np.abs(heights - median) <= HEIGHT_OUTLIER_MAD_SCALE * mad


def fast_percentile(values, percentile):
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    k = int(round((len(arr) - 1) * float(percentile) / 100.0))
    k = max(0, min(len(arr) - 1, k))
    return float(np.partition(arr, k)[k])


def measure_column_group(xs, tops, bottoms):
    if len(xs) == 0:
        return "", "", 0, 0.0

    vertical_heights = bottoms - tops + 1.0
    raw_height = fast_percentile(vertical_heights, HEIGHT_PERCENTILE)

    top_angle = robust_line_angle(xs, tops)
    bottom_angle = robust_line_angle(xs, bottoms)
    edge_angle = float(np.median(np.array([top_angle, bottom_angle], dtype=np.float32)))
    corrected_height = raw_height * abs(float(np.cos(np.radians(edge_angle))))

    return (
        int(round(float(np.median(tops)))),
        int(round(float(np.median(bottoms)))),
        int(round(corrected_height)),
        edge_angle,
    )


def measure_region_height(mask, hand_mask, x1, x2):
    xs, tops, bottoms, excluded_hand_cols = column_extent_points(mask, hand_mask, x1, x2)

    if len(xs) == 0:
        return "", "", 0, 0.0, {
            "xs": np.array([], dtype=np.int32),
            "tops": np.array([], dtype=np.float32),
            "bottoms": np.array([], dtype=np.float32),
            "raw_valid_cols": 0,
            "valid_cols": 0,
            "excluded_hand_cols": int(len(excluded_hand_cols)),
            "outlier_cols": 0,
            "excluded_hand_xs": excluded_hand_cols,
        }

    heights = bottoms - tops + 1.0
    keep = robust_height_filter(heights)
    filtered_xs = xs[keep]
    filtered_tops = tops[keep]
    filtered_bottoms = bottoms[keep]

    top, bottom, height, edge_angle = measure_column_group(filtered_xs, filtered_tops, filtered_bottoms)

    return top, bottom, height, edge_angle, {
        "xs": filtered_xs.astype(np.int32),
        "tops": filtered_tops,
        "bottoms": filtered_bottoms,
        "raw_valid_cols": int(len(xs)),
        "valid_cols": int(len(filtered_xs)),
        "excluded_hand_cols": int(len(excluded_hand_cols)),
        "outlier_cols": int(len(xs) - len(filtered_xs)),
        "excluded_hand_xs": excluded_hand_cols,
    }


def calc_segment_heights(mask, hand_mask=None, hand_stats=None):
    bbox = calc_bright_bbox(mask)
    info = empty_height_info()

    if bbox is None:
        return info

    bx1, by1, bx2, by2 = bbox
    top, bottom, total_h, edge_angle, cols = measure_region_height(mask, hand_mask, bx1, bx2)
    xs = cols["xs"]

    info.update({
        "bright_x1": bx1,
        "bright_y1": by1,
        "bright_x2": bx2,
        "bright_y2": by2,
        "total_top_y": top,
        "total_bottom_y": bottom,
        "total_height_px": total_h,
        "edge_angle_deg": round(edge_angle, 4),
        "valid_cols": cols["valid_cols"],
        "raw_valid_cols": cols["raw_valid_cols"],
        "excluded_hand_cols": cols["excluded_hand_cols"],
        "outlier_cols": cols["outlier_cols"],
        "valid_xs": xs.tolist(),
        "excluded_hand_xs": cols["excluded_hand_xs"].astype(int).tolist(),
    })

    if hand_stats is not None:
        info.update({
            "raw_hand_components": hand_stats["raw_components"],
            "kept_hand_components": hand_stats["kept_components"],
            "raw_hand_removed_components": hand_stats["removed_components"],
        })

    if len(xs) == 0:
        return info

    order = np.argsort(xs)
    xs = xs[order]
    tops = cols["tops"][order]
    bottoms = cols["bottoms"][order]

    for i in range(SEGMENTS):
        start = int(len(xs) * i / SEGMENTS)
        end = int(len(xs) * (i + 1) / SEGMENTS)

        if start >= end:
            continue

        group_xs = xs[start:end]
        group_tops = tops[start:end]
        group_bottoms = bottoms[start:end]
        top, bottom, height, seg_angle = measure_column_group(group_xs, group_tops, group_bottoms)

        info[f"seg{i + 1}_x1"] = int(group_xs[0])
        info[f"seg{i + 1}_x2"] = int(group_xs[-1])
        info[f"seg{i + 1}_top_y"] = top
        info[f"seg{i + 1}_bottom_y"] = bottom
        info[f"seg{i + 1}_height_px"] = height
        info[f"seg{i + 1}_edge_angle_deg"] = round(seg_angle, 4)
        info[f"seg{i + 1}_valid_cols"] = int(len(group_xs))

    return info


def count_from_height(height_px, pitch_px):
    if height_px == "" or height_px <= 0:
        return 0
    return max(0, int(round(float(height_px) / float(pitch_px))))


def height_count_candidates(info, pitch_px):
    candidates = []

    total_count = count_from_height(info["total_height_px"], pitch_px)
    candidates.append({
        "source": "total",
        "height_px": info["total_height_px"],
        "count": total_count,
        "weight": VOTE_WEIGHTS["total"],
    })

    for i in range(SEGMENTS):
        source = f"seg{i + 1}"
        height_px = info[f"{source}_height_px"]
        candidates.append({
            "source": source,
            "height_px": height_px,
            "count": count_from_height(height_px, pitch_px),
            "weight": VOTE_WEIGHTS[source],
        })

    return candidates


def x_runs(xs):
    if not xs:
        return []

    values = sorted(set(int(x) for x in xs))
    runs = []
    start = values[0]
    prev = values[0]

    for x in values[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = x
            prev = x

    runs.append((start, prev))
    return runs


def draw_text_panel(
    vis,
    lines,
    origin=(24, 32),
    font_scale=None,
    line_gap=None,
    color=(0, 255, 255),
    bg_color=(0, 0, 0),
    thickness=None,
):
    if not lines:
        return

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    if font_scale is None:
        font_scale = float(np.clip(vis.shape[1] / 2600.0 * 0.82, 0.72, 1.15))
    if thickness is None:
        thickness = 3 if font_scale >= 0.78 else 2
    if line_gap is None:
        line_gap = int(round(34 * font_scale))

    pad_x = int(round(18 * font_scale))
    pad_top = int(round(14 * font_scale))
    baseline = int(round(28 * font_scale))
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_w = max(width for width, _ in sizes) + pad_x * 2
    panel_h = line_gap * len(lines) + pad_top * 2

    x2 = min(vis.shape[1] - 1, x + panel_w)
    y2 = min(vis.shape[0] - 1, y + panel_h)
    roi = vis[y:y2, x:x2]
    if roi.size:
        overlay = roi.copy()
        overlay[:] = bg_color
        vis[y:y2, x:x2] = cv2.addWeighted(roi, 0.35, overlay, 0.65, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (x + pad_x, y + pad_top + baseline + i * line_gap),
            font,
            font_scale,
            color,
            thickness,
        )


def draw_segment_heights(img, info, candidates=None, method_count=None, pitch_px=0, hand_mask=None):
    vis = img.copy()
    h, _ = vis.shape[:2]
    candidate_by_source = {}
    if candidates is not None:
        candidate_by_source = {item["source"]: item for item in candidates}

    if hand_mask is not None:
        layer = np.zeros_like(vis)
        layer[:] = (0, 128, 255)
        m = hand_mask > 0
        if np.any(m):
            blended = cv2.addWeighted(vis, 0.65, layer, 0.35, 0)
            vis[m] = blended[m]

    if info["bright_x1"] != "":
        cv2.rectangle(
            vis,
            (info["bright_x1"], info["bright_y1"]),
            (info["bright_x2"], info["bright_y2"]),
            (255, 0, 255),
            2,
        )

    for x1, x2 in x_runs(info.get("excluded_hand_xs", [])):
        cv2.rectangle(vis, (x1, 0), (x2, h - 1), (0, 0, 255), 1)

    for x1, x2 in x_runs(info.get("valid_xs", [])):
        cv2.line(vis, (x1, h - 8), (x2, h - 8), (255, 255, 0), 2)

    panel_lines = [
        (
            f"total={info['total_height_px']}px "
            f"count={candidate_by_source.get('total', {}).get('count', '')} "
            f"method={method_count} pitch={pitch_px}"
        ),
        (
            f"edge_angle={info.get('edge_angle_deg', 0):.2f} "
            f"valid_cols={info.get('valid_cols', 0)}/{info.get('raw_valid_cols', 0)} "
            f"height_stat={info.get('height_stat', '')}"
        ),
        (
            f"excluded_hand_cols={info.get('excluded_hand_cols', 0)} "
            f"outliers={info.get('outlier_cols', 0)} "
            f"removed_hand_components={info.get('raw_hand_removed_components', 0)}"
        ),
    ]

    for i in range(SEGMENTS):
        x1 = info[f"seg{i + 1}_x1"]
        x2 = info[f"seg{i + 1}_x2"]
        top = info[f"seg{i + 1}_top_y"]
        bottom = info[f"seg{i + 1}_bottom_y"]
        height = info[f"seg{i + 1}_height_px"]
        count = candidate_by_source.get(f"seg{i + 1}", {}).get("count", "")
        cols = info.get(f"seg{i + 1}_valid_cols", 0)

        panel_lines.append(f"seg{i + 1}: h={height}px count={count} cols={cols}")

        if x1 == "":
            continue

        cv2.rectangle(vis, (x1, 0), (x2, h - 1), (255, 200, 0), 1)

        if height > 0:
            cv2.line(vis, (x1, top), (x2, top), (0, 255, 0), 2)
            cv2.line(vis, (x1, bottom), (x2, bottom), (0, 0, 255), 2)

    draw_text_panel(vis, panel_lines, origin=(24, 24))
    return vis


def weighted_median(values):
    expanded = []
    for count, weight in values:
        if count <= 0:
            continue
        repeat = max(1, int(round(weight * 4)))
        expanded.extend([count] * repeat)

    if not expanded:
        return 0.0

    return float(np.median(np.array(expanded, dtype=np.float32)))


def vote_plate_count(method_results):
    scores = {}
    weighted_values = []
    total_support = {}

    for result in method_results:
        for candidate in result["candidates"]:
            count = candidate["count"]
            if count <= 0:
                continue

            weight = candidate["weight"]
            scores[count] = scores.get(count, 0.0) + weight
            weighted_values.append((count, weight))

            if candidate["source"] == "total":
                total_support[count] = total_support.get(count, 0) + 1

    if not scores:
        return {
            "final_count": 0,
            "scores": {},
            "reason": "no_valid_candidates",
        }

    best_score = max(scores.values())
    winners = [count for count, score in scores.items() if abs(score - best_score) < 1e-6]

    if len(winners) == 1:
        final_count = winners[0]
        reason = "highest_weight"
    else:
        median_value = weighted_median(weighted_values)
        best_distance = min(abs(count - median_value) for count in winners)
        winners = [
            count
            for count in winners
            if abs(abs(count - median_value) - best_distance) < 1e-6
        ]

        if len(winners) == 1:
            final_count = winners[0]
            reason = "weighted_median_tiebreak"
        else:
            best_total_support = max(total_support.get(count, 0) for count in winners)
            winners = [
                count
                for count in winners
                if total_support.get(count, 0) == best_total_support
            ]
            final_count = sorted(winners)[0]
            reason = "total_support_tiebreak"

    return {
        "final_count": int(final_count),
        "scores": dict(sorted(scores.items())),
        "reason": reason,
    }


def configure_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    year, week, _ = datetime.now().isocalendar()
    log_path = LOG_DIR / f"{year}-W{week:02d}.log"
    logger.add(
        str(log_path),
        encoding="utf-8",
        rotation="00:00",
        retention="12 weeks",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    return log_path


def now_week_dir(base_dir: Path) -> Path:
    year, week, _ = datetime.now().isocalendar()
    path = base_dir / f"{year}-W{week:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def analyze_sheet_image(img_bgr, pitch_px=None):
    raw_hmask, hmask, hand_stats = hand_masks(img_bgr)
    non_hand = cv2.bitwise_not(hmask)

    bright = bright_otsu_mask(img_bgr)
    bright_non_hand = cv2.bitwise_and(bright, non_hand)
    plate_mask = keep_plate_like_components(bright_non_hand, img_bgr.shape)
    measure_mask = post_process(plate_mask)

    info = calc_segment_heights(
        measure_mask,
        hmask if USE_HAND_COLUMN_EXCLUSION else None,
        hand_stats,
    )
    detected_height_px = safe_float(info.get("total_height_px"))

    candidates = []
    vote_result = {"final_count": 0, "scores": {}, "reason": "pitch_px_not_available"}
    final_count = 0
    draw_pitch = safe_float(pitch_px)

    if draw_pitch > 0:
        candidates = height_count_candidates(info, draw_pitch)
        vote_result = vote_plate_count([{"method": "hsv_otsu", "candidates": candidates}])
        final_count = int(vote_result["final_count"])

    result_img = draw_segment_heights(
        img_bgr,
        info,
        candidates,
        final_count,
        draw_pitch if draw_pitch > 0 else 0,
        hmask,
    )

    return {
        "method": "hsv_otsu",
        "detected_height_px": detected_height_px,
        "final_count": final_count,
        "pitch_px": draw_pitch,
        "vote_result": vote_result,
        "info": info,
        "result_img": result_img,
    }


def detect_sheet_presence(img_bgr):
    """快速判断画面中是否已经出现压板。只用于自动触发，不用于最终计数。"""
    if img_bgr is None:
        return False, {}

    scale = float(AUTO_PRESENCE_SCALE)
    if scale < 0.999:
        h, w = img_bgr.shape[:2]
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        work_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        work_img = img_bgr

    bright = bright_otsu_mask(work_img)
    plate_mask = keep_plate_like_components(bright, work_img.shape)
    bbox = calc_bright_bbox(plate_mask)
    area = int(np.count_nonzero(plate_mask > 0))
    image_area = int(plate_mask.shape[0] * plate_mask.shape[1])
    area_ratio = area / max(1, image_area)

    has_sheet = bbox is not None and area_ratio >= 0.002
    info = {
        "bbox": bbox,
        "area": area,
        "area_ratio": area_ratio,
        "scale": scale,
    }
    return has_sheet, info


class SheetCountMainWindow(QWidget):
    MODE_CAMERA = 0
    MODE_PITCH = 1
    MODE_DETECT = 2
    REQUIRED_PITCH_SAMPLES = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("压板计数工具")
        self.setGeometry(180, 120, 1280, 900)

        self.cam = None
        self.image_bgr = None
        self.result_img = None
        self.result_qimg = None
        self.result_pixmap = None
        self.pitch_px = None
        self.samples = []
        self.loaded_param_path = None
        self.camera_params = self.empty_camera_params()
        self.log_path = configure_logger()

        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(AUTO_DETECT_INTERVAL_MS)
        self.auto_timer.timeout.connect(self.auto_detect_tick)
        self.auto_state = AUTO_STATE_IDLE
        self.auto_present_frames = 0
        self.auto_empty_frames = 0
        self.auto_busy = False

        self.init_ui()
        self.log_message(f"日志文件: {self.log_path}")

    def empty_camera_params(self):
        return {
            "ip": "",
            "exposure_time": None,
            "width": None,
            "height": None,
            "offset_x": None,
            "offset_y": None,
        }

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)

        self.home_widget = QWidget()
        home_layout = QVBoxLayout(self.home_widget)
        home_layout.setContentsMargins(120, 90, 120, 90)
        home_layout.setSpacing(28)

        home_title = QLabel("压板计数工具")
        home_title.setAlignment(Qt.AlignCenter)
        home_title.setStyleSheet("font-size: 24pt; font-weight: 700; color: #222;")
        home_layout.addWidget(home_title)

        home_subtitle = QLabel("请选择工作模式")
        home_subtitle.setAlignment(Qt.AlignCenter)
        home_subtitle.setStyleSheet("font-size: 12pt; color: #666;")
        home_layout.addWidget(home_subtitle)
        home_layout.addSpacing(18)

        entry_defs = [
            (self.MODE_CAMERA, "相机参数调试"),
            (self.MODE_PITCH, "压板参数确定"),
            (self.MODE_DETECT, "检测流程"),
        ]
        for mode_id, text in entry_defs:
            btn = QPushButton(text)
            btn.setMinimumHeight(76)
            btn.setStyleSheet(
                """
                QPushButton {
                    font-size: 17pt;
                    font-weight: 600;
                    background-color: #0B65C2;
                    border-radius: 8px;
                    padding: 12px;
                }
                QPushButton:hover {
                    background-color: #084C91;
                }
                """
            )
            btn.clicked.connect(lambda checked=False, m=mode_id: self.enter_mode(m))
            home_layout.addWidget(btn)
        home_layout.addStretch(1)
        root_layout.addWidget(self.home_widget)

        self.work_widget = QWidget()
        work_root_layout = QVBoxLayout(self.work_widget)
        work_root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.work_widget)

        self.current_mode_id = self.MODE_CAMERA
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        self.back_home_btn = QPushButton("返回主界面")
        self.back_home_btn.setMinimumHeight(54)
        self.back_home_btn.clicked.connect(self.show_home)
        self.back_prev_btn = QPushButton("返回上一级")
        self.back_prev_btn.setMinimumHeight(54)
        self.back_prev_btn.clicked.connect(self.back_to_previous_step)
        self.exit_btn = QPushButton("退出")
        self.exit_btn.setMinimumHeight(54)
        self.exit_btn.clicked.connect(self.exit_app)
        header_layout.addWidget(self.back_home_btn)
        header_layout.addWidget(self.back_prev_btn)
        header_layout.addWidget(self.exit_btn)
        header_layout.addStretch(1)
        work_root_layout.addLayout(header_layout)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)
        work_root_layout.addLayout(main_layout, 1)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(18)
        main_layout.addLayout(left_layout, 2)

        self.mode_title_label = QLabel("相机参数调试")
        self.mode_title_label.setStyleSheet("font-size: 14pt; font-weight: 600;")
        left_layout.addWidget(self.mode_title_label)

        self.camera_widget = CameraDebugWidget(self)
        self.pitch_widget = PitchCalibrationWidget(self)
        self.detection_widget = DetectionWidget(self)
        left_layout.addWidget(self.camera_widget)
        left_layout.addWidget(self.pitch_widget)
        left_layout.addWidget(self.detection_widget)

        self.cam_ip_input = self.camera_widget.cam_ip_input
        self.cam_exposure_input = self.camera_widget.cam_exposure_input
        self.cam_width_input = self.camera_widget.cam_width_input
        self.cam_height_input = self.camera_widget.cam_height_input
        self.cam_offset_x_input = self.camera_widget.cam_offset_x_input
        self.cam_offset_y_input = self.camera_widget.cam_offset_y_input
        self.cam_btn = self.camera_widget.cam_btn
        self.apply_camera_btn = self.camera_widget.apply_camera_btn
        self.preview_btn = self.camera_widget.preview_btn
        self.camera_status_label = self.camera_widget.camera_status_label
        self.correct_count_input = self.pitch_widget.correct_count_input
        self.pitch_label = self.pitch_widget.pitch_label
        self.pitch_samples_label = self.pitch_widget.samples_label
        self.param_label = self.detection_widget.param_label
        self.auto_btn = self.detection_widget.auto_btn
        self.auto_status_label = self.detection_widget.auto_status_label

        left_layout.addStretch(1)

        self.info_panel = QWidget()
        self.info_panel.setStyleSheet(
            """
            QWidget {
                background-color: #F3F4F6;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                color: #111827;
                font-size: 13pt;
                font-weight: 600;
                padding: 8px 10px 0 10px;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                color: #111827;
                font-size: 11pt;
                padding: 6px;
            }
            """
        )
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)

        self.image_path_label = QLabel("图片: 未采集")
        self.image_path_label.setWordWrap(True)
        info_layout.addWidget(self.image_path_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFixedHeight(190)
        info_layout.addWidget(self.output_text)
        left_layout.addWidget(self.info_panel)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(right_layout, 6)

        self.camera_summary_label = QLabel("相机调试状态\n\n等待采集预览图")
        self.camera_summary_label.setWordWrap(True)
        self.camera_summary_label.setMinimumHeight(150)
        self.camera_summary_label.setStyleSheet(
            """
            QLabel {
                background-color: #F3F4F6;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                color: #111827;
                font-size: 18pt;
                font-weight: 700;
                padding: 20px;
            }
            """
        )
        self.camera_summary_label.setVisible(False)

        self.result_table = QTableWidget(self)
        self.result_table.setColumnCount(8)
        self.result_table.setHorizontalHeaderLabels(
            ["流程", "图片名称", "正确数量", "厚度px", "样本pitch", "当前pitch", "计数结果", "相机参数"]
        )
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setVisible(False)

        self.image_scene = QGraphicsScene(self)
        self.image_view = QGraphicsView(self.image_scene)
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("background: #222; border: none;")
        self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.image_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        right_layout.addWidget(self.image_view, 1)

        self.on_mode_changed(self.MODE_CAMERA)
        self.show_home()

    def current_mode(self):
        return self.current_mode_id

    def on_mode_changed(self, mode_id):
        if mode_id != self.MODE_DETECT and hasattr(self, "auto_timer"):
            self.stop_auto_detect(silent=True)
        self.current_mode_id = mode_id
        titles = {
            self.MODE_CAMERA: "相机参数调试",
            self.MODE_PITCH: "压板参数确定",
            self.MODE_DETECT: "检测流程",
        }
        self.mode_title_label.setText(titles.get(mode_id, "相机参数调试"))
        self.camera_widget.setVisible(mode_id == self.MODE_CAMERA)
        self.pitch_widget.setVisible(mode_id == self.MODE_PITCH)
        self.detection_widget.setVisible(mode_id == self.MODE_DETECT)
        self.back_prev_btn.setVisible(mode_id in (self.MODE_PITCH, self.MODE_DETECT))
        self.result_table.setVisible(False)
        self.camera_summary_label.setVisible(False)
        self.update_pitch_status()

    def set_mode(self, mode_id):
        self.on_mode_changed(mode_id)

    def enter_mode(self, mode_id):
        self.set_mode(mode_id)
        self.home_widget.setVisible(False)
        self.work_widget.setVisible(True)

    def show_home(self):
        self.stop_auto_detect(silent=True)
        self.work_widget.setVisible(False)
        self.home_widget.setVisible(True)

    def back_to_camera_debug(self):
        self.enter_mode(self.MODE_CAMERA)

    def back_to_pitch_calibration(self):
        self.enter_mode(self.MODE_PITCH)

    def back_to_previous_step(self):
        if self.current_mode_id == self.MODE_DETECT:
            self.back_to_pitch_calibration()
        elif self.current_mode_id == self.MODE_PITCH:
            self.back_to_camera_debug()

    def exit_app(self):
        self.close()

    def log_message(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.append(f"[{timestamp}] {message}")
        getattr(logger, level, logger.info)(message)

    def show_error(self, title, message):
        self.log_message(message, "error")
        QMessageBox.critical(self, title, message)

    def parse_optional_int(self, widget, name, min_value=0):
        text = widget.text().strip()
        if text == "":
            return None
        try:
            value = int(float(text))
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if value < min_value:
            raise ValueError(f"{name} 不能小于 {min_value}")
        return value

    def read_camera_params(self, require_exposure=True):
        ip = self.cam_ip_input.text().strip()
        exposure_text = self.cam_exposure_input.text().strip()
        if require_exposure and not exposure_text:
            raise ValueError("ExposureTime 不能为空")
        exposure_time = None
        if exposure_text:
            try:
                exposure_time = float(exposure_text)
            except ValueError as exc:
                raise ValueError("ExposureTime 必须是数字") from exc
            if exposure_time <= 0:
                raise ValueError("ExposureTime 必须大于 0")

        return {
            "ip": ip,
            "exposure_time": exposure_time,
            "width": self.parse_optional_int(self.cam_width_input, "Width", 1),
            "height": self.parse_optional_int(self.cam_height_input, "Height", 1),
            "offset_x": self.parse_optional_int(self.cam_offset_x_input, "OffsetX", 0),
            "offset_y": self.parse_optional_int(self.cam_offset_y_input, "OffsetY", 0),
        }

    def fill_camera_params(self, params):
        params = params or {}
        self.cam_ip_input.setText(str(params.get("ip") or ""))
        self.cam_exposure_input.setText("" if params.get("exposure_time") in (None, "") else str(params.get("exposure_time")))
        self.cam_width_input.setText("" if params.get("width") in (None, "") else str(params.get("width")))
        self.cam_height_input.setText("" if params.get("height") in (None, "") else str(params.get("height")))
        self.cam_offset_x_input.setText("" if params.get("offset_x") in (None, "") else str(params.get("offset_x")))
        self.cam_offset_y_input.setText("" if params.get("offset_y") in (None, "") else str(params.get("offset_y")))

    def camera_params_text(self, params=None):
        params = params or self.camera_params
        return (
            f"ip={params.get('ip') or ''}, exposure={params.get('exposure_time') or ''}, "
            f"width={params.get('width') or ''}, height={params.get('height') or ''}, "
            f"offset_x={params.get('offset_x') or ''}, offset_y={params.get('offset_y') or ''}"
        )

    def update_camera_status(self):
        self.camera_status_label.setText(f"相机参数: {self.camera_params_text()}")
        self.update_camera_summary()

    def refresh_camera_status(self):
        self.update_camera_status()

    def update_pitch_status(self):
        sample_count = len(self.samples)
        if self.pitch_px is not None and self.pitch_px > 0:
            self.pitch_label.setText(f"样本 {sample_count} 张  |  pitch_px {self.pitch_px:.3f}")
        else:
            need = max(0, self.REQUIRED_PITCH_SAMPLES - sample_count)
            self.pitch_label.setText(f"样本 {sample_count} 张  |  还需 {need} 张")
        self.update_pitch_samples_text()

    def refresh_pitch_status(self):
        self.update_pitch_status()

    def update_camera_summary(self, image_name=None):
        status = "已连接" if self.cam is not None else "未连接"
        image_text = image_name or "未采集"
        self.camera_summary_label.setText(
            "相机调试状态\n\n"
            f"相机: {status}\n"
            f"预览图: {image_text}\n"
            f"参数: {self.camera_params_text()}"
        )

    def update_pitch_samples_text(self):
        if not self.samples:
            self.pitch_samples_label.setText("每张参数结果: 暂无")
            return

        lines = []
        for index, sample in enumerate(self.samples, start=1):
            lines.append(
                f"第 {index} 张  |  数量 {sample.get('correct_count', '')}  |  "
                f"厚度 {safe_float(sample.get('detected_height_px')):.1f}px  |  "
                f"pitch {safe_float(sample.get('sample_pitch_px')):.3f}"
            )
        if self.pitch_px is not None and self.pitch_px > 0:
            lines.append(f"\n最终参数  |  pitch_px {self.pitch_px:.3f}")
        else:
            need = max(0, self.REQUIRED_PITCH_SAMPLES - len(self.samples))
            lines.append(f"\n最终参数  |  还需 {need} 张有效样本")
        self.pitch_samples_label.setText("\n".join(lines))

    def refresh_param_status(self):
        if self.loaded_param_path is None:
            self.param_label.setText("参数: 未导入")
        else:
            self.param_label.setText(f"参数: 已导入 {self.loaded_param_path.name}")

    def connect_or_disconnect_camera(self, checked):
        if checked:
            try:
                params = self.read_camera_params(require_exposure=False)
                if not params["ip"]:
                    raise ValueError("请先输入相机 IP")
                if self.cam is None:
                    self.cam = CameraWrapper()
                self.cam.enable_device_by_ip(params["ip"])
                self.camera_params.update(params)
                self.cam_btn.setText("相机已连接")
                self.cam_btn.setStyleSheet("background-color: lightgreen")
                self.update_camera_status()
                self.log_message(f"相机已连接: {params['ip']}")
            except SystemExit as exc:
                self.cam_btn.setChecked(False)
                self.cam_btn.setText("连接相机")
                self.cam_btn.setStyleSheet("")
                self.cam = None
                self.show_error("相机连接失败", f"未找到相机或相机初始化失败: {exc}")
            except Exception as exc:
                self.cam_btn.setChecked(False)
                self.cam_btn.setText("连接相机")
                self.cam_btn.setStyleSheet("")
                self.cam = None
                self.show_error("相机连接失败", str(exc))
        else:
            self.stop_auto_detect(silent=True)
            if self.cam is not None:
                try:
                    self.cam.close_device()
                    self.log_message("相机已断开")
                except Exception as exc:
                    self.log_message(f"相机断开异常: {exc}", "error")
                finally:
                    self.cam = None
            self.cam_btn.setText("连接相机")
            self.cam_btn.setStyleSheet("background-color: lightgray")

    def on_cam_toggled(self, checked):
        self.connect_or_disconnect_camera(checked)

    def confirm_camera_debug(self):
        try:
            self.apply_camera_params(require_connected=True)
            self.log_message(f"相机参数确认完成: {self.camera_params_text()}")
            self.enter_mode(self.MODE_PITCH)
        except Exception as exc:
            self.show_error("相机参数确认失败", str(exc))

    def apply_camera_params_clicked(self):
        try:
            self.apply_camera_params(require_connected=True)
            self.log_message(f"相机参数已应用: {self.camera_params_text()}")
        except Exception as exc:
            self.show_error("应用相机参数失败", str(exc))

    def export_camera_params_json(self):
        try:
            camera_params = self.read_camera_params(require_exposure=True)
        except Exception as exc:
            self.show_error("导出相机参数失败", f"相机参数无效: {exc}")
            return

        default_name = f"camera_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出相机参数 JSON",
            str(ROOT / "json" / default_name),
            "JSON (*.json)",
        )
        if not path:
            return

        data = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "camera_params": camera_params,
        }

        try:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.camera_params = camera_params
            self.update_camera_status()
            self.log_message(f"相机参数 JSON 已导出: {out_path}, camera_params={camera_params}")
        except Exception as exc:
            self.show_error("导出相机参数失败", str(exc))

    def import_camera_params_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入相机参数 JSON",
            str(ROOT / "json"),
            "JSON (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            camera_params = data.get("camera_params", data)
            if not isinstance(camera_params, dict):
                raise ValueError("JSON 中缺少 camera_params")
            if not camera_params.get("ip") or safe_float(camera_params.get("exposure_time")) <= 0:
                raise ValueError("camera_params 缺少有效 ip 或 exposure_time")

            self.fill_camera_params(camera_params)
            self.camera_params = self.read_camera_params(require_exposure=True)
            self.update_camera_status()
            self.log_message(f"相机参数 JSON 已导入: {path}, camera_params={self.camera_params}")
        except Exception as exc:
            self.show_error("导入相机参数失败", str(exc))

    def apply_camera_params(self, require_connected=True):
        params = self.read_camera_params(require_exposure=True)
        self.camera_params = params
        self.update_camera_status()

        if require_connected and self.cam is None:
            raise RuntimeError("请先连接相机")
        if self.cam is None:
            return params

        self.cam.set_exposure_time(float(params["exposure_time"]))
        if params["width"] is not None:
            self.cam.set_width(int(params["width"]))
        if params["height"] is not None:
            self.cam.set_height(int(params["height"]))
        if params["offset_x"] is not None:
            self.cam.set_offset_x(int(params["offset_x"]))
        if params["offset_y"] is not None:
            self.cam.set_offset_y(int(params["offset_y"]))
        return params

    def ensure_camera_ready(self):
        if self.cam is None:
            raise RuntimeError("请先连接相机")
        return self.apply_camera_params(require_connected=True)

    def capture_preview(self):
        try:
            params = self.ensure_camera_ready()
            img_bgr, image_path = self.grab_image(set_exposure=False)
            self.result_img = img_bgr
            self.result_qimg = self.show_image(img_bgr)
            self.append_result_row("相机调试", image_path.name, "", "", "", self.pitch_px, "", params)
            self.update_camera_summary(image_path.name)
            self.log_message(f"预览图已采集: image={image_path}, camera_params={params}")
        except Exception as exc:
            self.show_error("采集预览失败", str(exc))

    def grab_image(self, set_exposure=True):
        if self.cam is None:
            raise RuntimeError("请先连接相机")
        if set_exposure:
            exposure_time = self.camera_params.get("exposure_time")
            if exposure_time is None or exposure_time <= 0:
                raise RuntimeError("请先设置有效 ExposureTime")
            self.cam.set_exposure_time(float(exposure_time))

        time.sleep(2.0)
        frame = self.cam.get_image()
        if frame is None:
            raise RuntimeError("相机未返回图像")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"图像格式异常: shape={frame.shape}")

        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = self.save_capture_image(img_bgr)
        self.image_bgr = img_bgr
        self.image_path_label.setText(f"图片: {image_path}")
        self.log_message(f"图片已采集: {image_path}")
        return img_bgr, image_path

    def grab_frame_fast(self):
        """自动检测轮询用：只取当前帧，不等待、不保存。"""
        if self.cam is None:
            raise RuntimeError("请先连接相机")

        frame = self.cam.get_image()
        if frame is None:
            raise RuntimeError("相机未返回图像")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"图像格式异常: shape={frame.shape}")

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def save_capture_image(self, img_bgr):
        folder = now_week_dir(IMAGE_DIR)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = folder / filename
        ok = cv2.imwrite(str(path), img_bgr)
        if not ok:
            raise RuntimeError(f"图片保存失败: {path}")
        return path

    def capture_pitch_sample(self, correct_count_text=None):
        try:
            params = self.ensure_camera_ready()
            if correct_count_text is None:
                correct_count_text = self.correct_count_input.text()
            correct_count = safe_int(str(correct_count_text).strip())
            if correct_count <= 0:
                raise ValueError("请先输入大于 0 的正确数量")

            img_bgr, image_path = self.grab_image(set_exposure=False)
            analysis = analyze_sheet_image(img_bgr)
            detected_height_px = analysis["detected_height_px"]
            if detected_height_px <= 0:
                raise RuntimeError("算法未检测到有效厚度")

            sample_pitch_px = detected_height_px / correct_count
            sample = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "correct_count": int(correct_count),
                "detected_height_px": float(detected_height_px),
                "sample_pitch_px": float(sample_pitch_px),
            }
            self.samples.append(sample)

            if len(self.samples) >= self.REQUIRED_PITCH_SAMPLES:
                self.pitch_px = float(np.mean([item["sample_pitch_px"] for item in self.samples]))
                display_analysis = analyze_sheet_image(img_bgr, self.pitch_px)
            else:
                display_analysis = analysis

            self.result_img = display_analysis["result_img"]
            self.result_qimg = self.show_image(self.result_img)
            self.update_pitch_status()
            self.append_result_row(
                "压板参数确定",
                image_path.name,
                correct_count,
                detected_height_px,
                sample_pitch_px,
                self.pitch_px,
                display_analysis.get("final_count", ""),
                params,
            )
            self.log_message(
                "压板参数样本完成: "
                f"image={image_path}, correct_count={correct_count}, "
                f"detected_height_px={detected_height_px:.3f}, "
                f"sample_pitch_px={sample_pitch_px:.6f}, pitch_px={self.pitch_px}"
            )
        except Exception as exc:
            self.show_error("压板参数确定失败", str(exc))

    def confirm_pitch_calibration(self):
        if self.pitch_px is None or self.pitch_px <= 0 or len(self.samples) < self.REQUIRED_PITCH_SAMPLES:
            self.show_error("压板参数确认失败", "至少需要 2 张有效压板参数样本后才能进入检测流程")
            return
        self.log_message(f"压板参数确认完成: pitch_px={self.pitch_px:.6f}")
        self.enter_mode(self.MODE_DETECT)

    def capture_count_result(self):
        try:
            if self.pitch_px is None or self.pitch_px <= 0:
                raise RuntimeError("请先导入或确定有效 pitch_px")
            if self.auto_timer.isActive():
                self.stop_auto_detect(silent=True)
            params = self.ensure_camera_ready()
            img_bgr, image_path = self.grab_image(set_exposure=False)
            self.run_count_analysis(img_bgr, image_path, params, flow="检测")
        except Exception as exc:
            self.detection_widget.set_count_result(None)
            self.show_error("检测失败", str(exc))

    def run_count_analysis(self, img_bgr, image_path=None, params=None, flow="检测"):
        if self.pitch_px is None or self.pitch_px <= 0:
            raise RuntimeError("请先导入或确定有效 pitch_px")
        if params is None:
            params = self.camera_params
        if image_path is None:
            image_path = self.save_capture_image(img_bgr)
            self.image_bgr = img_bgr
            self.image_path_label.setText(f"图片: {image_path}")
            self.log_message(f"图片已保存: {image_path}")

        analysis = analyze_sheet_image(img_bgr, self.pitch_px)
        detected_height_px = analysis["detected_height_px"]
        final_count = analysis["final_count"]
        if detected_height_px <= 0:
            raise RuntimeError("算法未检测到有效厚度")

        self.result_img = analysis["result_img"]
        self.result_qimg = self.show_image(self.result_img)
        self.detection_widget.set_count_result(final_count)
        self.append_result_row(
            flow,
            image_path.name,
            "",
            detected_height_px,
            "",
            self.pitch_px,
            final_count,
            params,
        )
        self.log_message(
            f"{flow}完成: "
            f"image={image_path}, image_name={image_path.name}, "
            f"detected_height_px={detected_height_px:.3f}, "
            f"pitch_px={self.pitch_px:.6f}, final_count={final_count}, "
            f"camera_params={params}, vote={analysis['vote_result']}"
        )
        return analysis

    def toggle_auto_detect(self, checked):
        if checked:
            self.start_auto_detect()
        else:
            self.stop_auto_detect()

    def start_auto_detect(self):
        try:
            if self.pitch_px is None or self.pitch_px <= 0:
                raise RuntimeError("请先导入或确定有效 pitch_px")
            self.ensure_camera_ready()
            self.auto_state = AUTO_STATE_IDLE
            self.auto_present_frames = 0
            self.auto_empty_frames = 0
            self.auto_busy = False
            self.auto_timer.start(AUTO_DETECT_INTERVAL_MS)
            self.detection_widget.set_auto_running(True)
            self.detection_widget.set_auto_status("自动检测: 等待压板进入")
            self.log_message(
                "自动检测已开启: "
                f"interval={AUTO_DETECT_INTERVAL_MS}ms, "
                f"present_frames={AUTO_PRESENT_FRAMES}, empty_frames={AUTO_EMPTY_FRAMES}"
            )
        except Exception as exc:
            self.detection_widget.set_auto_running(False)
            self.detection_widget.set_auto_status("自动检测: 启动失败")
            self.show_error("自动检测启动失败", str(exc))

    def stop_auto_detect(self, silent=False):
        was_active = hasattr(self, "auto_timer") and self.auto_timer.isActive()
        if hasattr(self, "auto_timer"):
            self.auto_timer.stop()
        self.auto_state = AUTO_STATE_IDLE
        self.auto_present_frames = 0
        self.auto_empty_frames = 0
        self.auto_busy = False
        if hasattr(self, "detection_widget"):
            self.detection_widget.set_auto_running(False)
            self.detection_widget.set_auto_status("自动检测: 未开启")
        if was_active and not silent:
            self.log_message("自动检测已关闭")

    def auto_detect_tick(self):
        if self.auto_busy:
            return
        if self.current_mode_id != self.MODE_DETECT:
            self.stop_auto_detect(silent=True)
            return

        self.auto_busy = True
        try:
            img_bgr = self.grab_frame_fast()
            has_sheet, presence_info = detect_sheet_presence(img_bgr)

            if self.auto_state == AUTO_STATE_IDLE:
                if has_sheet:
                    self.auto_present_frames += 1
                    self.auto_empty_frames = 0
                    self.detection_widget.set_auto_status(
                        f"自动检测: 检测到压板 {self.auto_present_frames}/{AUTO_PRESENT_FRAMES}"
                    )
                    if self.auto_present_frames >= AUTO_PRESENT_FRAMES:
                        params = self.camera_params.copy()
                        self.run_count_analysis(img_bgr, None, params, flow="自动检测")
                        self.auto_state = AUTO_STATE_COUNTED
                        self.auto_present_frames = 0
                        self.auto_empty_frames = 0
                        self.detection_widget.set_auto_status("自动检测: 已输出计数，等待压板离开")
                else:
                    self.auto_present_frames = 0
                    self.detection_widget.set_auto_status("自动检测: 等待压板进入")

            elif self.auto_state == AUTO_STATE_COUNTED:
                if has_sheet:
                    self.auto_empty_frames = 0
                    self.detection_widget.set_auto_status("自动检测: 已计数，等待压板离开")
                else:
                    self.auto_empty_frames += 1
                    self.detection_widget.set_auto_status(
                        f"自动检测: 离开确认 {self.auto_empty_frames}/{AUTO_EMPTY_FRAMES}"
                    )
                    if self.auto_empty_frames >= AUTO_EMPTY_FRAMES:
                        self.auto_state = AUTO_STATE_IDLE
                        self.auto_present_frames = 0
                        self.auto_empty_frames = 0
                        self.detection_widget.set_auto_status("自动检测: 等待下一次压板进入")

            logger.debug(f"auto_presence={has_sheet}, info={presence_info}, state={self.auto_state}")
        except Exception as exc:
            self.stop_auto_detect(silent=True)
            self.detection_widget.set_count_result(None)
            self.show_error("自动检测异常", str(exc))
        finally:
            self.auto_busy = False

    def append_result_row(
        self,
        flow,
        image_name,
        correct_count,
        detected_height_px,
        sample_pitch_px,
        pitch_px,
        final_count,
        camera_params=None,
    ):
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        values = [
            flow,
            image_name,
            str(correct_count),
            "" if detected_height_px == "" else f"{safe_float(detected_height_px):.3f}",
            "" if sample_pitch_px == "" else f"{safe_float(sample_pitch_px):.6f}",
            "" if pitch_px is None else f"{safe_float(pitch_px):.6f}",
            str(final_count),
            self.camera_params_text(camera_params) if camera_params else "",
        ]
        for col, value in enumerate(values):
            self.result_table.setItem(row, col, QTableWidgetItem(value))
        self.result_table.resizeColumnsToContents()

    def export_params_json(self):
        if self.pitch_px is None or self.pitch_px <= 0 or len(self.samples) < self.REQUIRED_PITCH_SAMPLES:
            self.show_error("导出失败", "至少需要 2 张有效 pitch 标定样本后才能导出参数 JSON")
            return

        try:
            camera_params = self.read_camera_params(require_exposure=True)
        except Exception as exc:
            self.show_error("导出失败", f"相机参数无效: {exc}")
            return

        default_name = f"sheet_count_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出参数 JSON",
            str(ROOT / "json" / default_name),
            "JSON (*.json)",
        )
        if not path:
            return

        data = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "camera_params": camera_params,
            "pitch_px": float(self.pitch_px),
            "pitch_rule": "average",
            "samples": self.samples,
        }

        try:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.log_message(f"参数 JSON 已导出: {out_path}")
        except Exception as exc:
            self.show_error("导出失败", str(exc))

    def import_params_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入参数 JSON",
            str(ROOT / "json"),
            "JSON (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pitch_px = safe_float(data.get("pitch_px"))
            if pitch_px <= 0:
                raise ValueError("参数 JSON 缺少有效 pitch_px")

            samples = data.get("samples", [])
            camera_params = data.get("camera_params")
            if camera_params:
                if not camera_params.get("ip") or safe_float(camera_params.get("exposure_time")) <= 0:
                    raise ValueError("参数 JSON 的 camera_params 缺少有效 ip 或 exposure_time")
                if len(samples) < self.REQUIRED_PITCH_SAMPLES:
                    raise ValueError("参数 JSON 的 samples 少于 2 条")
                self.fill_camera_params(camera_params)
                self.camera_params = self.read_camera_params(require_exposure=True)
                self.update_camera_status()
            else:
                self.log_message("导入旧版 JSON：未包含 camera_params，请手动输入并应用相机参数", "warning")

            self.pitch_px = pitch_px
            self.samples = samples if isinstance(samples, list) else []
            self.loaded_param_path = Path(path)
            self.refresh_param_status()
            self.update_pitch_status()
            self.log_message(f"参数 JSON 已导入: {path}, pitch_px={self.pitch_px:.6f}")
        except Exception as exc:
            self.show_error("导入失败", str(exc))

    def show_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.result_pixmap = QPixmap.fromImage(qimg)
        self.image_scene.clear()
        self.image_scene.addPixmap(self.result_pixmap)
        self.image_scene.setSceneRect(self.result_pixmap.rect())
        self.fit_image_to_view()
        return qimg

    def fit_image_to_view(self):
        if self.result_pixmap is None or self.result_pixmap.isNull():
            return
        self.image_view.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image_to_view()

    def closeEvent(self, event):
        self.stop_auto_detect(silent=True)
        if self.cam is not None:
            try:
                self.cam.close_device()
            except Exception as exc:
                logger.error(f"关闭窗口时断开相机失败: {exc}")
            finally:
                self.cam = None
        event.accept()


def tablewidget_to_dataframe(table: QTableWidget) -> pd.DataFrame:
    rows = table.rowCount()
    cols = table.columnCount()
    headers = []
    for col in range(cols):
        item = table.horizontalHeaderItem(col)
        headers.append(item.text() if item else f"col_{col}")

    data = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            item = table.item(row, col)
            row_data.append(item.text() if item else "")
        data.append(row_data)

    return pd.DataFrame(data, columns=headers)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget {
            font-family: "Microsoft YaHei", "SimSun", "Arial";
            font-size: 12pt;
        }
        QPushButton {
            background-color: #0078D7;
            color: white;
            border-radius: 6px;
            padding: 12px;
            min-height: 48px;
            font-size: 13pt;
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
        QPushButton:disabled {
            background-color: #888;
        }
        QLineEdit, QComboBox {
            border: 1px solid #aaa;
            border-radius: 4px;
            padding: 10px;
            min-height: 38px;
            font-size: 12pt;
        }
        """
    )
    win = SheetCountMainWindow()
    win.show()
    sys.exit(app.exec())
