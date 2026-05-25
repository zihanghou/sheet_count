
from pathlib import Path
import shutil
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

# ====== main param zone ======
INPUT_DIR = ROOT / "img"
OUTPUT_DIR = ROOT / "output"
PITCH_PX = 83.5

# 单张输入图只输出两张图片到 output 根目录
SAVE_MIDDLE_SUMMARY = True
SAVE_COUNT_RESULT = True
CLEAN_OUTPUT = True
JPG_QUALITY = 82

# 汇总中间图：2x2 面板，0.5 表示最终图尺寸约等于原图
SUMMARY_PANEL_SCALE = 0.5

# 节拍优化：手部/指甲掩码按缩小图计算，再映射回原图
HAND_PROCESS_SCALE = 0.5
USE_HAND_COLUMN_EXCLUSION = False
USE_FAST_BRIGHT_CHANNEL = False

# ====== algorithm param zone ======
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

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

# 亮区边界筛选
MIN_ROW_FRACTION = 0.01
MIN_COL_FRACTION = 0.005
MIN_COMPONENT_AREA = 80
MIN_COL_BRIGHT_PIXELS = 3

# 手部/指甲筛选
HAND_DILATE = 9
HAND_MIN_COMPONENT_AREA = 1800
HAND_KEEP_TOP_K = 3
HAND_TOP_RATIO = 0.34
NAIL_TOP_RATIO = 0.26
NAIL_NEAR_SKIN_DILATE = 25

# 排除手遮挡列
HAND_COL_OCCLUSION_MAX = 0.22

# 保留片材主体：用宽度和长宽比去掉指甲、小亮斑
PLATE_MIN_WIDTH_RATIO = 0.16
PLATE_MIN_ASPECT = 1.5
PLATE_MIN_AREA = 250
PLATE_BAND_MARGIN = 35

# 高度鲁棒统计
HEIGHT_OUTLIER_MAD_SCALE = 3.5
HEIGHT_PERCENTILE = 75

cv2.setUseOptimized(True)


def reset_dir(p: Path):
    if CLEAN_OUTPUT and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def image_paths(p: Path):
    return sorted(
        x for x in p.iterdir()
        if x.suffix.lower() in IMAGE_SUFFIXES
        and "mask" not in x.stem.lower()
    )


def imwrite_jpg(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])


def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def post_process(mask):
    mask = mask.astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
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
    """
    单次连通域版本：
    1. 一次 connectedComponents 同时完成小区域过滤和主体锚点搜索；
    2. 用横向长条锚点确定片材主体 y/x 带；
    3. 只保留主体带内且面积达标的组件，减少重复 CC 耗时。
    """
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

def make_overlay(img, mask, color=(0, 0, 255), alpha=0.55):
    vis = img.copy()
    m = mask > 0
    if not np.any(m):
        return vis

    color_img = np.empty_like(img)
    color_img[:] = color
    blended = cv2.addWeighted(img, 1.0 - alpha, color_img, alpha, 0)
    vis[m] = blended[m]
    return vis


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
            # 越靠上越像手部，排序时兼顾面积和位置
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
        # 当前场景基本是灰度成像，直接取 B 通道比 HSV 转换更快。
        v = img[:, :, 0]
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

    v = cv2.GaussianBlur(v, (5, 5), 0)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return post_process(mask)


def hsv_otsu_mask_from_hsv(hsv):
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
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

    top, bottom, height, edge_angle = measure_column_group(
        filtered_xs,
        filtered_tops,
        filtered_bottoms,
    )

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
        top, bottom, height, seg_angle = measure_column_group(
            group_xs,
            group_tops,
            group_bottoms,
        )

        source = f"seg{i + 1}"
        info[f"{source}_x1"] = int(group_xs[0])
        info[f"{source}_x2"] = int(group_xs[-1])
        info[f"{source}_top_y"] = top
        info[f"{source}_bottom_y"] = bottom
        info[f"{source}_height_px"] = height
        info[f"{source}_edge_angle_deg"] = round(seg_angle, 4)
        info[f"{source}_valid_cols"] = int(len(group_xs))

    return info


def count_from_height(height_px, pitch_px=PITCH_PX):
    if height_px == "" or height_px <= 0:
        return 0
    return max(0, int(round(float(height_px) / float(pitch_px))))


def height_count_candidates(info, pitch_px=PITCH_PX):
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
    winners = [
        count
        for count, score in scores.items()
        if abs(score - best_score) < 1e-6
    ]

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
        font_scale = float(np.clip(vis.shape[1] / 2600.0 * 0.82, 0.68, 1.12))
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


def draw_measure_overlay(img, info, candidates=None, method_count=None, hand_mask=None, concise=False, draw_columns=True):
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

    if draw_columns:
        for x1, x2 in x_runs(info.get("excluded_hand_xs", [])):
            cv2.rectangle(vis, (x1, 0), (x2, h - 1), (0, 0, 255), 1)

        for x1, x2 in x_runs(info.get("valid_xs", [])):
            cv2.line(vis, (x1, h - 8), (x2, h - 8), (255, 255, 0), 2)

    lines = [
        f"method_count={method_count}  total_h={info['total_height_px']}px  pitch={PITCH_PX}",
        (
            f"angle={info.get('edge_angle_deg', 0):.2f}  "
            f"valid_cols={info.get('valid_cols', 0)}/{info.get('raw_valid_cols', 0)}  "
            f"hand_excluded={info.get('excluded_hand_cols', 0)}"
        ),
    ]

    for i in range(SEGMENTS):
        source = f"seg{i + 1}"
        x1 = info[f"{source}_x1"]
        x2 = info[f"{source}_x2"]
        top = info[f"{source}_top_y"]
        bottom = info[f"{source}_bottom_y"]
        height = info[f"{source}_height_px"]
        count = candidate_by_source.get(source, {}).get("count", "")

        if not concise:
            lines.append(f"{source}: h={height}px count={count} cols={info.get(f'{source}_valid_cols', 0)}")

        if x1 == "":
            continue

        cv2.rectangle(vis, (x1, 0), (x2, h - 1), (255, 200, 0), 1)

        if height > 0:
            cv2.line(vis, (x1, top), (x2, top), (0, 255, 0), 2)
            cv2.line(vis, (x1, bottom), (x2, bottom), (0, 0, 255), 2)

    draw_text_panel(vis, lines, origin=(24, 24))
    return vis


def draw_count_result(img, result, vote_result, elapsed_ms):
    info = result["info"]
    candidates = result["candidates"]
    method_count = result["plate_count"]

    vis = draw_measure_overlay(
        img,
        info,
        candidates,
        method_count,
        hand_mask=None,
        concise=True,
    )

    total = next(item for item in candidates if item["source"] == "total")
    seg_counts = [
        item["count"]
        for item in candidates
        if item["source"].startswith("seg")
    ]

    lines = [
        f"FINAL COUNT = {vote_result['final_count']}",
        f"total_height={total['height_px']}px -> {total['count']}  pitch={PITCH_PX}",
        f"segment_counts={seg_counts}",
        f"scores={vote_result['scores']}  reason={vote_result['reason']}",
        f"algo_time={elapsed_ms:.1f} ms",
    ]
    draw_text_panel(
        vis,
        lines,
        origin=(24, 24),
        font_scale=float(np.clip(vis.shape[1] / 2200.0, 0.85, 1.35)),
        color=(0, 255, 255),
    )
    return vis


def draw_panel_label(img, label):
    vis = img.copy()
    draw_text_panel(
        vis,
        [label],
        origin=(12, 12),
        font_scale=float(np.clip(vis.shape[1] / 1600.0, 0.55, 0.85)),
        line_gap=28,
        color=(0, 255, 255),
    )
    return vis


def scale_height_info(info, sx, sy):
    out = dict(info)
    out["valid_xs"] = []
    out["excluded_hand_xs"] = []

    x_keys = ["bright_x1", "bright_x2"]
    y_keys = ["bright_y1", "bright_y2", "total_top_y", "total_bottom_y"]
    for i in range(SEGMENTS):
        source = f"seg{i + 1}"
        x_keys.extend([f"{source}_x1", f"{source}_x2"])
        y_keys.extend([f"{source}_top_y", f"{source}_bottom_y"])

    for key in x_keys:
        if out.get(key, "") != "":
            out[key] = int(round(float(out[key]) * sx))
    for key in y_keys:
        if out.get(key, "") != "":
            out[key] = int(round(float(out[key]) * sy))
    return out


def make_middle_summary(img, hmask, bright, plate_mask, info, candidates, method_count):
    h, w = img.shape[:2]
    panel_w = max(1, int(w * SUMMARY_PANEL_SCALE))
    panel_h = max(1, int(h * SUMMARY_PANEL_SCALE))
    sx = panel_w / max(1, w)
    sy = panel_h / max(1, h)

    small_img = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    small_hmask = cv2.resize(hmask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    small_bright = cv2.resize(bright, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    small_plate = cv2.resize(plate_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    small_info = scale_height_info(info, sx, sy)

    hand_vis = make_overlay(small_img, small_hmask, color=(0, 128, 255), alpha=0.40)
    bright_vis = ensure_bgr(small_bright)
    plate_vis = make_overlay(small_img, small_plate, color=(0, 0, 255), alpha=0.50)
    measure_vis = draw_measure_overlay(
        small_img,
        small_info,
        candidates,
        method_count,
        hand_mask=None,
        concise=True,
        draw_columns=False,
    )

    panels = [
        draw_panel_label(hand_vis, "01 hand/nail mask overlay"),
        draw_panel_label(bright_vis, "02 raw bright mask"),
        draw_panel_label(plate_vis, "03 plate-like bright only"),
        draw_panel_label(measure_vis, "04 measurement overlay"),
    ]

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


METHOD_NAME = "hsv_otsu"


def process_hsv_otsu(img, path, raw_hmask, hmask, hand_stats, non_hand, bright):

    # 先扣掉手部/指甲，再保留片材长条组件
    bright_non_hand = cv2.bitwise_and(bright, non_hand)
    plate_mask = keep_plate_like_components(bright_non_hand, img.shape)

    measure_mask = post_process(plate_mask)
    info = calc_segment_heights(measure_mask, hmask if USE_HAND_COLUMN_EXCLUSION else None, hand_stats)
    candidates = height_count_candidates(info, PITCH_PX)

    method_vote = vote_plate_count([{
        "method": METHOD_NAME,
        "candidates": candidates,
    }])
    method_count = method_vote["final_count"]

    return {
        "method": METHOD_NAME,
        "tilt_angle_deg": info.get("edge_angle_deg", 0.0),
        "plate_count": method_count,
        "candidates": candidates,
        "info": info,
        "debug": {
            "raw_hmask": raw_hmask,
            "hmask": hmask,
            "bright": bright,
            "plate_mask": plate_mask,
            "measure_mask": measure_mask,
        },
    }


def compact_height_info(info):
    keys = [
        "bright_x1",
        "bright_y1",
        "bright_x2",
        "bright_y2",
        "total_top_y",
        "total_bottom_y",
        "total_height_px",
        "edge_angle_deg",
        "valid_cols",
        "raw_valid_cols",
        "excluded_hand_cols",
        "outlier_cols",
        "height_stat",
        "raw_hand_components",
        "kept_hand_components",
        "raw_hand_removed_components",
    ]
    out = {key: info.get(key, "") for key in keys}

    out["segments"] = []
    for i in range(SEGMENTS):
        source = f"seg{i + 1}"
        out["segments"].append({
            "source": source,
            "x1": info.get(f"{source}_x1", ""),
            "x2": info.get(f"{source}_x2", ""),
            "top_y": info.get(f"{source}_top_y", ""),
            "bottom_y": info.get(f"{source}_bottom_y", ""),
            "height_px": info.get(f"{source}_height_px", 0),
            "valid_cols": info.get(f"{source}_valid_cols", 0),
            "edge_angle_deg": info.get(f"{source}_edge_angle_deg", 0.0),
        })

    return out


def result_summary(result):
    methods = []
    for item in result["height_results"]:
        methods.append({
            "method": item["method"],
            "plate_count": item["plate_count"],
        })

    return {
        "image": result["image"],
        "final_plate_count": result["vote_result"]["final_count"],
        "time_ms": round(result.get("time_ms", 0.0), 2),
        "methods": methods,
    }


def result_detail(result):
    methods = []
    for item in result["height_results"]:
        total = next(
            candidate
            for candidate in item["candidates"]
            if candidate["source"] == "total"
        )
        methods.append({
            "method": item["method"],
            "plate_count": item["plate_count"],
            "total_height_px": total["height_px"],
            "total_count": total["count"],
            "segment_counts": [
                candidate["count"]
                for candidate in item["candidates"]
                if candidate["source"].startswith("seg")
            ],
            "segment_heights_px": [
                candidate["height_px"]
                for candidate in item["candidates"]
                if candidate["source"].startswith("seg")
            ],
            "candidates": item["candidates"],
            "info": compact_height_info(item["info"]),
        })

    return {
        "image": result["image"],
        "final_plate_count": result["vote_result"]["final_count"],
        "time_ms": round(result.get("time_ms", 0.0), 2),
        "pitch_px": PITCH_PX,
        "methods": methods,
        "vote": {
            "scores": result["vote_result"]["scores"],
            "reason": result["vote_result"]["reason"],
        },
    }


def process_one(path: Path):
    t0 = time.perf_counter()

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")

    bright = bright_otsu_mask(img)

    raw_hmask, hmask, hand_stats = hand_masks(img)
    non_hand = cv2.bitwise_not(hmask)

    height_results = [
        process_hsv_otsu(
            img,
            path,
            raw_hmask,
            hmask,
            hand_stats,
            non_hand,
            bright,
        )
    ]

    vote_result = vote_plate_count(height_results)
    algo_ms = (time.perf_counter() - t0) * 1000.0

    best_result = height_results[0]
    debug = best_result["debug"]

    if SAVE_MIDDLE_SUMMARY:
        middle_img = make_middle_summary(
            img,
            hmask,
            debug["bright"],
            debug["plate_mask"],
            best_result["info"],
            best_result["candidates"],
            best_result["plate_count"],
        )
        imwrite_jpg(OUTPUT_DIR / f"{path.stem}_01_middle.jpg", middle_img)

    if SAVE_COUNT_RESULT:
        count_img = draw_count_result(img, best_result, vote_result, algo_ms)
        imwrite_jpg(OUTPUT_DIR / f"{path.stem}_02_count.jpg", count_img)

    total_ms = (time.perf_counter() - t0) * 1000.0

    # 不把大图数组写入 JSON
    for item in height_results:
        item.pop("debug", None)

    return {
        "image": path.name,
        "height_results": height_results,
        "vote_result": vote_result,
        "algo_ms": algo_ms,
        "time_ms": total_ms,
    }


def main():
    reset_dir(OUTPUT_DIR)
    paths = image_paths(INPUT_DIR)

    if not paths:
        raise RuntimeError(f"输入文件夹没有图像: {INPUT_DIR}")

    results = []

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")
        result = process_one(path)
        results.append(result)

        vote = result["vote_result"]
        method_counts = ", ".join(
            f"{item['method']}={item['plate_count']}"
            for item in result["height_results"]
        )
        print(
            f"  {method_counts}, final_plate_count={vote['final_count']} "
            f"scores={vote['scores']} algo={result['algo_ms']:.1f}ms total={result['time_ms']:.1f}ms"
        )

    print("处理完成")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("每张输入图仅输出两张图片：*_01_middle.jpg、*_02_count.jpg")


if __name__ == "__main__":
    main()
