import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"

# ROI 固定像素坐标（可直接改数值，不再按比例计算）
# 建议先只调 ROI，确保只覆盖板材堆叠区域，避开塑料袋边缘与强反光区。
ROI_X1 = 520
ROI_X2 = 1950
ROI_Y1 = 120
ROI_Y2 = 1780

# 迁移算法参数（来源于 sheet-counter-main 语义）
# 预处理对比度增益。值越大，纹理和噪声都会被放大。
CONTRAST_ALPHA = 1.25
# Canny 低阈值（高影响参数）：偏小会引入噪声边，偏大可能漏检细线。
CANNY_LOW = 25
# Canny 高阈值（高影响参数）：与 CANNY_LOW 配合决定边缘数量。
CANNY_HIGH = 100
# Hough 累加阈值（高影响参数）：越大越严格，线段数量会减少。
HOUGH_THRESHOLD = 100
# 最小线段长度占 ROI 宽度比例（高影响参数）：越大越不易误检短噪声线。
MIN_LINE_LENGTH_RATIO = 0.07
# Hough 线段最大连接间隙：越大越容易把断裂线拼成一条。
MAX_LINE_GAP = 10
# 水平线筛选阈值（高影响参数）：仅保留 abs(y1-y2) 小于该值的线段。
HORIZONTAL_DY_TH = 2
# DBSCAN 邻域半径（高影响参数）：越大越容易把相邻层合并成同一簇。
DBSCAN_EPS = 2
# DBSCAN 成簇最小样本数：越大越抑制孤立噪声，但可能漏掉弱层线。
DBSCAN_MIN_SAMPLES = 2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_roi(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1 = max(0, min(ROI_X1, w - 1))
    x2 = max(x1 + 1, min(ROI_X2, w))
    y1 = max(0, min(ROI_Y1, h - 1))
    y2 = max(y1 + 1, min(ROI_Y2, h))
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def dbscan_1d(values: list[int], eps: int, min_samples: int) -> list[int]:
    if not values:
        return []
    pts = np.array(values, dtype=np.int32)
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def neighbors(i: int) -> np.ndarray:
        return np.where(np.abs(pts - pts[i]) <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            continue
        labels[i] = cluster_id
        seeds = list(nbrs.tolist())
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors(j)
                if len(nbrs_j) >= min_samples:
                    for t in nbrs_j.tolist():
                        if t not in seeds:
                            seeds.append(t)
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1
        cluster_id += 1
    return labels.tolist()


def algo_migrated_dbscan(roi_bgr: np.ndarray) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    preprocessed = cv2.convertScaleAbs(roi_bgr, alpha=CONTRAST_ALPHA, beta=0)
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, apertureSize=3)

    min_line_length = max(20, int(roi_bgr.shape[1] * MIN_LINE_LENGTH_RATIO))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=MAX_LINE_GAP,
    )

    raw_vis = roi_bgr.copy()
    horizontal_vis = roi_bgr.copy()
    cluster_vis = roi_bgr.copy()
    result_vis = roi_bgr.copy()

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(raw_vis, (x1, y1), (x2, y2), (80, 170, 255), 1)
            if abs(y1 - y2) < HORIZONTAL_DY_TH:
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(horizontal_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

    y_midpoints = [int((y1 + y2) // 2) for x1, y1, x2, y2 in horizontal_lines]
    labels = dbscan_1d(y_midpoints, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    valid_labels = sorted({lb for lb in labels if lb != -1})
    migrated_count = len(valid_labels)

    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 160, 255),
        (255, 200, 80),
        (220, 120, 255),
        (120, 255, 220),
    ]
    for idx, line in enumerate(horizontal_lines):
        x1, y1, x2, y2 = line
        lb = labels[idx] if idx < len(labels) else -1
        color = (120, 120, 120) if lb == -1 else palette[lb % len(palette)]
        cv2.line(cluster_vis, (x1, y1), (x2, y2), color, 1)

    cluster_centers = []
    for lb in valid_labels:
        ys = [y_midpoints[i] for i, v in enumerate(labels) if v == lb]
        if ys:
            cluster_centers.append(int(round(float(np.mean(ys)))))
    cluster_centers.sort()

    for y in cluster_centers:
        cv2.line(result_vis, (0, y), (result_vis.shape[1] - 1, y), (0, 255, 255), 1)
    cv2.putText(
        result_vis,
        f"Migrated DBSCAN count={migrated_count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    mids = {
        "01_preprocessed.png": preprocessed,
        "02_edges_canny.png": edges,
        "03_hough_lines_raw.png": raw_vis,
        "04_horizontal_lines_filtered.png": horizontal_vis,
        "05_dbscan_clusters.png": cluster_vis,
    }
    return migrated_count, mids, result_vis


def save_gray_or_bgr(path: Path, img: np.ndarray) -> None:
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), img[:, :, :3])


def process_one(img_path: Path) -> dict[str, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read: {img_path}")

    roi, (x1, y1, x2, y2) = get_roi(img)
    out_base = OUT_DIR / img_path.stem
    ensure_dir(out_base)
    save_gray_or_bgr(out_base / "00_input.png", img)
    save_gray_or_bgr(out_base / "00_roi.png", roi)

    counts = {}
    name = "algo_migrated_dbscan"
    algo_dir = out_base / name
    ensure_dir(algo_dir)
    count, mids, result = algo_migrated_dbscan(roi)
    for fname, m in mids.items():
        save_gray_or_bgr(algo_dir / fname, m)
    save_gray_or_bgr(algo_dir / "99_result_roi.png", result)

    full = img.copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 255), 2)
    full[y1:y2, x1:x2] = result
    cv2.putText(full, f"{name}: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    save_gray_or_bgr(algo_dir / "99_result_full.png", full)
    counts[name] = count

    return counts


def main() -> None:
    ensure_dir(OUT_DIR)
    image_paths = sorted([p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}])
    if not image_paths:
        print("No images found in images/")
        return

    summary_path = OUT_DIR / "summary.csv"
    rows = []
    for p in image_paths:
        counts = process_one(p)
        row = {"image": p.name}
        row.update(counts)
        rows.append(row)
        print(f"{p.name} -> {counts}")

    headers = [
        "image",
        "migrated_dbscan_count",
    ]
    for row in rows:
        row["migrated_dbscan_count"] = row.get("algo_migrated_dbscan", 0)
    rows_for_csv = [{k: row.get(k, 0 if k != "image" else "") for k in headers} for row in rows]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
