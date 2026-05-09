from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output_height_pitch"
MODEL_DIR = OUT_DIR / "model"
MODEL_FILE = MODEL_DIR / "latest_pitch_model.json"

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


def read_image_paths(paths: list[str]) -> list[Path]:
    return sorted(Path(p) for p in paths if Path(p).suffix.lower() in IMAGE_SUFFIXES)


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


def base_roi(image: np.ndarray, config: PitchCounterConfig) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1 = int(width * config.roi_x1_ratio)
    x2 = int(width * config.roi_x2_ratio)
    y1 = int(height * config.roi_y1_ratio)
    y2 = int(height * config.roi_y2_ratio)
    return x1, y1, x2, y2


def split_rois(roi: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    mid = x1 + (x2 - x1) // 2
    return {"left": (x1, y1, mid, y2), "right": (mid, y1, x2, y2), "global": (x1, y1, x2, y2)}


def threshold_percentile(global_gray: np.ndarray, base_threshold: float) -> float:
    return float((global_gray <= base_threshold).mean() * 100.0)


def region_threshold(image: np.ndarray, roi: tuple[int, int, int, int], percentile: float, config: PitchCounterConfig) -> float:
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if config.gaussian_blur > 1:
        gray = cv2.GaussianBlur(gray, (config.gaussian_blur, config.gaussian_blur), 0)
    threshold = float(np.percentile(gray, percentile))
    return float(np.clip(threshold, config.min_region_threshold, config.max_region_threshold))


def count_region(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float,
    config: PitchCounterConfig,
    region_name: str,
) -> RegionCountResult:
    x1, y1, x2, y2 = roi
    patch = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if config.gaussian_blur > 1:
        gray = cv2.GaussianBlur(gray, (config.gaussian_blur, config.gaussian_blur), 0)

    bright_pixels_by_row = (gray >= threshold).mean(axis=1)
    row_profile = smooth_profile(bright_pixels_by_row, config.row_smooth_window)
    row_mask_before = row_profile >= config.row_bright_fraction

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.vertical_close_kernel))
    row_mask_after = (
        cv2.morphologyEx((row_mask_before.astype(np.uint8) * 255)[:, None], cv2.MORPH_CLOSE, kernel)[:, 0] > 0
    )
    runs = all_true_runs(row_mask_after)
    total_thickness = sum(end - start + 1 for start, end in runs)
    count = max(0, int(round(total_thickness / config.sheet_pitch_px)))
    abs_runs = [(s + y1, e + y1) for s, e in runs]

    return RegionCountResult(
        region=region_name,
        count=count,
        total_thickness=total_thickness,
        all_runs=abs_runs,
        roi=roi,
        threshold_used=float(threshold),
        gray=gray,
        row_profile=row_profile,
        row_mask_before_close=row_mask_before,
        row_mask_after_close=row_mask_after,
    )


def choose_by_priority(candidates: list[int], results: dict[str, RegionCountResult]) -> int:
    for name in TIE_BREAK_PRIORITY:
        if results[name].count in candidates:
            return results[name].count
    return candidates[0]


def vote_counts(results: dict[str, RegionCountResult]) -> VoteResult:
    counts = [results["left"].count, results["right"].count, results["global"].count]
    freq = Counter(counts)
    mode_freq = max(freq.values())
    mode_candidates = sorted([k for k, v in freq.items() if v == mode_freq])
    if len(mode_candidates) == 1:
        return VoteResult(mode_candidates[0], f"unique_mode_{mode_candidates[0]}", results)

    median_value = float(np.median(np.array(counts, dtype=np.float32)))
    distance = {c: abs(c - median_value) for c in mode_candidates}
    min_distance = min(distance.values())
    nearest = sorted([c for c, d in distance.items() if d == min_distance])
    if len(nearest) == 1:
        return VoteResult(nearest[0], f"mode_tie_median_nearest_{nearest[0]}", results)

    chosen = choose_by_priority(nearest, results)
    return VoteResult(chosen, f"mode_tie_priority_{chosen}", results)


def count_with_vote(image: np.ndarray, config: PitchCounterConfig) -> VoteResult:
    rois = split_rois(base_roi(image, config))
    gx1, gy1, gx2, gy2 = rois["global"]
    global_gray = cv2.cvtColor(image[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
    if config.gaussian_blur > 1:
        global_gray = cv2.GaussianBlur(global_gray, (config.gaussian_blur, config.gaussian_blur), 0)

    percentile = threshold_percentile(global_gray, config.bright_threshold)
    thresholds = {k: region_threshold(image, r, percentile, config) for k, r in rois.items()}
    results = {k: count_region(image, rois[k], thresholds[k], config, k) for k in ("left", "right", "global")}
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
        f"thr={vote.results['left'].threshold_used:.1f}/{vote.results['right'].threshold_used:.1f}/{vote.results['global'].threshold_used:.1f}"
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
        points: list[tuple[int, int]] = []
        for i, value in enumerate(profile):
            px = int(x0 + i * (x1 - x0) / (n - 1))
            py = int(y1 - value * (y1 - y0))
            points.append((px, py))
        cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (80, 255, 80), 1)
    cv2.putText(panel, f"{result.region} cnt={result.count} total={result.total_thickness}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
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


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class HeightPitchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Height Pitch Counter")
        self.resize(1400, 860)
        self.base_config = PitchCounterConfig()
        self.active_config = self.load_model_or_default()
        self.train_images: list[Path] = []
        self.detect_images: list[Path] = []
        self.detect_votes: dict[str, VoteResult] = {}
        self.detect_preview: dict[str, np.ndarray] = {}

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        mode_bar = QHBoxLayout()
        self.btn_train_mode = QPushButton("训练模式")
        self.btn_detect_mode = QPushButton("检测模式")
        self.btn_train_mode.clicked.connect(lambda: self.mode_stack.setCurrentIndex(0))
        self.btn_detect_mode.clicked.connect(lambda: self.mode_stack.setCurrentIndex(1))
        mode_bar.addWidget(self.btn_train_mode)
        mode_bar.addWidget(self.btn_detect_mode)
        mode_bar.addStretch(1)
        root_layout.addLayout(mode_bar)

        self.mode_stack = QStackedWidget()
        root_layout.addWidget(self.mode_stack, 1)
        self.mode_stack.addWidget(self._build_train_page())
        self.mode_stack.addWidget(self._build_detect_page())
        self._apply_teal_theme()
        self._refresh_model_label()

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        self.btn_train_load = QPushButton("导入训练图片")
        self.btn_train_run = QPushButton("开始训练")
        self.btn_train_load.clicked.connect(self.load_train_images)
        self.btn_train_run.clicked.connect(self.run_training)
        top.addWidget(self.btn_train_load)
        top.addWidget(self.btn_train_run)
        top.addStretch(1)
        layout.addLayout(top)

        self.train_table = QTableWidget(0, 4)
        self.train_table.setHorizontalHeaderLabels(["图片", "真实数量", "global厚度", "L/R/G计数"])
        self.train_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.train_table, 1)
        self.train_status = QLabel("训练要求：至少 4 张图，逐行输入真实数量。")
        layout.addWidget(self.train_status)
        return page

    def _build_detect_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        top = QHBoxLayout()
        self.btn_detect_load = QPushButton("导入检测图片")
        self.btn_detect_run = QPushButton("执行检测")
        self.btn_detect_export = QPushButton("导出本次结果")
        self.btn_detect_load.clicked.connect(self.load_detect_images)
        self.btn_detect_run.clicked.connect(self.run_detection)
        self.btn_detect_export.clicked.connect(self.export_detection)
        top.addWidget(self.btn_detect_load)
        top.addWidget(self.btn_detect_run)
        top.addWidget(self.btn_detect_export)
        top.addStretch(1)
        left.addLayout(top)

        self.model_label = QLabel("")
        left.addWidget(self.model_label)
        self.detect_table = QTableWidget(0, 8)
        self.detect_table.setHorizontalHeaderLabels(["图片", "L", "R", "G", "TL", "TR", "TG", "final(reason)"])
        self.detect_table.horizontalHeader().setStretchLastSection(True)
        self.detect_table.currentCellChanged.connect(self.on_detect_table_select)
        left.addWidget(self.detect_table, 1)
        self.detect_status = QLabel("请先训练并生成模型，再检测。")
        left.addWidget(self.detect_status)
        layout.addLayout(left, 3)

        right = QVBoxLayout()
        self.preview_label = QLabel("预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet("border:1px solid #2f6f73; background:#102a2d;")
        right.addWidget(self.preview_label, 1)
        layout.addLayout(right, 2)
        return page

    def _apply_teal_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #0f1d22; color: #d7f2f2; font-size: 13px; }
            QPushButton { background: #0e7490; color: white; border: 1px solid #14b8a6; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background: #0891b2; }
            QTableWidget { background: #10262a; gridline-color: #2b555b; }
            QHeaderView::section { background: #11444a; color: #d7f2f2; padding: 5px; }
            """
        )

    def load_model_or_default(self) -> PitchCounterConfig:
        if not MODEL_FILE.exists():
            return PitchCounterConfig()
        try:
            cfg = json.loads(MODEL_FILE.read_text(encoding="utf-8")).get("config", {})
            return PitchCounterConfig(**cfg)
        except Exception:
            return PitchCounterConfig()

    def _refresh_model_label(self) -> None:
        if MODEL_FILE.exists():
            self.model_label.setText(f"当前模型: {MODEL_FILE} | sheet_pitch={self.active_config.sheet_pitch_px}")
        else:
            self.model_label.setText("当前模型: 未找到，请先训练。")

    def load_train_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择训练图片", str(ROOT), "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        self.train_images = read_image_paths(files)
        self.train_table.setRowCount(len(self.train_images))
        for row, path in enumerate(self.train_images):
            self.train_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.train_table.setItem(row, 1, QTableWidgetItem(""))
            self.train_table.setItem(row, 2, QTableWidgetItem(""))
            self.train_table.setItem(row, 3, QTableWidgetItem(""))
        self.train_status.setText(f"已导入 {len(self.train_images)} 张训练图片。")

    def run_training(self) -> None:
        if len(self.train_images) < 4:
            QMessageBox.warning(self, "训练失败", "训练图片少于 4 张。")
            return
        labels: list[int] = []
        for row in range(len(self.train_images)):
            item = self.train_table.item(row, 1)
            text = item.text().strip() if item else ""
            if not text.isdigit() or int(text) <= 0:
                QMessageBox.warning(self, "训练失败", f"第 {row + 1} 行真实数量无效。")
                return
            labels.append(int(text))

        thickness_values: list[int] = []
        lrg_texts: list[str] = []
        for row, path in enumerate(self.train_images):
            image = cv2.imread(str(path))
            if image is None:
                QMessageBox.warning(self, "训练失败", f"读取失败: {path}")
                return
            vote = count_with_vote(image, self.base_config)
            thickness = vote.results["global"].total_thickness
            lrg = f"{vote.results['left'].count}/{vote.results['right'].count}/{vote.results['global'].count}"
            thickness_values.append(thickness)
            lrg_texts.append(lrg)
            self.train_table.setItem(row, 2, QTableWidgetItem(str(thickness)))
            self.train_table.setItem(row, 3, QTableWidgetItem(lrg))

        sheet_pitch = int(round(sum(thickness_values) / sum(labels)))
        self.active_config = replace(self.base_config, sheet_pitch_px=sheet_pitch)
        self._save_model(labels, thickness_values, lrg_texts)
        self._refresh_model_label()
        self.train_status.setText(f"训练完成：sheet_pitch={sheet_pitch}，样本={len(labels)}。")
        QMessageBox.information(self, "训练完成", f"单片厚度估计完成：{sheet_pitch}px")

    def _save_model(self, labels: list[int], thickness_values: list[int], lrg_texts: list[str]) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sample_count": len(labels),
            "sum_labels": int(sum(labels)),
            "sum_thickness": int(sum(thickness_values)),
            "sheet_pitch_px": int(self.active_config.sheet_pitch_px),
            "config": asdict(self.active_config),
            "samples": [
                {
                    "image": self.train_images[i].name,
                    "label_count": labels[i],
                    "global_total_thickness": thickness_values[i],
                    "lrg_counts": lrg_texts[i],
                }
                for i in range(len(labels))
            ],
        }
        MODEL_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_detect_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择检测图片", str(ROOT), "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        self.detect_images = read_image_paths(files)
        self.detect_table.setRowCount(0)
        self.detect_votes.clear()
        self.detect_preview.clear()
        self.detect_status.setText(f"已导入 {len(self.detect_images)} 张检测图片。")

    def run_detection(self) -> None:
        if not MODEL_FILE.exists():
            QMessageBox.warning(self, "检测失败", "未找到模型，请先完成训练。")
            return
        if not self.detect_images:
            QMessageBox.warning(self, "检测失败", "请先导入检测图片。")
            return
        self.active_config = self.load_model_or_default()
        self._refresh_model_label()

        self.detect_table.setRowCount(len(self.detect_images))
        self.detect_votes.clear()
        self.detect_preview.clear()
        for row, path in enumerate(self.detect_images):
            image = cv2.imread(str(path))
            if image is None:
                continue
            vote = count_with_vote(image, self.active_config)
            self.detect_votes[path.name] = vote

            debug = image.copy()
            x1, y1, x2, y2 = vote.results["global"].roi
            mid = x1 + (x2 - x1) // 2
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.line(debug, (mid, y1), (mid, y2), (255, 200, 0), 2)
            cv2.putText(debug, f"final={vote.final_count} reason={vote.vote_reason}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.detect_preview[path.name] = debug

            self.detect_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.detect_table.setItem(row, 1, QTableWidgetItem(str(vote.results["left"].count)))
            self.detect_table.setItem(row, 2, QTableWidgetItem(str(vote.results["right"].count)))
            self.detect_table.setItem(row, 3, QTableWidgetItem(str(vote.results["global"].count)))
            self.detect_table.setItem(row, 4, QTableWidgetItem(str(vote.results["left"].total_thickness)))
            self.detect_table.setItem(row, 5, QTableWidgetItem(str(vote.results["right"].total_thickness)))
            self.detect_table.setItem(row, 6, QTableWidgetItem(str(vote.results["global"].total_thickness)))
            self.detect_table.setItem(row, 7, QTableWidgetItem(f"{vote.final_count} ({vote.vote_reason})"))

        if self.detect_images:
            self.detect_table.selectRow(0)
            self.on_detect_table_select(0, 0, -1, -1)
        self.detect_status.setText(f"检测完成：{len(self.detect_images)} 张。")

    def on_detect_table_select(self, current_row: int, _current_col: int, _prev_row: int, _prev_col: int) -> None:
        if current_row < 0 or current_row >= self.detect_table.rowCount():
            return
        name_item = self.detect_table.item(current_row, 0)
        if name_item is None:
            return
        image = self.detect_preview.get(name_item.text())
        if image is None:
            return
        pix = bgr_to_pixmap(image).scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    def export_detection(self) -> None:
        if not self.detect_images or not self.detect_votes:
            QMessageBox.warning(self, "导出失败", "请先执行检测。")
            return
        overview_dir = OUT_DIR / "overview"
        inter_dir = OUT_DIR / "intermediate"
        reports_dir = OUT_DIR / "reports"
        for d in (overview_dir, inter_dir, reports_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, object]] = []
        region_rows: list[dict[str, object]] = []
        for path in self.detect_images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            vote = self.detect_votes[path.name]
            draw_overview(image, vote, None, overview_dir / f"{path.stem}_overview.jpg")
            image_dir = inter_dir / path.stem
            for region_name in ("left", "right", "global"):
                region = vote.results[region_name]
                region_dir = image_dir / region_name
                region_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(region_dir / "01_gray.jpg"), region.gray)
                cv2.imwrite(str(region_dir / "02_row_mask_before_close.jpg"), (region.row_mask_before_close.astype(np.uint8) * 255)[:, None])
                cv2.imwrite(str(region_dir / "03_row_mask_after_close.jpg"), (region.row_mask_after_close.astype(np.uint8) * 255)[:, None])
                cv2.imwrite(str(region_dir / "04_profile_panel.jpg"), render_profile_panel(region, self.active_config))
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
                    "expected": "",
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
                    "error": "",
                }
            )

        self._write_reports(summary_rows, region_rows)
        QMessageBox.information(self, "导出完成", f"已导出到: {OUT_DIR}")

    def _write_reports(self, summary_rows: list[dict[str, object]], region_rows: list[dict[str, object]]) -> None:
        reports_dir = OUT_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_headers = ["image", "expected", "left_count", "right_count", "global_count", "left_total", "right_total", "global_total", "left_thr", "right_thr", "global_thr", "final_count", "vote_reason", "error"]
        region_headers = ["image", "region", "count", "total_thickness", "threshold", "run_count", "runs"]
        with open(reports_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_headers)
            writer.writeheader()
            writer.writerows(summary_rows)
        with open(reports_dir / "regions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=region_headers)
            writer.writeheader()
            writer.writerows(region_rows)
        with open(reports_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("height_pitch_gui detect summary\n")
            f.write(f"images={len(summary_rows)}\n")
            f.write(f"model={MODEL_FILE}\n")
            f.write(f"config={self.active_config}\n")
            f.write(f"export_time={datetime.now().isoformat(timespec='seconds')}\n")


def main() -> None:
    app = QApplication([])
    win = HeightPitchWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
