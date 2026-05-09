# Sheet Count

当前项目保留两种图片计数方法。

## 文件

- `hough_dbscan_count.py`：Hough 线段检测 + DBSCAN 聚类方法，支持随机调参。
- `height_pitch_count.py`：亮色堆叠高度除以片距的方法，支持左/中/右/全局四路投票。

## 数据

- 输入图片：`images/`
- 期望数量：`images/num.txt`
- 标签按图片文件名排序后逐行对应。

期望数量：

```text
Image_20260509112750520.jpg -> 7
Image_20260509112817782.jpg -> 9
Image_20260509112834931.jpg -> 4
Image_20260509112846582.jpg -> 6
```

## 输出目录

- `hough_dbscan_count.py` 默认运行 -> `outputs/hough_dbscan_count/`
- `hough_dbscan_count.py --tune` 最优结果 -> `outputs/hough_dbscan_count/tuned_best/`
- `height_pitch_count.py` -> `outputs/height_pitch_count/`

## Python

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' -m pip install opencv-python numpy scikit-learn
```

## 运行 Height Pitch 四路投票

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\height_pitch_count.py
```

投票规则：

1. 分别计算 `left`、`center`、`right`、`global` 四路计数。
2. 先取四路众数。
3. 若众数并列，选与四路中位值最接近的候选。
4. 若仍并列，按 `global > center > left > right` 决策。

每张图会输出：

`left_count`, `center_count`, `right_count`, `global_count`, `final_count`, `vote_reason`。

## 运行 Hough DBSCAN 默认模式

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\hough_dbscan_count.py
```

## 运行 Hough DBSCAN 随机调参（50 次）

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\hough_dbscan_count.py --tune --trials 50 --seed 42
```

调参目标：

1. 最小化总绝对误差。
2. 若并列，优先命中数更高。

调参输出包含：

1. 50 轮日志（`pred`, `abs_error`, `exact`, `cfg`）。
2. 最优参数摘要。
3. 使用最优参数的复跑验证结果。
