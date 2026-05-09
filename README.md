# Sheet Count

This project keeps two image-counting methods.

## Files

- `hough_dbscan_count.py`: Hough line detection + DBSCAN clustering method, with optional random tuning.
- `height_pitch_count.py`: Bright stack height divided by pitch method, with left/center/right/global voting.

## Data

- Input images: `images/`
- Expected counts: `images/num.txt`
- Labels are matched by sorted image filename order.

Expected counts:

```text
Image_20260509112750520.jpg -> 7
Image_20260509112817782.jpg -> 9
Image_20260509112834931.jpg -> 4
Image_20260509112846582.jpg -> 6
```

## Output Folders

- `hough_dbscan_count.py` default run -> `outputs/hough_dbscan_count/`
- `hough_dbscan_count.py --tune` best run -> `outputs/hough_dbscan_count/tuned_best/`
- `height_pitch_count.py` -> `outputs/height_pitch_count/`

## Python

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' -m pip install opencv-python numpy scikit-learn
```

## Run Height Pitch Voting

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\height_pitch_count.py
```

Vote mechanism:

1. Compute counts on `left`, `center`, `right`, and `global` ROI.
2. Take mode of four counts.
3. If mode tie, pick the candidate nearest to the 4-way median.
4. If still tied, apply priority `global > center > left > right`.

Each image prints:

`left_count`, `center_count`, `right_count`, `global_count`, `final_count`, `vote_reason`.

## Run Hough DBSCAN Default

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\hough_dbscan_count.py
```

## Run Hough DBSCAN Random Tuning (50)

```powershell
& 'C:\Users\11941\AppData\Local\Programs\Python\Python310\python.exe' .\hough_dbscan_count.py --tune --trials 50 --seed 42
```

Tuning objective:

1. Minimize total absolute error.
2. If tied, maximize exact matches.

Tuning output includes:

1. 50 trial logs (`pred`, `abs_error`, `exact`, `cfg`).
2. Best config summary.
3. Verification run with the best config.
