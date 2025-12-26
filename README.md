# Tennis Video Arbitration (simplified Hawk-Eye)

This repository contains a **clean Python implementation** of a simplified tennis video-arbitration system:

- **2D ball detection** from a video stream
- **Court line / keypoint detection** on a reference frame
- **Camera calibration** (projection matrix estimation from 2D↔3D correspondences, DLT)
- **3D triangulation** from 2 synchronized cameras

The overall method mirrors the presentation included in `docs/presentation.pdf`.

## Repository layout

```
.
├── tennis_arbitration/      # Library code
├── scripts/                # Small runnable helpers / demos
├── tests/                  # Unit tests (math-only, no videos)
├── docs/                   # Project documentation (incl. presentation)
├── data/                   # Put your input videos/images here (ignored by git)
└── outputs/                # Generated outputs (ignored by git)
```

## Install

Python ≥ 3.10.

```bash
pip install -e .
```

For development (lint/tests):

```bash
pip install -e ".[dev]"
```

## Quickstart

### 1) Detect the ball in 2D (single camera)

```bash
tennis-arb detect-ball --video data/clip.mp4 --out outputs/ball_2d.csv --show
```

This writes a CSV with columns: `frame,x,y,confidence`.

### 2) Detect court lines / corners on a reference image

```bash
tennis-arb detect-court --image data/frame.jpg --out outputs/court_keypoints.json --show
```

### 3) Calibrate a camera with known 3D court points

Create two JSON files:

- `points_3d.json`: an array of 3D points in meters `[[X,Y,Z], ...]`
- `points_2d.json`: an array of matching image points in pixels `[[x,y], ...]`

Then:

```bash
tennis-arb calibrate --points-3d data/points_3d.json --points-2d data/points_2d.json --out outputs/P1.npy
```

### 4) Triangulate a 3D trajectory (2 cameras)

```bash
tennis-arb triangulate \
  --P1 outputs/P1.npy --P2 outputs/P2.npy \
  --points1 outputs/ball_cam1.csv --points2 outputs/ball_cam2.csv \
  --out outputs/trajectory_3d.csv
```

## Notes

- This project is **not** a full Hawk-Eye reimplementation. It is intentionally simplified and intended for learning / experimentation.
- The code is written to be **headless by default** (no GUI) and only uses OpenCV windows when `--show` is provided.

## License

MIT.
