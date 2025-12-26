"""Tennis video arbitration (simplified) library.

Modules:
- ball_detection: 2D ball detection from video frames
- court_detection: court line detection and keypoint extraction
- calibration: camera calibration (DLT) from 2D↔3D correspondences
- triangulation: 3D reconstruction from 2 calibrated cameras
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
