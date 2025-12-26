from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_points_2d(path: str | Path) -> np.ndarray:
    """Load [[x,y], ...] from JSON."""
    data = load_json(path)
    return np.asarray(data, dtype=float)


def load_points_3d(path: str | Path) -> np.ndarray:
    """Load [[X,Y,Z], ...] from JSON."""
    data = load_json(path)
    return np.asarray(data, dtype=float)


def write_points_2d_csv(points: list[tuple[int, float, float, float]], path: str | Path) -> None:
    """Write (frame,x,y,confidence) points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "confidence"])
        for frame_idx, x, y, conf in points:
            w.writerow([frame_idx, x, y, conf])


def read_points_2d_csv(path: str | Path) -> np.ndarray:
    """Read a CSV produced by `detect-ball`.

    Returns (N,2) points ordered by increasing frame.
    Missing frames are ignored.
    """
    rows: list[tuple[int, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((int(row["frame"]), float(row["x"]), float(row["y"])))

    rows.sort(key=lambda t: t[0])
    return np.asarray([[x, y] for _, x, y in rows], dtype=float)


def save_npy(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)
