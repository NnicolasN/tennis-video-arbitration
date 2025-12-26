from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    P: np.ndarray  # (3, 4)
    reprojection_rmse_px: float


def _to_homogeneous(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Expected (N, D) array")
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([x.astype(float), ones])


def _normalize_points_2d(points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization for 2D points.

    Returns:
        points_norm_h: (N,3) normalized homogeneous points
        T: (3,3) normalization transform
    """
    pts = np.asarray(points_2d, dtype=float)
    if pts.shape[1] != 2:
        raise ValueError("points_2d must be (N,2)")

    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    d = np.sqrt((shifted**2).sum(axis=1))
    mean_d = d.mean() if d.size else 1.0
    s = np.sqrt(2) / mean_d if mean_d > 0 else 1.0

    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1],
    ])

    pts_h = _to_homogeneous(pts)
    pts_norm = (T @ pts_h.T).T
    return pts_norm, T


def _normalize_points_3d(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization for 3D points.

    Returns:
        points_norm_h: (N,4) normalized homogeneous points
        U: (4,4) normalization transform
    """
    pts = np.asarray(points_3d, dtype=float)
    if pts.shape[1] != 3:
        raise ValueError("points_3d must be (N,3)")

    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    d = np.sqrt((shifted**2).sum(axis=1))
    mean_d = d.mean() if d.size else 1.0
    s = np.sqrt(3) / mean_d if mean_d > 0 else 1.0

    U = np.array([
        [s, 0, 0, -s * centroid[0]],
        [0, s, 0, -s * centroid[1]],
        [0, 0, s, -s * centroid[2]],
        [0, 0, 0, 1],
    ])

    pts_h = _to_homogeneous(pts)
    pts_norm = (U @ pts_h.T).T
    return pts_norm, U


def estimate_projection_matrix(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Estimate the camera projection matrix P (3x4) using normalized DLT.

    Args:
        points_3d: (N,3) world points (meters)
        points_2d: (N,2) image points (pixels)

    Returns:
        P: (3,4) projection matrix such that x ~ P X
    """
    Xn, U = _normalize_points_3d(points_3d)
    xn, T = _normalize_points_2d(points_2d)

    n = Xn.shape[0]
    if n < 6:
        raise ValueError("Need at least 6 correspondences for a stable DLT calibration")

    A = np.zeros((2 * n, 12), dtype=float)
    for i in range(n):
        X, Y, Z, W = Xn[i]
        x, y, w = xn[i]
        # Two equations per correspondence
        A[2 * i] = [
            0,
            0,
            0,
            0,
            -W * X,
            -W * Y,
            -W * Z,
            -W * W,
            y * X,
            y * Y,
            y * Z,
            y * W,
        ]
        A[2 * i + 1] = [
            W * X,
            W * Y,
            W * Z,
            W * W,
            0,
            0,
            0,
            0,
            -x * X,
            -x * Y,
            -x * Z,
            -x * W,
        ]

    # Solve Ap=0 via SVD; p is last column of V (or last row of V^T)
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)

    # Denormalize
    P = np.linalg.inv(T) @ Pn @ U

    # Scale for stability (optional): normalize so that ||P[2,:]|| = 1
    scale = np.linalg.norm(P[2, :])
    if scale > 0:
        P = P / scale

    return P


def project(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using P.

    Returns (N,2).
    """
    P = np.asarray(P, dtype=float)
    Xh = _to_homogeneous(np.asarray(points_3d, dtype=float))
    xh = (P @ Xh.T).T
    xh = xh / xh[:, 2:3]
    return xh[:, :2]


def reprojection_rmse(P: np.ndarray, points_3d: np.ndarray, points_2d: np.ndarray) -> float:
    pred = project(P, points_3d)
    gt = np.asarray(points_2d, dtype=float)
    err = pred - gt
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def calibrate(points_3d: np.ndarray, points_2d: np.ndarray) -> CalibrationResult:
    P = estimate_projection_matrix(points_3d, points_2d)
    rmse = reprojection_rmse(P, points_3d, points_2d)
    return CalibrationResult(P=P, reprojection_rmse_px=rmse)
