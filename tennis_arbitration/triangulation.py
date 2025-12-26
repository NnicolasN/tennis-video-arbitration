from __future__ import annotations

import numpy as np


def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear triangulation via SVD.

    Args:
        P1, P2: (3,4) projection matrices
        x1, x2: (2,) image points in pixels

    Returns:
        X: (3,) 3D point
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    if x1.size != 2 or x2.size != 2:
        raise ValueError("x1 and x2 must be 2D points")

    A = np.zeros((4, 4), dtype=float)
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    Xh = Xh / Xh[3]
    return Xh[:3]


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate a list of corresponding points."""
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError("pts1 and pts2 must be (N,2) and have identical shapes")

    out = np.zeros((pts1.shape[0], 3), dtype=float)
    for i, (a, b) in enumerate(zip(pts1, pts2)):
        out[i] = triangulate_point(P1, P2, a, b)
    return out
