import numpy as np

from tennis_arbitration.calibration import estimate_projection_matrix, project
from tennis_arbitration.triangulation import triangulate_points


def _random_camera(rng: np.random.Generator) -> np.ndarray:
    # Simple random camera: P = K [R|t]
    K = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Random rotation via QR
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    R = Q
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    t = rng.normal(scale=2.0, size=(3, 1))
    Rt = np.hstack([R, t])
    return K @ Rt


def test_triangulation_recovers_points():
    rng = np.random.default_rng(0)
    P1 = _random_camera(rng)
    P2 = _random_camera(rng)

    X = rng.normal(size=(50, 3))
    x1 = project(P1, X)
    x2 = project(P2, X)

    X_hat = triangulate_points(P1, P2, x1, x2)

    err = np.linalg.norm(X_hat - X, axis=1)
    assert float(np.median(err)) < 1e-2


def test_dlt_estimates_projection_matrix_up_to_scale():
    rng = np.random.default_rng(1)
    P = _random_camera(rng)
    X = rng.normal(size=(30, 3))
    x = project(P, X)

    P_hat = estimate_projection_matrix(X, x)

    # Compare reprojection error instead of raw matrix (scale ambiguity)
    x_hat = project(P_hat, X)
    rmse = np.sqrt(np.mean(np.sum((x_hat - x) ** 2, axis=1)))
    assert float(rmse) < 1e-6
