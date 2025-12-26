from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetectorConfig:
    """Parameters controlling the 2D ball detector.

    The default values are inspired by the original prototype scripts.
    """

    threshold: int = 25
    morph_kernel: int = 5
    min_blob_area: int = 10
    ball_min_area: int = 15
    ball_max_area: int = 150

    remove_human_blob: bool = True
    human_min_area: int = 200
    proximity_threshold_px: int = 50


def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    out = binary.copy()
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(out, [c], -1, 0, -1)
    return out


def _largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _rect_near(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int], threshold: int) -> bool:
    x1, y1, x2, y2 = r1
    a1, b1, a2, b2 = r2
    return (
        abs(x1 - a1) < threshold
        or abs(x2 - a2) < threshold
        or abs(y1 - b1) < threshold
        or abs(y2 - b2) < threshold
    )


def _human_like_region(binary: np.ndarray, largest: np.ndarray, cfg: BallDetectorConfig) -> list[np.ndarray]:
    """Grow a region around the largest contour by merging nearby contours.

    This is a *heuristic* used to remove the player & racket from a diff image.
    """

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [largest]

    x, y, w, h = cv2.boundingRect(largest)
    current = (x, y, x + w, y + h)

    for c in contours:
        if c is largest:
            continue
        if cv2.contourArea(c) < cfg.human_min_area:
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        r = (cx, cy, cx + cw, cy + ch)
        if _rect_near(current, r, cfg.proximity_threshold_px):
            blobs.append(c)
            current = (min(current[0], r[0]), min(current[1], r[1]), max(current[2], r[2]), max(current[3], r[3]))

    return blobs


class BallDetector:
    """Frame-difference-based tennis ball detector."""

    def __init__(self, cfg: BallDetectorConfig = BallDetectorConfig()):
        self.cfg = cfg
        self._prev_candidates: list[tuple[int, int]] = []

    def reset(self) -> None:
        self._prev_candidates = []

    def detect_candidates(self, prev_frame_bgr: np.ndarray, frame_bgr: np.ndarray) -> list[tuple[int, int, float]]:
        """Return candidate ball centers for the current frame.

        Returns a list of (x, y, score) where score is a simple confidence proxy.
        """

        prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, self.cfg.threshold, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_kernel, self.cfg.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = _remove_small_blobs(mask, self.cfg.min_blob_area)

        if self.cfg.remove_human_blob:
            largest = _largest_contour(mask)
            if largest is not None and cv2.contourArea(largest) >= self.cfg.human_min_area:
                blobs = _human_like_region(mask, largest, self.cfg)
                for c in blobs:
                    cv2.drawContours(mask, [c], -1, 0, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[int, int, float]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if not (self.cfg.ball_min_area <= area <= self.cfg.ball_max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / h
            # Perfect circle ~1; keep lenient thresholds.
            if not (0.3 <= aspect <= 3.0):
                continue
            cx, cy = x + w // 2, y + h // 2
            score = 1.0 / (1.0 + abs(1.0 - aspect))
            candidates.append((cx, cy, score))

        return candidates

    def choose_ball(self, candidates: Iterable[tuple[int, int, float]]) -> Optional[tuple[int, int, float]]:
        """Pick one candidate using a motion heuristic."""

        candidates = list(candidates)
        if not candidates:
            self._prev_candidates = []
            return None

        if not self._prev_candidates:
            # If first time, choose highest score.
            best = max(candidates, key=lambda t: t[2])
            self._prev_candidates = [(c[0], c[1]) for c in candidates]
            return best

        def motion_score(c: tuple[int, int, float]) -> float:
            cx, cy, s = c
            # maximize distance from previous candidates (ball is small & moves fast)
            d = max(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in self._prev_candidates)
            return d + 10.0 * s

        best = max(candidates, key=motion_score)
        self._prev_candidates = [(c[0], c[1]) for c in candidates]
        return best
