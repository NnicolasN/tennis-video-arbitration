from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtDetectorConfig:
    # Preprocessing
    resize_factor: float = 1.0  # e.g. 0.33 to downscale
    threshold_value: int = 200

    # Hough
    rho: float = 1.0
    theta: float = np.pi / 180
    hough_threshold: int = 30
    min_line_length: int = 200
    max_line_gap: int = 8

    # ROI filtering (fractions of width/height)
    margin_left_px: int = 100
    margin_top_px: int = 100
    margin_right_px: int = 50
    margin_bottom_px: int = 50

    # Net post heuristic
    vertical_dx_px: int = 10
    vertical_min_height_frac: float = 0.33


def _as_line_list(lines: Optional[np.ndarray]) -> list[tuple[int, int, int, int]]:
    if lines is None:
        return []
    # HoughLinesP returns shape (N, 1, 4)
    lines = np.asarray(lines).reshape(-1, 4)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines]


def _filter_by_roi(lines: list[tuple[int, int, int, int]], roi: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    xmin, ymin, xmax, ymax = roi
    kept = []
    for x1, y1, x2, y2 in lines:
        if min(x1, x2) < xmin or min(y1, y2) < ymin or max(x1, x2) > xmax or max(y1, y2) > ymax:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


def _intersection(l1: tuple[int, int, int, int], l2: tuple[int, int, int, int]) -> Optional[tuple[int, int]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Keep only intersections that lie on both segments (with a small tolerance)
    tol = 1.0
    if (
        min(x1, x2) - tol <= px <= max(x1, x2) + tol
        and min(y1, y2) - tol <= py <= max(y1, y2) + tol
        and min(x3, x4) - tol <= px <= max(x3, x4) + tol
        and min(y3, y4) - tol <= py <= max(y3, y4) + tol
    ):
        return int(round(px)), int(round(py))
    return None


def _corners_from_intersections(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not pts:
        return []
    arr = np.array(pts, dtype=np.int32)
    x = arr[:, 0]
    y = arr[:, 1]
    # Standard heuristic for 4 extreme corners in an image
    tl = arr[np.argmin(x + y)]
    bl = arr[np.argmin(x - y)]
    tr = arr[np.argmax(x - y)]
    br = arr[np.argmax(x + y)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def _vertical_lines(lines: list[tuple[int, int, int, int]], img_h: int, cfg: CourtDetectorConfig) -> list[tuple[int, int, int, int]]:
    kept = []
    min_h = cfg.vertical_min_height_frac * img_h
    for x1, y1, x2, y2 in lines:
        if abs(x1 - x2) < cfg.vertical_dx_px and abs(y1 - y2) > min_h:
            kept.append((x1, y1, x2, y2))
    return kept


def _post_tops(verticals: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    tops = []
    for x1, y1, x2, y2 in verticals:
        tops.append((int(round((x1 + x2) / 2)), int(min(y1, y2))))
    return tops


class CourtDetector:
    """Detects court lines, intersections, corners and (optionally) net posts."""

    def __init__(self, cfg: CourtDetectorConfig = CourtDetectorConfig()):
        self.cfg = cfg

    def detect(self, image_bgr: np.ndarray) -> dict:
        cfg = self.cfg

        img = image_bgr
        if cfg.resize_factor != 1.0:
            img = cv2.resize(img, (0, 0), fx=cfg.resize_factor, fy=cfg.resize_factor)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(
            bw,
            rho=cfg.rho,
            theta=cfg.theta,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.min_line_length,
            maxLineGap=cfg.max_line_gap,
        )
        line_list = _as_line_list(lines)

        h, w = bw.shape[:2]
        roi = (
            cfg.margin_left_px,
            cfg.margin_top_px,
            w - cfg.margin_right_px,
            h - cfg.margin_bottom_px,
        )
        line_list = _filter_by_roi(line_list, roi)

        intersections: list[tuple[int, int]] = []
        for i, l1 in enumerate(line_list):
            for l2 in line_list[i + 1 :]:
                p = _intersection(l1, l2)
                if p is not None:
                    intersections.append(p)

        corners = _corners_from_intersections(intersections)

        verticals = _vertical_lines(line_list, img_h=h, cfg=cfg)
        post_tops = _post_tops(verticals)

        return {
            "image_shape": {"height": int(h), "width": int(w)},
            "roi": {"xmin": roi[0], "ymin": roi[1], "xmax": roi[2], "ymax": roi[3]},
            "lines": [list(l) for l in line_list],
            "intersections": [list(p) for p in intersections],
            "corners": [list(p) for p in corners],
            "vertical_lines": [list(l) for l in verticals],
            "net_post_tops": [list(p) for p in post_tops],
        }

    @staticmethod
    def draw_debug(image_bgr: np.ndarray, result: dict) -> np.ndarray:
        out = image_bgr.copy()
        for x1, y1, x2, y2 in result.get("lines", []):
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x, y in result.get("corners", []):
            cv2.circle(out, (x, y), 10, (0, 0, 255), -1)
        for x, y in result.get("net_post_tops", []):
            cv2.circle(out, (x, y), 10, (255, 0, 0), -1)
        return out
