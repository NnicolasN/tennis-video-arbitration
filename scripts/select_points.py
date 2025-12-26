"""Interactively click points on an image and export them to JSON.

Example:
    python scripts/select_points.py --image data/frame.jpg --out data/points_cam1.json

Keys:
    - left click: add point
    - backspace: remove last point
    - q / esc: quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def select_points(image_path: Path) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    cv2.namedWindow("select-points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select-points", on_mouse)

    while True:
        img = cv2.imread(str(image_path))
        if img is None:
            raise SystemExit(f"Could not read image: {image_path}")

        for (x, y) in points:
            cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        cv2.imshow("select-points", img)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (8, 127) and points:  # backspace / delete
            points.pop()

    cv2.destroyAllWindows()
    return points


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pts = select_points(args.image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([[x, y] for x, y in pts], f, indent=2)

    print(f"Saved {len(pts)} points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
