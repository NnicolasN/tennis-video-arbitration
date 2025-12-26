from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .ball_detection import BallDetector, BallDetectorConfig
from .calibration import calibrate
from .court_detection import CourtDetector, CourtDetectorConfig
from .io import (
    load_npy,
    load_points_2d,
    load_points_3d,
    read_points_2d_csv,
    save_json,
    save_npy,
    write_points_2d_csv,
)
from .triangulation import triangulate_points


def _cmd_detect_ball(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    ok, prev = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame")

    detector = BallDetector(BallDetectorConfig())
    detections: list[tuple[int, float, float, float]] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cands = detector.detect_candidates(prev, frame)
        chosen = detector.choose_ball(cands)

        if chosen is not None:
            x, y, conf = chosen
            detections.append((frame_idx, float(x), float(y), float(conf)))
            if args.show:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)

        if args.show:
            cv2.imshow("detect-ball", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev = frame
        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    if args.out:
        write_points_2d_csv(detections, args.out)
        print(f"Wrote {len(detections)} detections to {args.out}")
    else:
        print(f"Detected {len(detections)} points")

    return 0


def _cmd_detect_court(args: argparse.Namespace) -> int:
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    cfg = CourtDetectorConfig(resize_factor=args.resize)
    det = CourtDetector(cfg)
    result = det.detect(img)

    if args.out:
        save_json(result, args.out)
        print(f"Wrote court detection to {args.out}")

    if args.show:
        dbg = CourtDetector.draw_debug(img if args.resize == 1.0 else cv2.resize(img, (0, 0), fx=args.resize, fy=args.resize), result)
        cv2.imshow("detect-court", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    X = load_points_3d(args.points_3d)
    x = load_points_2d(args.points_2d)

    res = calibrate(X, x)
    print(f"Reprojection RMSE: {res.reprojection_rmse_px:.3f} px")

    if args.out:
        save_npy(res.P, args.out)
        print(f"Saved P to {args.out}")
    return 0


def _cmd_triangulate(args: argparse.Namespace) -> int:
    P1 = load_npy(args.P1)
    P2 = load_npy(args.P2)
    pts1 = read_points_2d_csv(args.points1)
    pts2 = read_points_2d_csv(args.points2)

    n = min(len(pts1), len(pts2))
    if n == 0:
        raise SystemExit("No points to triangulate")

    X = triangulate_points(P1, P2, pts1[:n], pts2[:n])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "x,y,z"
    np.savetxt(out_path, X, delimiter=",", header=header, comments="")
    print(f"Wrote {n} 3D points to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tennis-arb", description="Simplified tennis video arbitration tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ball = sub.add_parser("detect-ball", help="Detect the ball in 2D from a video")
    p_ball.add_argument("--video", type=Path, required=True, help="Path to the input video")
    p_ball.add_argument("--out", type=Path, default=None, help="Output CSV path")
    p_ball.add_argument("--show", action="store_true", help="Show an OpenCV window (press q to quit)")
    p_ball.set_defaults(func=_cmd_detect_ball)

    p_court = sub.add_parser("detect-court", help="Detect court lines and keypoints from an image")
    p_court.add_argument("--image", type=Path, required=True, help="Path to a reference frame")
    p_court.add_argument("--out", type=Path, default=None, help="Output JSON path")
    p_court.add_argument("--resize", type=float, default=1.0, help="Resize factor (e.g. 0.33)")
    p_court.add_argument("--show", action="store_true", help="Show a debug visualization")
    p_court.set_defaults(func=_cmd_detect_court)

    p_cal = sub.add_parser("calibrate", help="Calibrate a camera (estimate projection matrix P) using DLT")
    p_cal.add_argument("--points-3d", dest="points_3d", type=Path, required=True, help="JSON [[X,Y,Z],...] in meters")
    p_cal.add_argument("--points-2d", dest="points_2d", type=Path, required=True, help="JSON [[x,y],...] in pixels")
    p_cal.add_argument("--out", type=Path, required=True, help="Output .npy file for P")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_tri = sub.add_parser("triangulate", help="Triangulate 3D points from 2 cameras")
    p_tri.add_argument("--P1", type=Path, required=True, help="Projection matrix for camera 1 (.npy)")
    p_tri.add_argument("--P2", type=Path, required=True, help="Projection matrix for camera 2 (.npy)")
    p_tri.add_argument("--points1", type=Path, required=True, help="CSV from detect-ball for camera 1")
    p_tri.add_argument("--points2", type=Path, required=True, help="CSV from detect-ball for camera 2")
    p_tri.add_argument("--out", type=Path, required=True, help="Output CSV for triangulated 3D points")
    p_tri.set_defaults(func=_cmd_triangulate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
