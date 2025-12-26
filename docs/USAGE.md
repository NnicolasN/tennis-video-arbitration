# Usage notes

## Coordinate conventions

The library uses:

- **Image points**: `(x, y)` in pixels, with `(0,0)` at the top-left.
- **World points**: `(X, Y, Z)` in meters.

The example 3D points in `docs/examples/points_3d_tennis_court.json` use a simple court frame:

- `(0,0,0)` is one court corner
- `X` runs along the court length (0 → 23.77 m)
- `Y` runs across the court width (0 → 10.97 m)
- `Z` points upward

The two net-post points use `Z=1.07 m` (net height at posts), matching the prototype.

## Collecting 2D correspondences

1. Extract a reference frame (first frame, or a still image).
2. Run the point picker:

```bash
python scripts/select_points.py --image data/frame.jpg --out data/points_cam1.json
```

3. Make sure the clicked 2D points are in the **same order** as the 3D file.

## Calibration quality

After `tennis-arb calibrate`, you get a reprojection RMSE (pixels). If it is large, typical causes are:

- wrong point order
- low-quality clicks (zoom in when selecting)
- motion blur / rolling shutter
- points not exactly on the assumed 3D plane
