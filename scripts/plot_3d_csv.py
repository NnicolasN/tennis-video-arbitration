"""Plot a 3D trajectory from a CSV produced by `tennis-arb triangulate`."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
