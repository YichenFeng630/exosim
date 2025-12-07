#!/usr/bin/env python3
r"""
transform_obj.py
================

Applies a rigid transformation to every vertex (`v` lines) of one or more OBJ files.

Features
--------
1.  **Wildcard support**: specify multiple files with shell-style patterns.
2.  **Matrix inputs**:
    a. From a text file containing 16 numbers (4×4 homogeneous).  
    b. From exactly 12 numbers specifying the top 3×4 block.  
    c. From a translation vector + intrinsic XYZ Euler angles.
3.  **Preserves** all non‐vertex lines (`vn`, `vt`, `f`, comments, etc.).

Usage
-----
1.  Via matrix file:
    ```
    python transform_obj.py \
        --in "models/jaw_*.obj" \
        --matrix T.txt \
        --out-dir transformed/
    ```
2.  Via 12 numbers (row‐major 3×3 R and t):
    ```
    python transform_obj.py \
        --in "models/*.obj" \
        --mat12 1 0 0 0  0 1 0 0  0 0 1 0 \
        --out-dir transformed/
    ```
3.  Via position + Euler angles:
    ```
    python transform_obj.py \
        --in "models/*.obj" \
        --pos 0.01 -0.02 0.03 \
        --euler 0 90 0 \
        --out-dir transformed/
    ```

Mathematical Background
-----------------------
Given a vertex \(\mathbf p\in\mathbb R^3\) and a rigid transform
\[
\mathbf T = \begin{bmatrix}
  \mathbf R & \mathbf t \\
  \mathbf 0^\top & 1
\end{bmatrix},
\]
the transformed coordinate is
\[
\mathbf p' = \mathbf R\,\mathbf p + \mathbf t,
\]
where \(\mathbf R\in\mathrm{SO}(3)\) and \(\mathbf t\in\mathbb R^3\).
"""

from __future__ import annotations
import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np


def load_matrix_file(path: Path) -> np.ndarray:
    """
    Load a 4×4 homogeneous matrix from a whitespace‐separated text file.
    The file must contain exactly 16 numbers (row‐major) or 4 lines×4 numbers.
    Returns
    -------
    T : (4,4) ndarray
        Homogeneous transform with last row [0,0,0,1].
    """
    vals = np.fromfile(path, sep=" ")
    if vals.size != 16:
        raise ValueError(f"{path} must contain 16 values; got {vals.size}")
    T = vals.reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Last row must be [0 0 0 1]")
    return T


def build_matrix_12(nums: Sequence[float]) -> np.ndarray:
    """
    Build a 4×4 homogeneous matrix from 12 numbers:
      [r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz]
    Returns
    -------
    T : (4,4) ndarray
    """
    if len(nums) != 12:
        raise ValueError("Exactly 12 numbers required for --mat12")
    R = np.array(nums[0:3] + nums[4:7] + nums[8:11]).reshape(3, 3)
    t = np.array([nums[3], nums[7], nums[11]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def euler_xyz_to_matrix(rx: float, ry: float, rz: float, degrees: bool = True) -> np.ndarray:
    """
    Convert intrinsic XYZ Euler angles to a 3×3 rotation matrix.
    Uses the quintic polynomial smoothstep? No, basic rotations:
    R = Rz @ Ry @ Rx.
    """
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    # Rotations about principal axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0          ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,            0,         1]])
    return Rz @ Ry @ Rx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a rigid transform to OBJ vertices (supports wildcards)."
    )
    p.add_argument(
        "--in", dest="pattern", required=True,
        help="Input OBJ files (shell wildcard)."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--matrix", type=Path,
                     help="Path to text file with 16 numbers (4×4).")
    grp.add_argument("--mat12", nargs=12, type=float,
                     metavar=("r00","r01","r02","tx","r10","r11","r12","ty",
                              "r20","r21","r22","tz"),
                     help="Twelve numbers: R(3×3) row‐major then t.")
    grp.add_argument("--pos", nargs=3, type=float, metavar=("X","Y","Z"),
                     help="Translation vector (meters). Requires --euler.")
    p.add_argument("--euler", nargs=3, type=float, metavar=("RX","RY","RZ"),
                   help="Intrinsic XYZ Euler angles (deg). Requires --pos.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write transformed OBJs.")
    return p.parse_args()


def write_transformed(infile: Path, outfile: Path, T: np.ndarray) -> None:
    """
    Reads `infile`, transforms each vertex with T, writes to `outfile`.
    Preserves all lines not starting with 'v '.
    """
    R, t = T[:3, :3], T[:3, 3]
    with infile.open("r") as src, outfile.open("w") as dst:
        for line in src:
            if line.startswith("v "):
                parts = line.split()
                p = np.fromstring(" ".join(parts[1:4]), sep=" ")
                p_t = R @ p + t
                dst.write(f"v {p_t[0]:.10g} {p_t[1]:.10g} {p_t[2]:.10g}\n")
            else:
                dst.write(line)


def main() -> None:
    args = parse_args()

    # Determine T
    if args.matrix:
        T = load_matrix_file(args.matrix)
    elif args.mat12:
        T = build_matrix_12(args.mat12)
    else:
        if args.euler is None:
            sys.exit("Error: --pos requires --euler")
        R = euler_xyz_to_matrix(*args.euler, degrees=True)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.array(args.pos, dtype=float)

    # Expand wildcards
    files = sorted(glob(args.pattern))
    if not files:
        sys.exit(f"No files match pattern: {args.pattern}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        infile = Path(path)
        outfile = args.out_dir / infile.name
        write_transformed(infile, outfile, T)
        print(f"Transformed: {infile} → {outfile}")


if __name__ == "__main__":
    main()
