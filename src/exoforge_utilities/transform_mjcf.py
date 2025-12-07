#!/usr/bin/env python3
r"""
transform_mjcf.py
=================

Apply a rigid-body transform to every `pos` and `quat` attribute in an MJCF file.

Features
--------
1. Wildcard support: process multiple XML files via shell patterns.
2. Transform input via:
   1. 4×4 homogeneous matrix file (`--matrix mat.txt`)
   2. 12 numbers for [R | t] block (`--mat12 …`)
   3. Translation + intrinsic XYZ Euler angles (`--pos X Y Z --euler RX RY RZ`)
3. Fully documented with LaTeX for core formulas.

Usage Examples
--------------
# Matrix file:
python transform_mjcf.py --in "models/*.xml" --matrix T.txt --out-dir out/

# 12 numbers:
python transform_mjcf.py --in model.xml \
    --mat12 1 0 0 0  0 1 0 0  0 0 1 0 \
    --out-dir out/

# Position + Euler:
python transform_mjcf.py --in jaw.xml \
    --pos 0 0 0.1 --euler 90 0 90 \
    --out-dir out/
"""

from __future__ import annotations
import argparse
import sys
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def load_matrix_file(path: Path) -> np.ndarray:
    """
    Load a 4×4 homogeneous matrix from a text file with 16 numbers.

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


def build_matrix_12(nums: list[float]) -> np.ndarray:
    """
    Build a 4×4 transform from 12 numbers:
    [r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz].
    """
    if len(nums) != 12:
        raise ValueError("12 numbers required for --mat12")
    R = np.array(nums[0:3] + nums[4:7] + nums[8:11]).reshape(3, 3)
    t = np.array([nums[3], nums[7], nums[11]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def euler_xyz_to_matrix(rx: float, ry: float, rz: float, degrees: bool=True) -> np.ndarray:
    """
    Convert intrinsic XYZ Euler angles to 3×3 rotation matrix.
    Uses R = Rz(γ)·Ry(β)·Rx(α).

    LaTeX:
    \[
      R_x(\alpha)=\begin{bmatrix}
        1&0&0\\0&\cos\alpha&-\sin\alpha\\0&\sin\alpha&\cos\alpha
      \end{bmatrix},\ 
      R_y(\beta)=\begin{bmatrix}
        \cos\beta&0&\sin\beta\\0&1&0\\-\sin\beta&0&\cos\beta
      \end{bmatrix},\ 
      R_z(\gamma)=\begin{bmatrix}
        \cos\gamma&-\sin\gamma&0\\\sin\gamma&\cos\gamma&0\\0&0&1
      \end{bmatrix},
    \]
    \[
      R = R_z(\gamma)\,R_y(\beta)\,R_x(\alpha).
    \]
    """
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                   [0,1,0],
                   [-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz), np.cos(rz),0],
                   [0,0,1]])
    return Rz @ Ry @ Rx


def rotate_quat(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Rotate quaternion q by rotation R extracted from T:
    q' = q_rot ⊗ q, where q_rot is quaternion of R.

    Quaternion of R:
    If trace(R)=t,
    \[
      w = \tfrac12\sqrt{1+t},\quad
      x = \tfrac{R_{32}-R_{23}}{4w},\ 
      y = \tfrac{R_{13}-R_{31}}{4w},\ 
      z = \tfrac{R_{21}-R_{12}}{4w}.
    \]
    """
    R = T[:3,:3]
    t = np.trace(R)
    w = 0.5*np.sqrt(1+t)
    x = (R[2,1]-R[1,2])/(4*w)
    y = (R[0,2]-R[2,0])/(4*w)
    z = (R[1,0]-R[0,1])/(4*w)
    q_rot = np.array([w,x,y,z])
    # quaternion multiplication (q_rot ⊗ q)
    w1,x1,y1,z1 = q_rot
    w2,x2,y2,z2 = q
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def transform_xml(infile: Path, outfile: Path, T: np.ndarray) -> None:
    """
    Parse XML, transform all pos and quat attributes, write to outfile.
    """
    tree = ET.parse(infile)
    root = tree.getroot()
    for elem in root.iter():
        if 'pos' in elem.attrib:
            p = np.fromstring(elem.attrib['pos'], sep=' ', dtype=float)
            p_t = T[:3,:3] @ p + T[:3,3]
            elem.attrib['pos'] = f"{p_t[0]:.9g} {p_t[1]:.9g} {p_t[2]:.9g}"
        if 'quat' in elem.attrib:
            q = np.fromstring(elem.attrib['quat'], sep=' ', dtype=float)
            q_t = rotate_quat(q, T)
            elem.attrib['quat'] = f"{q_t[0]:.9g} {q_t[1]:.9g} {q_t[2]:.9g} {q_t[3]:.9g}"
    tree.write(outfile, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transform pos/quat in MJCF by a rigid-body transform."
    )
    p.add_argument("--in", dest="pattern", required=True,
                   help="Input XML files (shell wildcard).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--matrix", type=Path,
                     help="Text file with 16 numbers (4×4).")
    grp.add_argument("--mat12", nargs=12, type=float,
                     metavar=("r00", "r01", "r02", "tx",
                              "r10", "r11", "r12", "ty",
                              "r20", "r21", "r22", "tz"),
                     help="Twelve numbers: 3×3 R row-major + t.")
    grp.add_argument("--pos", nargs=3, type=float, metavar=("X","Y","Z"), default=(0,0,0),
                     help="Translation (meters); requires --euler.")
    p.add_argument("--euler", nargs=3, type=float, metavar=("RX","RY","RZ"), default=(0,0,0),
                   help="Intrinsic XYZ Euler (deg); requires --pos.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for transformed XML files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # build T
    if args.matrix:
        T = load_matrix_file(args.matrix)
    elif args.mat12:
        T = build_matrix_12(args.mat12)
    else:
        if args.euler is None:
            sys.exit("Error: --pos requires --euler")
        R = euler_xyz_to_matrix(*args.euler)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = np.array(args.pos)
    # expand files
    files = sorted(glob(args.pattern))
    if not files:
        sys.exit(f"No files match: {args.pattern}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        infile = Path(fp)
        outfile = args.out_dir / infile.name
        transform_xml(infile, outfile, T)
        print(f"Transformed {infile} → {outfile}")


if __name__ == "__main__":
    main()
