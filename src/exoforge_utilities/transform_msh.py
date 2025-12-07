#!/usr/bin/env python3
"""
transform_msh.py

A command-line utility to apply an arbitrary rigid-body transform
to all nodes in a Gmsh .msh mesh file, where the transform can be
specified as:
  • a 12-value transformation matrix (3×4: rotation + translation),
  • Euler angles + translation,
  • quaternion + translation.

Usage:
    python transform_msh.py input.msh output.msh [--matrix M1 … M12]
                                           [--euler ROLL PITCH YAW --pos X Y Z]
                                           [--quat QW QX QY QZ --pos X Y Z]

Arguments:
    input             Path to the input Gmsh .msh file.
    output            Path to the output Gmsh .msh file.

Options:
    --matrix M1 … M12   Twelve floats defining the 3×4 transform matrix:
                        [ R11 R12 R13 Tx  R21 R22 R23 Ty  R31 R32 R33 Tz ]
    --euler R P Y       Euler angles (degrees) roll, pitch, yaw.
    --quat W X Y Z      Quaternion (W, X, Y, Z).
    --pos X Y Z         Translation vector (meters). Required with --euler or --quat.
    --center X Y Z      Center of rotation (default: [0,0,0]).

Dependencies:
    - meshio
    - numpy

Author:
    Paul-Otto Müller
    Date: 2025-07-22
"""

import argparse
import numpy as np
import meshio

def build_matrix_from_args(args):
    """
    Build a 3×4 transformation matrix from user arguments.
    Priority: --matrix > (--euler + --pos) > (--quat + --pos).
    Returns a (3,4) numpy array.
    """
    if args.matrix:
        M = np.array(args.matrix, dtype=float)
        if M.size != 12:
            raise ValueError("Expected 12 values for --matrix")
        return M.reshape(3, 4)
    # Euler + pos
    if args.euler is not None:
        if args.pos is None:
            raise ValueError("--pos required with --euler")
        roll, pitch, yaw = np.deg2rad(args.euler)
        R = euler_rotation_matrix(roll, pitch, yaw)
        T = np.array(args.pos, dtype=float).reshape(3, 1)
        return np.hstack((R, T))
    # Quat + pos
    if args.quat is not None:
        if args.pos is None:
            raise ValueError("--pos required with --quat")
        R = quat_to_matrix(args.quat)
        T = np.array(args.pos, dtype=float).reshape(3, 1)
        return np.hstack((R, T))
    raise ValueError("One of --matrix, --euler, or --quat must be specified")

def euler_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return 3×3 rotation matrix from roll, pitch, yaw (radians)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_x = np.array([[1,    0,     0],
                    [0,   cr,   -sr],
                    [0,   sr,    cr]])
    R_y = np.array([[ cp,  0,   sp],
                    [  0,  1,    0],
                    [-sp,  0,   cp]])
    R_z = np.array([[cy,  -sy,   0],
                    [sy,   cy,   0],
                    [ 0,    0,   1]])
    return R_z @ R_y @ R_x

def quat_to_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = map(float, quat)
    # normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x+z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x+y*y)]
    ])

def transform_mesh(input_path, output_path, M, center):
    """
    Apply transformation M (3×4) about center to mesh points.
    """
    mesh = meshio.read(input_path)
    pts = mesh.points.astype(float)
    # shift to center, apply R and T, shift back
    R = M[:, :3]
    T = M[:, 3]
    pts = ((pts - center) @ R.T) + center + T
    mesh.points = pts
    meshio.write(output_path, mesh, file_format="gmsh")
    print(f"✅ Written transformed mesh to '{output_path}'")

def parse_args():
    p = argparse.ArgumentParser(
        description="Apply rigid-body transform to Gmsh .msh mesh"
    )
    p.add_argument("input", help="Input .msh file")
    p.add_argument("output", help="Output .msh file")
    p.add_argument(
        "--matrix",
        nargs=12,
        metavar=("M1","M2","M3","Tx","M4","M5","M6","Ty","M7","M8","M9","Tz"),
        help="12 values of 3×4 transform matrix"
    )
    p.add_argument(
        "--euler",
        nargs=3,
        type=float,
        metavar=("ROLL","PITCH","YAW"),
        help="Euler angles (deg) roll, pitch, yaw"
    )
    p.add_argument(
        "--quat",
        nargs=4,
        metavar=("W","X","Y","Z"),
        help="Quaternion (w, x, y, z)"
    )
    p.add_argument(
        "--pos",
        nargs=3,
        type=float,
        metavar=("X","Y","Z"),
        help="Translation vector"
    )
    p.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=[0.0,0.0,0.0],
        metavar=("CX","CY","CZ"),
        help="Center of rotation (default: origin)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    center = np.array(args.center, dtype=float)
    M = build_matrix_from_args(args)
    transform_mesh(args.input, args.output, M, center)

if __name__ == "__main__":
    main()
