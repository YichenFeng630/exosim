#!/usr/bin/env python3

"""
mesh_validate.py

Command-line tool for validating and repairing 3D meshes for MuJoCo.

Checks manifoldness, watertightness, convexity, vertex/face counts, and volume ratio.
Optionally repairs common issues and exports a repaired mesh.
"""

import argparse
import sys
import numpy as np
import trimesh
# ensure networkx is available for convex_decomposition
import networkx  

def check_mesh(mesh):
    """
    Compute diagnostic metrics for a trimesh.Trimesh object.
    Returns a dict of metrics.
    """
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    watertight = mesh.is_watertight

    # Count edges
    faces = mesh.faces
    all_edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    non_manifold_edges_count = int(np.sum(counts > 2))
    boundary_edges_count = int(np.sum(counts == 1))

    is_convex = mesh.is_convex
    volume = mesh.volume

    hull_volume = None
    try:
        hull = mesh.convex_hull
        hull_volume = hull.volume
    except Exception:
        pass

    volume_ratio = None
    if hull_volume and hull_volume > 0:
        volume_ratio = volume / hull_volume

    return {
        'vertex_count': vertex_count,
        'face_count': face_count,
        'watertight': watertight,
        'non_manifold_edges_count': non_manifold_edges_count,
        'boundary_edges_count': boundary_edges_count,
        'is_convex': is_convex,
        'volume': volume,
        'convex_hull_volume': hull_volume,
        'volume_ratio': volume_ratio
    }

def format_metrics(metrics):
    """
    Format the metrics dict into human-readable text.
    """
    lines = [
        f"Vertices: {metrics['vertex_count']}",
        f"Faces: {metrics['face_count']}",
        f"Watertight (2-manifold): {metrics['watertight']}",
        f"Non-manifold edges: {metrics['non_manifold_edges_count']}",
        f"Boundary edges: {metrics['boundary_edges_count']}",
        f"Convex: {metrics['is_convex']}",
        f"Mesh volume: {metrics['volume']:.6f}" if metrics['volume'] is not None else "Mesh volume: N/A",
        f"Convex hull volume: {metrics['convex_hull_volume']:.6f}" if metrics['convex_hull_volume'] is not None else "Convex hull volume: N/A",
        f"Volume ratio (orig/convex_hull): {metrics['volume_ratio']:.6f}" if metrics['volume_ratio'] is not None else "Volume ratio: N/A"
    ]
    return "\n".join(lines)

def repair_mesh(mesh):
    """
    Attempt to repair common mesh issues:
    - Fill small holes
    - Merge near-duplicate vertices
    - Fix normals and remove degenerate faces
    - Convex decomposition fallback
    Returns the repaired mesh.
    """
    repaired = mesh.copy()
    repaired.fill_holes()
    repaired.merge_vertices()
    repaired.remove_degenerate_faces()
    repaired.fix_normals()

    # Attempt trimesh repair utilities
    try:
        trimesh.repair.fill_holes(repaired)
        trimesh.repair.fix_normals(repaired)
    except Exception:
        pass

    # If still not watertight, fallback to largest convex component
    if not repaired.is_watertight:
        try:
            parts = repaired.convex_decomposition()
            if parts:
                repaired = max(parts, key=lambda m: m.volume)
        except Exception:
            pass

    return repaired

def main():
    parser = argparse.ArgumentParser(
        description="Validate and repair 3D meshes for MuJoCo collision requirements."
    )
    parser.add_argument(
        'input_mesh',
        help="Path to input mesh file (OBJ/STL/PLY, etc.)"
    )
    parser.add_argument(
        '--check', action='store_true',
        help="Perform mesh diagnostics and print metrics."
    )
    parser.add_argument(
        '--repair', action='store_true',
        help="Attempt to repair mesh and optionally export repaired version."
    )
    parser.add_argument(
        '--export', metavar='OUTPUT_MESH',
        help="Path to save repaired mesh (requires --repair)."
    )

    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_mesh)
        if not isinstance(mesh, trimesh.Trimesh):
            # handle scene by concatenating geometry
            mesh = mesh.dump(concatenate=True)
    except Exception as e:
        print(f"Error: Unable to load mesh '{args.input_mesh}': {e}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        metrics = check_mesh(mesh)
        print("Mesh diagnostics:")
        print(format_metrics(metrics))

    if args.repair:
        before_metrics = check_mesh(mesh)
        repaired = repair_mesh(mesh)
        after_metrics = check_mesh(repaired)

        print("\n--- Before Repair ---")
        print(format_metrics(before_metrics))
        print("\n--- After Repair ---")
        print(format_metrics(after_metrics))

        if args.export:
            try:
                repaired.export(args.export)
                print(f"\nRepaired mesh exported to: {args.export}")
            except Exception as e:
                print(f"Error: Unable to export repaired mesh: {e}", file=sys.stderr)
                sys.exit(1)

    if not args.check and not args.repair:
        parser.print_help()
        sys.exit(0)

if __name__ == '__main__':
    main()
