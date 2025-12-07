#!/usr/bin/env python3
"""
mesh_converter.py

A modular command-line tool to convert between VTK PolyData (.vtp) and Wavefront OBJ (.obj),
and extendable to other mesh formats (e.g., STL, PLY).

Usage:
    # Convert all .vtp to .obj in a directory
    python mesh_converter.py vtp2obj --input-dir ./meshes --output-dir ./converted

    # Convert .obj to .vtp
    python mesh_converter.py obj2vtp --input ./meshes/sample.obj --output ./out/sample.vtp

    # General convert (auto-detect)
    python mesh_converter.py convert --input ./meshes/sample.ply --output ./out/sample.stl
"""

import argparse
import sys
from pathlib import Path
import vtk


def read_mesh(file_path: Path) -> vtk.vtkPolyData:
    """Read a mesh from a file into a vtkPolyData object."""
    suffix = file_path.suffix.lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    reader.SetFileName(str(file_path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read mesh or mesh is empty: {file_path}")
    return polydata


def write_mesh(polydata: vtk.vtkPolyData, file_path: Path) -> None:
    """Write a vtkPolyData object to a file."""
    suffix = file_path.suffix.lower()
    if suffix == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    elif suffix == ".obj":
        writer = vtk.vtkOBJWriter()
    elif suffix == ".stl":
        writer = vtk.vtkSTLWriter()
    elif suffix == ".ply":
        writer = vtk.vtkPLYWriter()
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    writer.SetFileName(str(file_path))
    # For XML formats, enable binary for smaller size
    if isinstance(writer, vtk.vtkXMLPolyDataWriter):
        writer.SetDataModeToBinary()
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write mesh to: {file_path}")


def convert_file(input_path: Path, output_path: Path) -> None:
    """
    Convert a single mesh file from input_path to output_path.
    Ensures output directory exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polydata = read_mesh(input_path)
    write_mesh(polydata, output_path)
    print(f"Converted {input_path} → {output_path}")


def batch_convert(input_dir: Path, output_dir: Path, src_ext: str, dst_ext: str) -> None:
    """
    Convert all files with extension src_ext in input_dir (recursively)
    and save them as dst_ext in output_dir, preserving relative paths.
    """
    for src in input_dir.rglob(f"*{src_ext}"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(dst_ext)
        convert_file(src, dst)


def cli():
    parser = argparse.ArgumentParser(
        description="Mesh Converter: vtp ↔ obj (and more) using VTK"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # vtp to obj batch
    p1 = sub.add_parser("vtp2obj", help="Convert all .vtp files in a directory to .obj")
    p1.add_argument("--input-dir", type=Path, required=True, help="Directory containing source .vtp files")
    p1.add_argument("--output-dir", type=Path, required=True, help="Directory to save .obj files")

    # obj to vtp batch
    p2 = sub.add_parser("obj2vtp", help="Convert .obj to .vtp (single or batch)")
    p2_group = p2.add_mutually_exclusive_group(required=True)
    p2_group.add_argument("--input", type=Path, help="Single .obj file to convert")
    p2_group.add_argument("--input-dir", type=Path, help="Directory containing .obj files")
    p2.add_argument("--output", type=Path, help="Output .vtp file (for single)")
    p2.add_argument("--output-dir", type=Path, help="Output directory for batch")

    # general convert
    p3 = sub.add_parser("convert", help="General mesh format conversion")
    p3.add_argument("--input", type=Path, required=True, help="Source mesh file")
    p3.add_argument("--output", type=Path, required=True, help="Target mesh file")

    args = parser.parse_args()

    try:
        if args.command == "vtp2obj":
            batch_convert(args.input_dir, args.output_dir, ".vtp", ".obj")
        elif args.command == "obj2vtp":
            if args.input:
                if not args.output:
                    parser.error("--output is required when converting a single file")
                convert_file(args.input, args.output)
            else:
                if not args.output_dir:
                    parser.error("--output-dir is required for batch conversion")
                batch_convert(args.input_dir, args.output_dir, ".obj", ".vtp")
        elif args.command == "convert":
            convert_file(args.input, args.output)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
