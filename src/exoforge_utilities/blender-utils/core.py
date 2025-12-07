#!/usr/bin/env python3
import os
import sys
import csv
import math
from typing import List, Dict, Tuple, Sequence, Optional

import bpy
from mathutils import Matrix, Vector

# Automatically add the folder containing this script to sys.path
script_dir = os.path.dirname(bpy.data.texts["core.py"].filepath)
if script_dir not in sys.path:
    sys.path.append(script_dir)

# import visualize_trajectory as vt


# def compute_coordinate_system(points: np.ndarray, origin_index: int = -1) -> np.ndarray:
#     """
#     Compute a coordinate system from three or more points in space using PCA, ensuring that the resulting
#     coordinate system is orthonormal.
#
#     :param points: A numpy array of shape (n, 3) where n is the number of points.
#     :param origin_index: The index of the point to be used as the origin. If -1, the geometrical center is used.
#     :return: A homogeneous transformation matrix (4x4) representing the computed coordinate system.
#     """
#     if points.shape[0] < 3:
#         raise ValueError("At least three points are required to compute a coordinate system.")
#
#     if origin_index == -1:
#         origin = np.mean(points, axis=0)
#     else:
#         origin = points[origin_index]
#
#     centered_points = points - origin
#
#     pca = PCA(n_components=3)
#     pca.fit(centered_points)
#
#     coordinate_system = pca.components_.T
#     coordinate_system = QualysisData._make_rotation_matrix_orthonormal(coordinate_system)
#
#     T = np.eye(4)
#     T[:3, :3] = coordinate_system
#     T[:3, 3] = origin
#
#     return T


def export_attachments(
    object_names: Sequence[str],
    file_path: str,
    scaling_factor: float = 1.0,
    transform: Matrix = Matrix.Identity(4)
) -> None:
    """
    Export attachment (single-vertex vertex group) coordinates shared across multiple objects.

    Behavior:
    - Scans the provided objects.
    - For every vertex group that contains exactly one vertex in each object, collects its world-space
      coordinate per object where it exists.
    - Applies: scaled_coords = scale * world_coord, then homogeneous transform (4x4) including rotation + translation.
    - Writes ONE CSV row per vertex group aggregating all objects that contain that group.
    - Dynamic columns accommodate the maximum number of objects sharing any vertex group.
    - If a group exists in >=2 objects, an extra column 'norm_12' gives the Euclidean distance between
      the first two listed object coordinates (useful for pair distance checks). Blank if <2.

    Transform:
      A 4x4 homogeneous Matrix (rotation + optional translation; scale should be provided via scaling_factor).

    :param object_names: List of object names to inspect.
    :param file_path: Output CSV path.
    :param scaling_factor: Uniform scale applied to coordinates before transform.
    :param transform: 4x4 homogeneous transform applied after scaling (Matrix.Identity(4) by default).
    """
    # Ensure all provided objects exist in the scene
    objects: List["bpy.types.Object"] = [bpy.data.objects[name] for name in object_names if name in bpy.data.objects]
    if len(objects) != len(object_names):
        print("Error: One or more objects in the provided list could not be found.")
        return

    vertex_groups_dict: Dict[str, List[Tuple[str, Vector]]] = {}  # vg_name -> list[(object_name, Vector)]
    # Collect single-vertex group coordinates
    for obj in objects:
        if obj.type != 'MESH':
            print(f"Warning: Object '{obj.name}' is not a mesh. Skipping.")
            continue
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for vg in obj.vertex_groups:
            group_indices: List[int] = [v.index for v in obj.data.vertices if any(g.group == vg.index for g in v.groups)]
            if len(group_indices) != 1:
                continue
            vertex = obj.data.vertices[group_indices[0]]
            global_coords: Vector = obj.matrix_world @ vertex.co
            scaled_coords: Vector = global_coords * scaling_factor
            # Apply 4x4 homogeneous transform (preserve translation)
            transformed_4 = transform @ scaled_coords.to_4d()
            transformed_coords: Vector = Vector((transformed_4.x, transformed_4.y, transformed_4.z))
            vertex_groups_dict.setdefault(vg.name, []).append((obj.name, transformed_coords))

    if not vertex_groups_dict:
        print("No qualifying single-vertex groups found.")
        return

    # Determine maximum number of objects any group spans
    max_objs: int = max(len(v) for v in vertex_groups_dict.values())
    add_norm: bool = max_objs >= 2

    # Build header
    header: List[str] = ["vertex_group", "count"]
    for i in range(1, max_objs + 1):
        header.extend([f"object_name_{i}", f"x_{i}", f"y_{i}", f"z_{i}"])
    if add_norm:
        header.append("norm_12")

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for vg_name, data in sorted(vertex_groups_dict.items()):
            # Sort entries by object name for deterministic ordering
            data_sorted: List[Tuple[str, Vector]] = sorted(data, key=lambda x: x[0])
            row: List[Optional[str | float]] = [vg_name, len(data_sorted)]
            # Append per-object data
            for obj_name, coords in data_sorted:
                row.extend([obj_name,
                            round(coords.x, 6),
                            round(coords.y, 6),
                            round(coords.z, 6)])
            # Pad missing slots
            missing = max_objs - len(data_sorted)
            for _ in range(missing):
                row.extend(["", "", "", ""])
            # Distance between first two (if desired)
            if add_norm:
                if len(data_sorted) >= 2:
                    c1 = data_sorted[0][1]
                    c2 = data_sorted[1][1]
                    d: float = math.sqrt((c2.x - c1.x) ** 2 + (c2.y - c1.y) ** 2 + (c2.z - c1.z) ** 2)
                    row.append(round(d, 6))
                else:
                    row.append("")
            writer.writerow(row)

    print(f"Attachment data written to '{file_path}'. Groups: {len(vertex_groups_dict)} (max objects per group: {max_objs}).")


# def export_vertex_groups_to_csv(
#     object_names: Sequence[str],
#     file_path: str,
#     scaling_factor: float = 1.0,
#     rotation: Matrix = Matrix.Identity(3)
# ) -> None:
#     """
#     Exports vertex group coordinates of objects to a CSV file if the vertex group contains only one vertex.
    
#     :param object_names: A list of Blender objects to process.
#     :param file_path: Path to the CSV file to save the output.
#     :param scaling_factor: A factor to scale the vertex coordinates.
#     :param rotation: A 3x3 rotation matrix to apply to the scaled coordinates.
#     """
#     # Ensure all provided objects exist in the scene
#     objects: List["bpy.types.Object"] = [bpy.data.objects[name] for name in object_names if name in bpy.data.objects]

#     if len(objects) != len(object_names):
#         print("Error: One or more objects in the provided list could not be found.")
#     else:
#         # Open the CSV file for writing
#         with open(file_path, mode='w', newline='') as csvfile:
#             writer = csv.writer(csvfile)

#             # Write the header row
#             writer.writerow(['object_name', 'vertex_group', 'x', 'y', 'z'])

#             # Iterate through the objects
#             for obj in objects:
#                 # Ensure the object is a mesh
#                 if obj.type != 'MESH':
#                     print(f"Skipping {obj.name}: Not a mesh object.")
#                     continue

#                 # Switch to object mode and ensure the object has vertex groups
#                 bpy.context.view_layer.objects.active = obj
#                 bpy.ops.object.mode_set(mode='OBJECT')

#                 if not obj.vertex_groups:
#                     print(f"Skipping {obj.name}: No vertex groups.")
#                     continue

#                 # Access the mesh data
#                 mesh = obj.data  # type: ignore[assignment]

#                 # Iterate through vertex groups
#                 for vg in obj.vertex_groups:
#                     # Collect vertex indices belonging to the vertex group
#                     vertex_indices: List[int] = [
#                         i for i, v in enumerate(mesh.vertices)
#                         if any(g.group == vg.index for g in v.groups)
#                     ]

#                     # Check if the vertex group has exactly one vertex
#                     if len(vertex_indices) == 1:
#                         vertex_index = vertex_indices[0]
#                         vertex = mesh.vertices[vertex_index]
#                         global_co: Vector = obj.matrix_world @ vertex.co
#                         scaled_co: Vector = global_co * scaling_factor
#                         rotated_co: Vector = rotation @ scaled_co

#                         # Write to the CSV file
#                         writer.writerow(
#                             [obj.name, vg.name, round(rotated_co.x, 6), round(rotated_co.y, 6), round(rotated_co.z, 6)])
#                     else:
#                         print(
#                             f"Skipping vertex group '{vg.name}' in object '{obj.name}': Contains {len(vertex_indices)} vertices.")

#         print(f"Vertex group data successfully exported to {file_path}")


def main() -> None:
    # Provide a list of object names to search
    object_names: List[str] = [
        "mandible", "skull", "maxilla", "hyoid",
        "tmj_disc_imprint.cube.left.bool.proc.cut",
        "tmj_disc_imprint.cube.right.bool.proc.cut"
    ]

    # Get the directory of the currently opened .blend file
    # If the file hasn't been saved yet, it will default to Blender's temp directory
    project_directory: str = bpy.path.abspath("//")

    # Define the output CSV file path
    output_csv_file: str = project_directory + "attachments.csv"

    # Coordinates scaling factora
    co_scale: float = 1e-3

    # Define a transformation matrix (for rotation, scaling, or translation)
    # Blender model -> MuJoCo (4x4 homogeneous rotation about Z)
    transformation_matrix: Matrix = Matrix.Rotation(math.radians(-90), 4, 'Z')

    # Export vertex group coordinates to a CSV file
    export_attachments(object_names, output_csv_file, co_scale, transformation_matrix)

    # Alternatively, export all vertex groups with one vertex to a CSV file
    # object_names_2: List[str] = ["tmj_plane_constr_left", "tmj_plane_constr_right"]
    # output_csv_file_2: str = project_directory + "vertex_group_coordinates.csv"
    # export_vertex_groups_to_csv(object_names_2, output_csv_file_2, co_scale, transformation_matrix)


if __name__ == "__main__":
    main()
    #vt.main()
