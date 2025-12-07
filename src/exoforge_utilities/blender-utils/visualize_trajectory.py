#!/usr/bin/env python3

"""
visualize_trajectory.py: This script visualizes a trajectory using translations and rotations from an HDF5 file in Blender.
"""

__author__ = "Paul-Otto Müller"
__copyright__ = "Copyright 2025, Paul-Otto Müller"
__credits__ = ["Paul-Otto Müller"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Paul-Otto Müller"
__status__ = "Development"
__date__ = '30.12.2024'

import bpy
import h5py
import numpy as np
from mathutils import Matrix, Quaternion


# Load translations and rotations from an HDF5 file
def load_hdf5(hdf5_file, group_name):
    with h5py.File(hdf5_file, 'r') as f:
        # Access the specified group
        group = f[group_name]

        # Load translations
        translations = np.array(group['translations'])  # Shape: (N, 3)

        # Load rotations
        if 'rotations' in group:
            rotations = np.array(group['rotations'])  # Shape: (N, 4) for quaternions or (N, 3, 3) for matrices
        else:
            raise ValueError(f"Rotations dataset not found in group {group_name}")

        # Check sample rate (optional, for debugging/logging)
        sample_rate = group.attrs.get('sample_rate', None)
        if sample_rate:
            print(f"Sample rate for {group_name}: {sample_rate} Hz")

    return translations, rotations


# Create a sphere to represent the moving object
def create_sphere(name="TrajectorySphere", location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(location=location, radius=0.1)
    obj = bpy.context.object
    obj.name = name
    return obj


# Apply translations and rotations to animate the object
def animate_object(obj, translations, rotations):
    for frame, (translation, rotation) in enumerate(zip(translations, rotations)):
        # Set translation
        obj.location = translation

        # Convert rotation to Euler angles
        if rotation.shape == (4,):  # Quaternion
            quat = Quaternion(rotation)  # Convert to Blender's Quaternion
            obj.rotation_euler = quat.to_euler()
        elif rotation.shape == (3, 3):  # Rotation matrix
            mat = Matrix(rotation)  # Convert to Blender's Matrix
            obj.rotation_euler = mat.to_euler()
        else:
            raise ValueError(f"Invalid rotation shape: {rotation.shape}")

        # Insert keyframes for location and rotation
        obj.keyframe_insert(data_path="location", frame=frame + 1)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame + 1)


# Create a 3D curve from points (trajectory visualization)
def create_curve_from_points(points):
    # Create a new curve
    curve_data = bpy.data.curves.new(name="TrajectoryCurve", type='CURVE')
    curve_data.dimensions = '3D'

    # Add points to the curve
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)
    for i, point in enumerate(points):
        x, y, z = point
        polyline.points[i].co = (x, y, z, 1)  # (x, y, z, w)

    # Create a curve object
    curve_obj = bpy.data.objects.new("TrajectoryCurve", curve_data)
    bpy.context.collection.objects.link(curve_obj)
    return curve_obj


# Main function
def main():
    # Step 1: Load translations and rotations from HDF5
    hdf5_file = "/home/paul/MEGAsync/Uni/Promotion/workspace/exosim/scripts/kinematics/mocap/jaw_motion.h5"
    group_name = "T_1"  # Specify the group name (e.g., T_0, T_1, etc.)
    translations, rotations = load_hdf5(hdf5_file, group_name)

    # Step 2: Create a sphere object
    sphere = create_sphere(name="TrajectorySphere")

    # Step 3: Animate the sphere using the translations and rotations
    animate_object(sphere, translations, rotations)

    # Step 4: Create a curve to visualize the trajectory
    create_curve_from_points(translations)
