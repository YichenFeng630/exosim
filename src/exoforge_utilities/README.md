# Exoforge Utilities

This directory collects standalone helper scripts for model processing and analysis. Below is an overview of the available tooling and example invocations.

## Flex vertex remapping

`update_flex_vertex_bodies.py` remaps selected flex vertices to different bodies while preserving their world-space poses. The script recomputes the coordinates in the destination body frame and can optionally apply a user-specified local offset.

### Basic usage

Dry-run a single vertex reassignment without writing to disk:

```
python3 src/exoforge_utilities/update_flex_vertex_bodies.py \
  --model src/exoforge_description/mjcf/exoskeleton/jaw/mjmodel.xml \
  --flex-name jaw_skin_flexcomp \
  --vertices 257 258 259 \
  --new-body _jaw_skin \
  --dry-run --verbose
```

The script prints a summary of the converted coordinates. Omit `--dry-run` and provide `--output` or `--in-place` to persist the edits.

### Mapping file workflow

Larger remapping jobs can be described in a JSON file:

```json
[
  {"vertices": [257, 258, "259-265"], "new_body": "_jaw_skin", "offset": [0.0, 0.0, 0.0]},
  {"vertices": "566,567,591", "new_body": "_jaw_skin_support", "offset": [0.0, -0.001, 0.0]}
]
```

Execute the script with the mapping file to update all groups in one run:

```
python3 src/exoforge_utilities/update_flex_vertex_bodies.py \
  --model src/exoforge_description/mjcf/exoskeleton/jaw/mjmodel.xml \
  --flex-name jaw_skin_flexcomp \
  --mapping my_vertex_swaps.json \
  --output data/jaw_skin_remapped.xml
```

Pass `--in-place` to modify the original MJCF file directly; a `.bak` backup is created automatically in that mode.

### Recording reproducible runs

Add `--record-script scripts/replay_flex_run.sh` (choose any path) to generate a
helper script that captures the executed command. The generated script supports
two modes:

- `run` (default): executes the stored command once. If it has already been
  applied, the script exits without re-running.
- `revert`: restores the original MJCF contents using the backup captured during
  the initial run. For newly created files, the revert mode simply removes the
  generated file.

Both the state file and the backups are colocated with the replay script, making
it easy to version the entire workflow alongside the model files.

### Notes

- Vertex indices refer to the global `flexvert` order reported by MuJoCo.
- The destination body name must exist in the compiled model.
- The optional `offset` is interpreted in the destination body frame after aligning the vertex by its world pose.

---

## Site generation from mesh vertices

`add_sites_from_vertices.py` appends `<site>` elements to a target body using
coordinates sourced either from mesh vertices or manually provided triplets. The
script converts the coordinates into the body frame, applies optional styling,
and writes the updated MJCF file.

### Mesh-driven sites

Sample run that copies vertices `12-15` from an OBJ mesh into the hand body:

```
python3 src/exoforge_utilities/add_sites_from_vertices.py \
  --model src/exoforge_description/mjcf/hands/model.xml \
  --body right_hand \
  --mesh-file assets/hands/right_hand.obj \
  --vertices 12-15 \
  --site-prefix tactile \
  --site-size 0.004 \
  --site-rgba 0.1 0.8 0.2 1.0 \
  --output data/right_hand_with_sites.xml
```

By default, vertex indices are treated as 0-based; pass `--index-base 1` if your
mesh tooling numbers vertices starting at 1. Use `--mesh-scale` (scalar or
3-vector) when the mesh needs to be rescaled before site placement.

### Manual coordinates and mixed inputs

You can add sites from explicit coordinates, optionally combining them with
mesh vertices in a single command:

```
python3 src/exoforge_utilities/add_sites_from_vertices.py \
  --model src/exoforge_description/mjcf/hands/model.xml \
  --body right_hand \
  --coords 0.02 0.00 0.03  0.025 -0.01 0.028 \
  --coord-frame body \
  --site-names pinch_tip pinch_side \
  --site-attrs rgba=0.9 0.6 0.1 1.0
```

Coordinates default to the body frame; choose `--coord-frame world` if inputs
are already expressed in world space. When using another body as the source
frame, specify `--reference-body NAME`.

### Flex body anchors

Provide `--flex-prefix` together with `--vertices` to place sites directly
on the flex component bodies themselves. Each specified vertex number maps to a
body named `<prefix>_<vertex>` and receives a site located at `0 0 0` in that
body's frame:

```
python3 src/exoforge_utilities/add_sites_from_vertices.py \
  --model src/exoforge_description/mjcf/exoskeleton/jaw/mjmodel_wo_chin.xml \
  --flex-prefix jaw_exo_mask_flexcomp \
  --vertices 1-3 40 44 \
  --site-prefix mask_landmark \
  --site-size 0.003 \
  --output data/jaw_mask_flex_sites.xml
```

The `--vertices` flag doubles for both mesh-driven selections and flex body
targets; provide ranges or individual indices as needed. Use `--site-names` if
you prefer explicit naming; otherwise the prefix is combined with the vertex
label.

### Additional options

- `--site-group`, `--site-type`, and `--site-class` map directly to the
  corresponding MuJoCo attributes.
- Provide repeated `--site-attrs key=value` pairs for any extra attributes.
- Use `--dry-run` to preview transformed positions without writing to disk.
- Pass `--in-place` to overwrite the input model, or `--output` to create a new
  file (default: `<model>_sites.xml`).
- Add `--record-script scripts/replay_sites.sh` (choose any path) to emit a
  runnable bash helper capturing the command and an automatic revert workflow.

---

## Mesh format conversion

`mesh_converter.py` converts meshes between different formats using VTK. Supports VTP, OBJ, STL, and PLY formats.

### Convert single file

```
python3 src/exoforge_utilities/mesh_converter.py obj2vtp \
  --input meshes/model.obj \
  --output meshes/model.vtp
```

### Batch convert directory

```
python3 src/exoforge_utilities/mesh_converter.py vtp2obj \
  --input-dir ./meshes \
  --output-dir ./converted
```

### Auto-detect formats

```
python3 src/exoforge_utilities/mesh_converter.py convert \
  --input meshes/sample.ply \
  --output meshes/sample.stl
```

---

## Mesh validation and repair

`mesh_validate.py` checks mesh quality and repairs common issues for MuJoCo compatibility.

### Check mesh quality

```
python3 src/exoforge_utilities/mesh_validate.py \
  --input meshes/collision_mesh.obj \
  --check
```

Reports manifoldness, watertightness, convexity, vertex/face counts, and volume ratios.

### Repair mesh

```
python3 src/exoforge_utilities/mesh_validate.py \
  --input meshes/broken_mesh.obj \
  --repair \
  --output meshes/repaired_mesh.obj
```

Fixes non-manifold edges, fills holes, and ensures watertightness.

---

## Convex decomposition

`coacd` wraps the CoACD library to decompose non-convex meshes into convex components suitable for MuJoCo collision detection.

### Basic decomposition

```
python3 src/exoforge_utilities/coacd \
  --input meshes/complex_shape.obj \
  --output meshes/convex_parts.obj \
  --threshold 0.05
```

### Fine-grained decomposition

```
python3 src/exoforge_utilities/coacd \
  --input meshes/detailed_mesh.obj \
  --output meshes/detailed_convex/ \
  --threshold 0.01 \
  --resolution 4000
```

Lower threshold values (0.01-0.05) produce more fine-grained decompositions. Higher resolution improves accuracy but increases computation time.

---

## Vertex extraction from planes

`extract_vertices.py` extracts mesh vertices above or below a specified cutting plane, useful for pinning vertices in flex simulations.

### Extract vertices above plane

```
python3 src/exoforge_utilities/extract_vertices.py \
  --model src/exoforge_description/mjcf/anatomical/jaw/jaw_skin_model.xml \
  --plane-normal 0 0 1 \
  --plane-offset -0.01 \
  --output config/pinned_vertices_top.xml
```

This finds all vertices where `0*x + 0*y + 1*z + (-0.01) > 0`, effectively selecting vertices above z = 0.01.

---

## Bulk file renaming

`remove_string.py` recursively removes a substring from all file and directory names in a tree.

### Remove suffix from filenames

```
python3 src/exoforge_utilities/remove_string.py \
  meshes/exports/ \
  "_exported"
```

Renames `model_exported.obj` to `model.obj`, `texture_exported.png` to `texture.png`, etc.

---

## Blender utilities

The `blender-utils/` directory contains Blender Python scripts for integration with simulation workflows.

### Export attachment points

`core.py` provides functions to export vertex group coordinates from Blender meshes:

```python
import bpy
from blender_utils.core import export_attachments

export_attachments(
    object_names=["jaw_model", "skull"],
    file_path="attachments.csv",
    scaling_factor=0.001
)
```

### Visualize trajectories

`visualize_trajectory.py` loads HDF5 trajectory data and animates objects in Blender:

```python
import bpy
from blender_utils.visualize_trajectory import visualize_trajectory

visualize_trajectory(
    hdf5_file="simulation_data.h5",
    group_name="mandible_motion",
    target_object="mandible_mesh"
)
```

---

## Additional mesh transformation tools

Three specialized mesh transformation utilities are available:

- `transform_mjcf.py` - Transform coordinates in MJCF files
- `transform_msh.py` - Transform mesh files (.msh format)
- `transform_obj.py` - Transform OBJ mesh files

These scripts apply rigid transformations (translation, rotation, scaling) to mesh coordinates while preserving file structure.
