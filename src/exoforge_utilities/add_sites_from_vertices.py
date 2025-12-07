#! /usr/bin/env python3
"""Add MuJoCo sites sourced from mesh vertices or manual coordinates."""

from __future__ import annotations

import argparse
import base64
import shlex
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np

try:  # pragma: no cover - dependency check
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    mujoco = None  # type: ignore[assignment]


def parse_index_sequence(tokens: Sequence[str]) -> List[int]:
    """Expand integers and closed ranges (e.g. ``5`` or ``10-15``)."""

    indices: List[int] = []

    def _handle(part: str) -> None:
        part = part.strip()
        if not part:
            return
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{part}': end < start")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    for token in tokens:
        for part in token.replace(",", " ").split():
            _handle(part)

    return indices


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertex data found in OBJ file: {path}")
    return np.asarray(vertices, dtype=float)


def load_mesh_vertices(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj_vertices(path)

    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"Unsupported mesh format '{suffix}'. Install trimesh to enable more formats."
        ) from exc

    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None or np.asarray(vertices).size == 0:
        raise ValueError(f"File '{path}' does not contain vertex data")
    return np.asarray(vertices, dtype=float)


def apply_scale(vertices: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    scale_arr = np.asarray(scale, dtype=float)
    if scale_arr.size not in (1, 3):
        raise ValueError("Scale must be a scalar or 3-vector")
    if scale_arr.size == 1:
        scale_arr = np.repeat(scale_arr, 3)
    return vertices * scale_arr.reshape(1, 3)


def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def build_replay_command_args(flag: str) -> List[str]:
    """Construct command arguments that reproduce the current invocation."""

    filtered: List[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = True
            continue
        if token.startswith(f"{flag}="):
            continue
        filtered.append(token)

    path_flags = {"--model", "--mesh-file", "--output"}

    resolved: List[str] = []
    idx = 0
    while idx < len(filtered):
        token = filtered[idx]
        matched_flag = next((flag_name for flag_name in path_flags if token.startswith(flag_name + "=")), None)
        if matched_flag:
            option, value = token.split("=", 1)
            value_path = Path(value).expanduser().resolve()
            resolved.append(f"{option}={value_path}")
            idx += 1
            continue
        if token in path_flags and idx + 1 < len(filtered):
            resolved.append(token)
            resolved.append(str(Path(filtered[idx + 1]).expanduser().resolve()))
            idx += 2
            continue
        resolved.append(token)
        idx += 1

    return [sys.executable, str(Path(__file__).resolve())] + resolved


def write_reproduction_script(
    script_path: Path,
    command_args: Sequence[str],
    workdir: Path,
    target_path: Path,
    existed_before: bool,
    original_content: Optional[bytes],
) -> None:
    """Write a self-contained helper script that can rerun or revert the command."""

    ensure_parent(script_path)
    script_path = script_path.resolve()
    workdir = workdir.resolve()
    target_path = target_path.resolve()

    cmd_tokens = " ".join(shlex.quote(str(arg)) for arg in command_args)
    backup_b64 = (
        base64.b64encode(original_content).decode("ascii")
        if existed_before and original_content is not None
        else ""
    )

    script_body = (
        textwrap.dedent(
            f"""
                #! /usr/bin/env bash
                set -euo pipefail

                SCRIPT_PATH="$(readlink -f "${{BASH_SOURCE[0]}}")"
                WORKDIR={shlex.quote(str(workdir))}
                TARGET_FILE={shlex.quote(str(target_path))}
                CMD=({cmd_tokens})
                EXISTED_BEFORE="{'1' if existed_before else '0'}"
                BACKUP_B64="{backup_b64}"

                decode_backup() {{
                    python3 - "$SCRIPT_PATH" "$TARGET_FILE" <<'PY'
import sys, pathlib, re, base64
script_path = pathlib.Path(sys.argv[1])
target_path = pathlib.Path(sys.argv[2])
text = script_path.read_text()
match = re.search(r'BACKUP_B64="([^"]*)"', text)
if match is None:
    raise SystemExit("Failed to locate BACKUP_B64 in script")
encoded = match.group(1)
if encoded:
    target_path.write_bytes(base64.b64decode(encoded.encode('ascii')))
else:
    target_path.write_bytes(b"")
PY
                }}

                encode_file() {{
                    python3 - "$1" <<'PY'
import base64, pathlib, sys
path = pathlib.Path(sys.argv[1])
print(base64.b64encode(path.read_bytes()).decode("ascii"))
PY
                }}

                ensure_original_state() {{
                    if [[ "$EXISTED_BEFORE" == "1" ]]; then
                        if [[ ! -f "$TARGET_FILE" ]]; then
                            echo "Original file missing; cannot safely apply command." >&2
                            exit 1
                        fi
                        local current_b64
                        current_b64="$(encode_file "$TARGET_FILE")"
                        if [[ "$current_b64" != "$BACKUP_B64" ]]; then
                            echo "Command already executed; run \"$SCRIPT_PATH\" revert first." >&2
                            exit 0
                        fi
                    else
                        if [[ -e "$TARGET_FILE" ]]; then
                            echo "Command already executed; run \"$SCRIPT_PATH\" revert first." >&2
                            exit 0
                        fi
                    fi
                }}

                usage() {{
                    echo "Usage: $0 [run|revert]" >&2
                }}

                mode="${{1:-run}}"

                case "$mode" in
                    run)
                        cd "$WORKDIR"
                        ensure_original_state
                        "${{CMD[@]}}"
                        ;;
                    revert)
                        cd "$WORKDIR"
                        if [[ "$EXISTED_BEFORE" == "1" ]]; then
                            decode_backup
                        else
                            rm -f "$TARGET_FILE"
                        fi
                        ;;
                    *)
                        usage
                        exit 1
                        ;;
                esac
            """
        ).strip()
        + "\n"
    )

    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o111)


def to_triplets(values: Sequence[float]) -> List[np.ndarray]:
    if len(values) % 3 != 0:
        raise ValueError("Coordinate list must contain multiples of three")
    array = np.asarray(values, dtype=float).reshape(-1, 3)
    return [row for row in array]


def transform_point(
    point: np.ndarray,
    model: Any,
    data: Any,
    reference_name: str,
    target_name: str,
    frame: str,
) -> np.ndarray:
    if frame == "world":
        world = np.asarray(point, dtype=float)
    else:
        ref_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, reference_name)  # type: ignore[attr-defined]
        if ref_id < 0:
            raise ValueError(f"Reference body '{reference_name}' not found in model")
        ref_pos = np.asarray(data.xpos[ref_id])
        ref_mat = np.asarray(data.xmat[ref_id]).reshape(3, 3)
        world = ref_pos + ref_mat @ np.asarray(point, dtype=float)

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)  # type: ignore[attr-defined]
    if target_id < 0:
        raise ValueError(f"Target body '{target_name}' not found in model")
    target_pos = np.asarray(data.xpos[target_id])
    target_mat = np.asarray(data.xmat[target_id]).reshape(3, 3)

    local = target_mat.T @ (world - target_pos)
    return local


def parse_site_attributes(pairs: Sequence[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Site attribute '{pair}' must use key=value format")
        key, value = pair.split("=", 1)
        attrs[key.strip()] = value.strip()
    return attrs


def format_float_sequence(values: Sequence[float] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=float).ravel()
    return " ".join(f"{float(v):.9g}" for v in arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append MuJoCo sites based on mesh vertices or manual coordinates."
    )
    parser.add_argument(
        "-m", "--model", required=True, help="Path to the MJCF model file."
    )
    parser.add_argument(
        "-b",
        "--body",
        help="Name of the body that should receive the sites.",
    )
    parser.add_argument(
        "--mesh-file", help="Mesh file containing vertex data (OBJ, STL, ...)."
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scale applied to mesh vertices before processing (scalar or 3-vector).",
    )
    parser.add_argument(
        "--vertices",
        nargs="+",
        help=(
            "Vertex indices (supporting ranges). Required when --mesh-file is provided"
            " and reused for --flex-prefix selections."
        ),
    )
    parser.add_argument(
        "--index-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether vertex indices are 0- or 1-based (applies to mesh vertices only).",
    )
    parser.add_argument(
        "--flex-prefix",
        help=(
            "Prefix of flex body names (e.g. jaw_exo_mask_flexcomp). When provided,"
            " the script creates sites on the matching flex bodies."
        ),
    )
    parser.add_argument(
        "--coords",
        type=float,
        nargs="+",
        help="Manual coordinates (triples). May be combined with mesh vertices.",
    )
    parser.add_argument(
        "--reference-body",
        help="Body whose frame the input coordinates are expressed in (defaults to target body).",
    )
    parser.add_argument(
        "--coord-frame",
        choices=("body", "world"),
        default="body",
        help="Interpret input coordinates relative to the reference body frame or world frame.",
    )
    parser.add_argument(
        "--site-prefix", default="site", help="Prefix for auto-generated site names."
    )
    parser.add_argument(
        "--site-names",
        nargs="+",
        help="Explicit site names; must match the number of provided coordinates.",
    )
    parser.add_argument(
        "--site-size",
        type=float,
        nargs="+",
        default=[0.005],
        help="Site size attribute (scalar replicated to 3 values if needed).",
    )
    parser.add_argument(
        "--site-rgba",
        type=float,
        nargs=4,
        help="Optional RGBA color for the sites.",
    )
    parser.add_argument("--site-group", type=int, help="Optional site group index.")
    parser.add_argument(
        "--site-type", help="Optional MuJoCo site type (sphere, box, etc.)."
    )
    parser.add_argument(
        "--site-class", dest="site_class", help="Optional site class attribute."
    )
    parser.add_argument(
        "--site-attrs",
        nargs="+",
        default=[],
        help="Additional site attributes in key=value form.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination MJCF file. Defaults to <model>_sites.xml unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Modify the input file directly."
    )
    parser.add_argument(
        "--record-script",
        help=(
            "Optional path to a bash helper that replays this command and offers a revert option. "
            "Incompatible with --dry-run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned sites without editing the file.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information."
    )

    args = parser.parse_args()

    if mujoco is None:
        raise RuntimeError(
            "The 'mujoco' Python package is required for this script. Install it with 'pip install mujoco'."
        )

    flex_mode = args.flex_prefix is not None
    if flex_mode:
        if not args.vertices:
            raise ValueError("--vertices is required when using --flex-prefix")
        if args.mesh_file or args.coords:
            raise ValueError(
                "Flex mode cannot be combined with --mesh-file or --coords inputs"
            )
        if args.reference_body:
            raise ValueError(
                "--reference-body is not applicable when using --flex-prefix"
            )
    else:
        if not args.body:
            raise ValueError("--body is required unless --flex-prefix is provided")

    if args.record_script and args.dry_run:
        raise ValueError("--record-script cannot be used together with --dry-run")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.in_place and args.output:
        raise ValueError("--output cannot be combined with --in-place")

    if not flex_mode and args.mesh_file is None and not args.coords:
        raise ValueError("Provide either --mesh-file with --vertices or --coords")

    output_path = (
        model_path
        if args.in_place
        else (
            Path(args.output).expanduser().resolve()
            if args.output
            else model_path.with_name(f"{model_path.stem}_sites{model_path.suffix}")
        )
    )

    record_script_path: Optional[Path] = None
    original_content: Optional[bytes] = None
    target_existed_before = output_path.exists()

    if args.record_script:
        record_script_path = Path(args.record_script).expanduser().resolve()
        ensure_parent(record_script_path)
        if target_existed_before:
            original_content = output_path.read_bytes()

    site_specs: List[dict[str, Any]] = []

    if flex_mode:
        flex_indices = parse_index_sequence(args.vertices)
        if not flex_indices:
            raise ValueError("No flex vertices provided")
        for idx in flex_indices:
            body_name = f"{args.flex_prefix}_{idx}"
            zero = np.zeros(3, dtype=float)
            site_specs.append(
                {
                    "target_body": body_name,
                    "label": f"flex{idx}",
                    "raw": zero,
                    "local": zero,
                }
            )
    else:
        coordinates: List[np.ndarray] = []
        labels: List[str] = []

        if args.mesh_file:
            mesh_path = Path(args.mesh_file).expanduser().resolve()
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            if not args.vertices:
                raise ValueError("--vertices must be provided when using --mesh-file")
            mesh_vertices = apply_scale(load_mesh_vertices(mesh_path), args.mesh_scale)
            indices = parse_index_sequence(args.vertices)
            base = args.index_base
            for idx in indices:
                adjusted = idx - base
                if adjusted < 0 or adjusted >= mesh_vertices.shape[0]:
                    raise IndexError(
                        f"Vertex index {idx} (adjusted {adjusted}) out of range for mesh with {mesh_vertices.shape[0]} vertices"
                    )
                coordinates.append(mesh_vertices[adjusted])
                labels.append(f"v{idx}")

        if args.coords:
            coords_triplets = to_triplets(args.coords)
            start = len(labels)
            for offset, triplet in enumerate(coords_triplets):
                coordinates.append(triplet)
                labels.append(f"c{start + offset}")

        if not coordinates:
            raise ValueError("No coordinates provided to create sites")

        reference_body = args.reference_body or args.body

        model = mujoco.MjModel.from_xml_path(str(model_path))  # type: ignore[arg-type]
        data = mujoco.MjData(model)  # type: ignore[call-arg]
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]

        for label, coord in zip(labels, coordinates):
            local = transform_point(
                point=np.asarray(coord, dtype=float),
                model=model,
                data=data,
                reference_name=reference_body,
                target_name=args.body,
                frame=args.coord_frame,
            )
            site_specs.append(
                {
                    "target_body": args.body,
                    "label": label,
                    "raw": np.asarray(coord, dtype=float),
                    "local": local,
                }
            )

    if not site_specs:
        raise ValueError("No sites to create")

    if args.site_names:
        if len(args.site_names) != len(site_specs):
            raise ValueError("--site-names length must match number of generated sites")
        for spec, name in zip(site_specs, args.site_names):
            spec["name"] = name
    else:
        for spec in site_specs:
            spec["name"] = f"{args.site_prefix}_{spec['label']}"

    size_values = args.site_size
    if len(size_values) not in (1, 3):
        raise ValueError("--site-size must contain 1 or 3 values")
    if len(size_values) == 1:
        size_values = size_values * 3

    site_attrs = parse_site_attributes(args.site_attrs)

    if args.dry_run or args.verbose:
        print(f"Prepared {len(site_specs)} site(s).")
        for spec in site_specs:
            raw = spec["raw"]
            local = spec["local"]
            print(
                f" - {spec['name']} @ {spec['target_body']}: "
                f"input=({raw[0]:.6f}, {raw[1]:.6f}, {raw[2]:.6f}) "
                f"-> body=({local[0]:.6f}, {local[1]:.6f}, {local[2]:.6f})"
            )

    if args.dry_run:
        return

    tree = ET.parse(model_path)
    root = tree.getroot()
    body_map: dict[str, ET.Element] = {}
    for elem in root.findall(".//body"):
        name = elem.get("name")
        if name:
            body_map[name] = elem

    size_str = format_float_sequence(size_values)
    rgba_str = format_float_sequence(args.site_rgba) if args.site_rgba else None

    for spec in site_specs:
        body_element = body_map.get(spec["target_body"])
        if body_element is None:
            raise ValueError(f"Body '{spec['target_body']}' not found in XML")
        site_el = ET.Element("site")
        site_el.set("name", spec["name"])
        site_el.set("pos", format_float_sequence(spec["local"]))
        site_el.set("size", size_str)
        if rgba_str:
            site_el.set("rgba", rgba_str)
        if args.site_group is not None:
            site_el.set("group", str(args.site_group))
        if args.site_type:
            site_el.set("type", args.site_type)
        if args.site_class:
            site_el.set("class", args.site_class)
        for key, value in site_attrs.items():
            site_el.set(key, value)
        body_element.append(site_el)

    ensure_parent(output_path)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    if record_script_path is not None:
        command_args = build_replay_command_args("--record-script")
        write_reproduction_script(
            script_path=record_script_path,
            command_args=command_args,
            workdir=Path.cwd().resolve(),
            target_path=output_path.resolve(),
            existed_before=target_existed_before,
            original_content=original_content,
        )
        print(f"Reproduction script written to {record_script_path}")

    print(f"Added {len(site_specs)} site(s). Output: {output_path}")


if __name__ == "__main__":
    main()
