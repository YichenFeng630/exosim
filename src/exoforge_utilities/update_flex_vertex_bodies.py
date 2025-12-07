#! /usr/bin/env python3
"""Utility to remap flex vertices to different bodies while preserving world positions."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass
class VertexAssignment:
    """Container describing the reassignment of a single vertex."""

    vertex: int
    new_body: str
    offset: np.ndarray


@dataclass
class VertexUpdateResult:
    """Summary of the changes applied to a vertex."""

    vertex: int
    old_body: str
    new_body: str
    old_local: np.ndarray
    new_local: np.ndarray
    world_position: np.ndarray
    offset: np.ndarray


def parse_vertex_sequence(spec: Sequence[str]) -> List[int]:
    """Expand a sequence of vertex index specifications.

    Supports individual integers, comma separated values, and closed ranges
    expressed as ``start-end``. For example ``["5", "7-10", "11,13"]``
    resolves to ``[5, 7, 8, 9, 10, 11, 13]``.
    """

    indices: List[int] = []

    def _add_value(token: str) -> None:
        token = token.strip()
        if not token:
            return
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{token}': end < start")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))

    for item in spec:
        for token in item.replace(",", " ").split():
            _add_value(token)

    # Preserve input ordering but drop duplicates while keeping last occurrence
    deduped = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def load_assignments_from_mapping(path: Path) -> List[VertexAssignment]:
    """Load vertex assignments from a JSON mapping file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    assignments: List[VertexAssignment] = []

    if not isinstance(raw, list):
        raise ValueError("Mapping file must contain a list of assignment objects")

    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each mapping entry must be an object")
        if "vertices" not in entry or "new_body" not in entry:
            raise ValueError("Each mapping entry requires 'vertices' and 'new_body'")

        raw_vertices = entry["vertices"]
        if isinstance(raw_vertices, str):
            vertex_ids = parse_vertex_sequence([raw_vertices])
        elif isinstance(raw_vertices, Iterable):
            parts: List[str] = []
            for item in raw_vertices:
                if isinstance(item, (int, np.integer)):
                    parts.append(str(int(item)))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    raise ValueError(
                        "Vertices must be integers, strings, or sequences of those types"
                    )
            vertex_ids = parse_vertex_sequence(parts)
        else:
            raise ValueError("'vertices' must be a string, list, or tuple")

        new_body = str(entry["new_body"])
        raw_offset = entry.get("offset", [0.0, 0.0, 0.0])
        offset = np.asarray(raw_offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError("Offset must be a sequence of three numeric values")

        for vertex in vertex_ids:
            assignments.append(
                VertexAssignment(vertex=vertex, new_body=new_body, offset=offset)
            )

    return assignments


def build_assignments(args: argparse.Namespace) -> List[VertexAssignment]:
    """Construct the list of vertex assignments from CLI arguments."""

    if args.mapping:
        return load_assignments_from_mapping(Path(args.mapping))

    if not args.new_body:
        raise ValueError("--new-body is required when --mapping is not provided")

    vertex_ids = parse_vertex_sequence(args.vertices)
    offset = np.asarray(args.offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError("--offset must include exactly three floats")

    return [
        VertexAssignment(vertex=vertex, new_body=args.new_body, offset=offset)
        for vertex in vertex_ids
    ]


def format_float_sequence(values: Sequence[float]) -> str:
    """Format a sequence of floats as a compact whitespace separated string."""

    return " ".join(f"{value:.9g}" for value in values)


def format_string_sequence(values: Sequence[str]) -> str:
    """Format a sequence of strings as a whitespace separated string."""

    return " ".join(values)


def ensure_parent_directory(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def build_replay_command_args() -> List[str]:
    """Return command arguments that reproduce the current invocation.

    The generated command uses the current Python interpreter and script path,
    filtering out the ``--record-script`` flag so that replays do not attempt to
    recreate the script recursively.
    """

    filtered: List[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--record-script":
            skip_next = True
            continue
        if token.startswith("--record-script="):
            continue
        filtered.append(token)

    path_flags = {"--model", "--mapping", "--output"}

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
    """Persist a self-contained bash helper for rerun/revert without extra files."""

    ensure_parent_directory(script_path)
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


def to_float_tuple(
    values: Sequence[float] | np.ndarray, expected: int = 3
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected:
        raise ValueError(
            f"Expected sequence of length {expected}, received {array.size} entries"
        )
    return tuple(float(v) for v in array)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remap selected flex vertices to different bodies and update their local coordinates. "
            "The script preserves the world-space position of each vertex by recomputing the "
            "coordinates in the destination body's local frame."
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the MJCF file that defines the flex component.",
    )
    parser.add_argument(
        "-f",
        "--flex-name",
        required=True,
        help="Name of the <flex> component to edit.",
    )
    mapping_group = parser.add_mutually_exclusive_group(required=True)
    mapping_group.add_argument(
        "--mapping",
        help=(
            "Path to a JSON file describing vertex remapping instructions. Each entry should "
            "provide 'vertices', 'new_body', and optionally 'offset'."
        ),
    )
    mapping_group.add_argument(
        "--vertices",
        nargs="+",
        help="Vertex indices or ranges (e.g. 12 18-27 42,45). Requires --new-body.",
    )
    parser.add_argument(
        "--new-body",
        help="Destination body name for --vertices assignments.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Optional Cartesian offset applied in the destination body frame.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the modified MJCF. Defaults to <model>_updated.xml unless --in-place is given.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input model in place. A .bak backup is created automatically.",
    )
    parser.add_argument(
        "--record-script",
        help=(
            "Optional path to a bash script that reruns this command and provides a"
            " revert workflow. The script is created after applying changes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all computations and print a summary without writing any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each updated vertex.",
    )

    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.in_place and args.output:
        raise ValueError("--output cannot be combined with --in-place")

    if not args.in_place:
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        else:
            output_path = model_path.with_name(
                f"{model_path.stem}_updated{model_path.suffix}"
            )
    else:
        output_path = model_path

    record_script_path: Optional[Path] = None
    original_content: Optional[bytes] = None
    target_existed_before = output_path.exists()

    if args.record_script:
        if args.dry_run:
            raise ValueError("--record-script cannot be used with --dry-run")
        record_script_path = Path(args.record_script).expanduser().resolve()
        ensure_parent_directory(record_script_path)
        if target_existed_before:
            original_content = output_path.read_bytes()

    assignments = build_assignments(args)
    if not assignments:
        raise ValueError("No vertices were specified for reassignment")

    # Load model and simulate to populate pose information.
    model = mujoco.MjModel.from_xml_path(str(model_path))  # type: ignore[arg-type]
    data = mujoco.MjData(model)  # type: ignore[call-arg]
    mujoco.mj_forward(model, data)  # type: ignore[attr-defined]

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, args.flex_name)  # type: ignore[attr-defined]
    if flex_id < 0:
        raise ValueError(f"Flex component '{args.flex_name}' not found in model")

    flex_start = model.flex_vertadr[flex_id]
    flex_count = model.flex_vertnum[flex_id]
    flex_range = range(flex_start, flex_start + flex_count)

    # Build quick lookup for XML manipulation.
    tree = ET.parse(model_path)
    root = tree.getroot()
    flex_element = None
    for candidate in root.findall(".//flex"):
        if candidate.get("name") == args.flex_name:
            flex_element = candidate
            break
    if flex_element is None:
        raise ValueError(f"Flex element '{args.flex_name}' not found in XML")

    body_attr = flex_element.get("body")
    vertex_attr = flex_element.get("vertex")
    if body_attr is None or vertex_attr is None:
        raise ValueError(
            "Target flex element must define both 'body' and 'vertex' attributes"
        )

    body_names = body_attr.split()
    if len(body_names) != flex_count:
        raise ValueError(
            f"Mismatch between XML body list ({len(body_names)}) and model data ({flex_count})"
        )

    vertex_values = list(map(float, vertex_attr.split()))
    if len(vertex_values) != flex_count * 3:
        raise ValueError(
            f"Vertex coordinate length mismatch: expected {flex_count * 3}, got {len(vertex_values)}"
        )

    updates: List[VertexUpdateResult] = []

    for assignment in assignments:
        vertex_idx = assignment.vertex
        if vertex_idx not in flex_range:
            raise ValueError(
                f"Vertex {vertex_idx} does not belong to flex '{args.flex_name}' (valid range: {flex_start}-{flex_start + flex_count - 1})"
            )

        local_idx = vertex_idx - flex_start

        new_body_name = assignment.new_body
        new_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, new_body_name)  # type: ignore[attr-defined]
        if new_body_id < 0:
            raise ValueError(f"Destination body '{new_body_name}' not found in model")

        old_body_id = model.flex_vertbodyid[vertex_idx]
        old_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, old_body_id)  # type: ignore[attr-defined]

        world_pos = np.asarray(data.flexvert_xpos[vertex_idx])

        new_body_pos = np.asarray(data.xpos[new_body_id])
        new_body_mat = np.asarray(data.xmat[new_body_id]).reshape(3, 3)

        relative = world_pos - new_body_pos
        local_coords = new_body_mat.T @ relative
        local_coords += assignment.offset

        old_local = np.asarray(model.flex_vert[vertex_idx], dtype=float)

        body_names[local_idx] = new_body_name
        vertex_values[local_idx * 3 : local_idx * 3 + 3] = list(
            map(float, local_coords)
        )

        updates.append(
            VertexUpdateResult(
                vertex=vertex_idx,
                old_body=old_body_name,
                new_body=new_body_name,
                old_local=old_local,
                new_local=local_coords,
                world_position=world_pos,
                offset=assignment.offset,
            )
        )

    if args.dry_run or args.verbose:
        print(
            f"Prepared {len(updates)} vertex reassignment(s) on flex '{args.flex_name}'."
        )
        for update in updates:
            wx, wy, wz = to_float_tuple(update.world_position)
            ox, oy, oz = to_float_tuple(update.old_local)
            nx, ny, nz = to_float_tuple(update.new_local)
            dx, dy, dz = to_float_tuple(update.offset)
            print(
                " - Vertex {vertex}: {old_body} -> {new_body} | "
                "world=({wx:.6f}, {wy:.6f}, {wz:.6f}) | "
                "local_old=({ox:.6f}, {oy:.6f}, {oz:.6f}) | "
                "local_new=({nx:.6f}, {ny:.6f}, {nz:.6f}) | "
                "offset=({dx:.6f}, {dy:.6f}, {dz:.6f})".format(
                    vertex=update.vertex,
                    old_body=update.old_body,
                    new_body=update.new_body,
                    wx=wx,
                    wy=wy,
                    wz=wz,
                    ox=ox,
                    oy=oy,
                    oz=oz,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                )
            )

    if args.dry_run:
        print("Dry-run requested: no files were written.")
        return

    if args.in_place:
        backup_path = model_path.with_suffix(model_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_bytes(model_path.read_bytes())
            print(f"Backup written to {backup_path}")
        else:
            print(f"Backup already exists at {backup_path}; skipping backup creation.")

    flex_element.set("body", format_string_sequence(body_names))
    flex_element.set("vertex", format_float_sequence(vertex_values))

    ensure_parent_directory(output_path)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    if record_script_path is not None:
        command_args = build_replay_command_args()
        write_reproduction_script(
            script_path=record_script_path,
            command_args=command_args,
            workdir=Path.cwd().resolve(),
            target_path=output_path.resolve(),
            existed_before=target_existed_before,
            original_content=original_content,
        )
        print(f"Reproduction script written to {record_script_path}")

    print(f"Updated model written to {output_path}")


if __name__ == "__main__":
    main()
