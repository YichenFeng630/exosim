#! /usr/bin/env python3
import mujoco
import argparse
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vertices above a cutting plane."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="jaw_skin_model.xml",
        help="Path to the MJCF model file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="pinned_vertices.xml",
        help="Output file for pinned vertices.",
    )
    parser.add_argument(
        "-pn",
        "--plane_normal",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Normal vector of the cutting plane.",
    )
    parser.add_argument(
        "-po",
        "--plane_offset",
        type=float,
        default=-0.01,
        help="Offset of the cutting plane (ax + by + cz + d = 0).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    args = parser.parse_args()

    verbose = args.verbose

    model = mujoco.MjModel.from_xml_path(args.model) # type: ignore
    data = mujoco.MjData(model) # type: ignore
    mujoco.mj_fwdPosition(model, data) # type: ignore

    plane_normal = np.array(args.plane_normal, dtype=np.float64)
    plane_offset = args.plane_offset

    pin_vertices = []

    for vert_index in range(model.nflexvert):
        pos = data.flexvert_xpos[vert_index]

        print(
            f"Vertex {vert_index} at position {pos}: plane test = {plane_normal.dot(pos) + plane_offset}"
        ) if verbose else None

        if plane_normal.dot(pos) + plane_offset > 0:
            pin_vertices.append(vert_index)

    # for comp_index in range(model.nflex):
    #     start = model.flex_vertadr[comp_index]
    #     count = model.flex_vertnum[comp_index]

    #     for v in range(start, start + count):
    #         bodyid = model.flex_vertbodyid[v]
    #         pos = data.xpos[bodyid]

    #         print(f"Vertex {v} on body {bodyid} at position {pos}: plane test = {plane_normal.dot(pos) + plane_offset}") if verbose else None

    #         if plane_normal.dot(pos) + plane_offset > 0:
    #             pin_vertices.append(v)

    pin_list_str = " ".join(str(v) for v in pin_vertices)
    
    # Create metadata about the extraction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "timestamp": timestamp,
        "model_file": args.model,
        "plane_normal": args.plane_normal,
        "plane_offset": args.plane_offset,
        "total_vertices": model.nflexvert,
        "pinned_vertices": len(pin_vertices),
        "plane_equation": f"{plane_normal[0]:.3f}*x + {plane_normal[1]:.3f}*y + {plane_normal[2]:.3f}*z + {plane_offset:.3f} > 0"
    }
    
    print(f'<pin id="{pin_list_str}"/> <!-- {len(pin_vertices)} vertices pinned -->')
    
    if verbose:
        print("\nExtraction metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    with open(args.output, "w") as f:
        # Write XML comment with metadata
        f.write("<!-- Vertex extraction metadata:\n")
        for key, value in metadata.items():
            f.write(f"     {key}: {value}\n")
        f.write("-->\n")
        
        # Write the pin element
        f.write(f'<pin id="{pin_list_str}"/> <!-- {len(pin_vertices)} vertices pinned -->\n')
