"""Load PLATEAU (3D city) data into a Genesis scene.

This script scans a directory for mesh tiles (GLB/GLTF/OBJ/PLY), optionally
computes a global center offset, and instantiates each tile as a static mesh in
the Genesis scene. Use viewer to inspect or run headless and record frames.

Examples (PowerShell):
  # Minimal (auto-center on sampled files, viewer off)
  python examples/plateau_city_loader.py --data-dir C:\\data\\PLATEAU\\tiles --max-files 200 --quiet

  # With viewer and camera
  python examples/plateau_city_loader.py --data-dir C:\\data\\PLATEAU --vis \
    --cam-pos 200 200 120 --cam-lookat 0 0 20

  # Without centering (use local model coords)
  python examples/plateau_city_loader.py --data-dir ./tiles --center none

  # Snapshots every 20 steps
  python examples/plateau_city_loader.py --data-dir ./tiles --snap-dir snaps --snap-interval 20 --quiet
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh as tm

try:
    import genesis as gs
except ImportError as exc:
    raise SystemExit(
        "This example requires the 'genesis' package (Genesis-Embodied-AI)."
    ) from exc


SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".ply"}


def iter_mesh_files(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    seen = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                # de-dup if patterns overlap
                if p.suffix.lower() in SUPPORTED_EXTS and p not in seen:
                    seen.add(p)
                    yield p


def estimate_global_center(files: Sequence[Path], sample: int = 100) -> np.ndarray:
    centers = []
    for i, f in enumerate(files[:sample]):
        try:
            m = tm.load(f, force="mesh")
            if isinstance(m, tm.Trimesh):
                centers.append(m.centroid)
            elif isinstance(m, tm.Scene):
                bbox = m.bounds
                centers.append(((bbox[0] + bbox[1]) * 0.5).astype(np.float64))
        except Exception:
            continue
    if not centers:
        return np.zeros(3, dtype=np.float64)
    centers_arr = np.vstack(centers)
    return np.median(centers_arr, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing PLATEAU tiles (GLB/GLTF/OBJ/PLY)")
    ap.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["*.glb", "*.gltf", "*.obj", "*.ply"],
        help="Glob patterns to match files",
    )
    ap.add_argument("--max-files", type=int, default=500, help="Limit the number of tiles to load")
    ap.add_argument(
        "--center",
        choices=["scan", "none"],
        default="scan",
        help="scan: estimate global center using sampled files; none: no centering",
    )
    ap.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("OX", "OY", "OZ"),
        help="Manual offset to subtract from all tiles (applied after --center)",
    )
    ap.add_argument("--scale", type=float, default=1.0, help="Uniform scale for all tiles")
    ap.add_argument("--vis", action="store_true", help="Show viewer")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging noise")
    ap.add_argument("--steps", type=int, default=300, help="Simulation steps (for viewing/recording)")
    ap.add_argument("--add-plane", action="store_true", help="Add a ground plane at z=0")

    # Optional recording/snapshots
    ap.add_argument("--record", type=str, help="Save an MP4 video to this path")
    ap.add_argument("--fps", type=int, default=60, help="Recording FPS")
    ap.add_argument("--res", type=int, nargs=2, metavar=("W", "H"), default=(1280, 720), help="Camera resolution")
    ap.add_argument("--cam-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(300.0, 300.0, 150.0))
    ap.add_argument("--cam-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 20.0))
    ap.add_argument("--snap-dir", type=str, help="Directory to save PNG snapshots")
    ap.add_argument("--snap-interval", type=int, default=0, help="Save a PNG every k steps (0 disables)")
    ap.add_argument("--snap-prefix", type=str, default="city", help="Snapshot filename prefix")

    args = ap.parse_args()

    init_kwargs = dict(backend=gs.cpu, theme="dumb")
    if args.quiet:
        init_kwargs["logging_level"] = logging.WARNING
    gs.init(**init_kwargs)

    # Gather mesh files
    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise SystemExit(f"Data directory not found: {data_root}")
    files = list(iter_mesh_files(data_root, args.patterns))
    if not files:
        raise SystemExit("No mesh files found (supported: *.glb, *.gltf, *.obj, *.ply)")
    files = files[: args.max_files]

    # Estimate a global center to keep geometry near origin
    center = np.zeros(3, dtype=np.float64)
    if args.center == "scan":
        center = estimate_global_center(files, sample=min(len(files), 200))
    if args.offset is not None:
        center = center + np.array(args.offset, dtype=np.float64)

    # Build scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=bool(args.vis),
    )

    if args.add_plane:
        scene.add_entity(gs.morphs.Plane())

    # Add tiles as static meshes
    for p in files:
        # Compute per-tile centroid for local placement around global center
        try:
            m = tm.load(p, force="mesh")
            if isinstance(m, tm.Trimesh):
                c = m.centroid
            elif isinstance(m, tm.Scene):
                b = m.bounds
                c = ((b[0] + b[1]) * 0.5).astype(np.float64)
            else:
                c = np.zeros(3)
        except Exception:
            c = np.zeros(3)

        pos = tuple((c - center).astype(float))
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(p),
                scale=args.scale,
                pos=pos,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(),
        )

    # Optional camera for recording/snapshots (must be added before build)
    cam = None
    if args.record or (args.snap_dir and args.snap_interval > 0):
        cam = scene.add_camera(
            res=tuple(map(int, args.res)),
            pos=tuple(map(float, args.cam_pos)),
            lookat=tuple(map(float, args.cam_lookat)),
            fov=50,
            GUI=False,
        )

    scene.build(n_envs=1)
    if cam is not None and args.record:
        cam.start_recording()

    steps = int(args.steps)
    for step in range(steps):
        scene.step()
        if cam is not None:
            rgb_arr, *_ = cam.render(rgb=True)
            if args.snap_dir and args.snap_interval and (step % int(args.snap_interval) == 0):
                os.makedirs(args.snap_dir, exist_ok=True)
                img = rgb_arr if isinstance(rgb_arr, np.ndarray) else rgb_arr.detach().cpu().numpy()
                gs.tools.save_img_arr(img, os.path.join(args.snap_dir, f"{args.snap_prefix}_{step:04d}.png"))

    if cam is not None and args.record:
        cam.stop_recording(save_to_filename=args.record, fps=int(args.fps))


if __name__ == "__main__":
    main()

