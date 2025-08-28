"""Drone flight under wind and rain using Genesis.

This example uses the upstream Genesis API (Scene + Drone morph) to spawn a
Crazyflie CF2X quadrotor and simulate environmental effects:
- Light mode: constant wind + downward rain force (+ optional linear drag)
- SPH mode: particle-based rain (SPH Liquid + emitter)

Usage examples:
  python examples/drone_wind_sim.py                         # light mode
  python examples/drone_wind_sim.py --mode sph              # SPH rain
  python examples/drone_wind_sim.py --wind 1 0 0 --steps 300
  python examples/drone_wind_sim.py --drag 0.2 --vis
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
import torch

try:
    import genesis as gs
except ImportError as exc:  # pragma: no cover - library not installed in CI
    raise SystemExit(
        "This example requires the 'genesis' package (Genesis-Embodied-AI)."
    ) from exc


def setup_sph_rain(scene):
    """Configure an SPH-based rain emitter before scene.build().

    Returns the created emitter and a dict of emission parameters.
    """
    # Water-like liquid (tune viscosity/surface tension as desired)
    rain_mat = gs.materials.SPH.Liquid(mu=0.002, gamma=0.005)
    emitter = scene.add_emitter(material=rain_mat, max_particles=15000)

    emit_cfg = dict(
        droplet_shape="circle",  # circle|sphere|square|rectangle
        droplet_size=0.03,        # meters; smaller -> more particles
        pos=(0.0, 0.0, 2.5),      # spawn above the drone
        direction=(0.0, 0.0, -1.0),
        speed=4.0,                # initial particle speed
    )
    return emitter, emit_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["light", "sph"], default="light", help="Rain model: light or SPH")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")

    # General runtime controls
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument(
        "--wind", type=float, nargs=3, metavar=("WX", "WY", "WZ"), default=(2.0, 0.0, 0.0), help="Wind force (N)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging (suppress INFO banners/emojis)")

    # Recording controls (offscreen camera)
    parser.add_argument("--record", type=str, help="Save an MP4 video to this path (e.g., out.mp4)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS (when --record is set)")
    parser.add_argument("--res", type=int, nargs=2, metavar=("W", "H"), default=(640, 480), help="Camera resolution")
    parser.add_argument(
        "--cam-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(3.5, 0.0, 2.5), help="Camera position"
    )
    parser.add_argument(
        "--cam-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.5), help="Camera lookat"
    )

    # Light rain parameters
    parser.add_argument("--rain-down", type=float, default=0.8, help="Downward rain force magnitude (N)")
    parser.add_argument("--drag", type=float, default=0.0, help="Linear drag coefficient (N·s/m), 0 disables")

    # SPH parameters
    parser.add_argument("--sph-size", type=float, default=0.03, help="Droplet size (m); smaller => more particles")
    parser.add_argument("--sph-speed", type=float, default=4.0, help="Initial droplet speed (m/s)")
    parser.add_argument(
        "--sph-pos",
        type=float,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        default=(0.0, 0.0, 2.5),
        help="Emitter position (m)",
    )
    parser.add_argument("--sph-interval", type=int, default=1, help="Emit every k steps (>=1)")

    # Presets (applies after parsing and may override provided defaults)
    parser.add_argument(
        "-p",
        "--preset",
        choices=[
            "calm",
            "breezy",
            "storm",
            "drizzle-sph",
            "downpour-sph",
        ],
        help="Convenience presets for typical conditions",
    )

    args = parser.parse_args()

    # Apply presets by updating parsed args as baseline (flags still can override by re-running)
    if args.preset:
        presets = {
            # Light mode presets
            "calm": {
                "mode": "light",
                "wind": (0.5, 0.0, 0.0),
                "rain_down": 0.2,
                "drag": 0.05,
            },
            "breezy": {
                "mode": "light",
                "wind": (2.0, 0.3, 0.0),
                "rain_down": 0.6,
                "drag": 0.12,
            },
            "storm": {
                "mode": "light",
                "wind": (8.0, 1.0, 0.0),
                "rain_down": 2.0,
                "drag": 0.3,
            },
            # SPH mode presets
            "drizzle-sph": {
                "mode": "sph",
                "wind": (1.0, 0.0, 0.0),
                "sph_size": 0.035,
                "sph_speed": 3.0,
                "sph_interval": 4,
                "sph_pos": (0.0, 0.0, 2.5),
            },
            "downpour-sph": {
                "mode": "sph",
                "wind": (3.0, 0.5, 0.0),
                "sph_size": 0.02,
                "sph_speed": 6.0,
                "sph_interval": 1,
                "sph_pos": (0.0, 0.0, 2.5),
            },
        }
        cfg = presets[args.preset]
        for k, v in cfg.items():
            setattr(args, k, v)
    # Initialize Genesis; default to CPU backend on Windows
    # Use a simple theme to avoid Unicode box characters on some consoles
    init_kwargs = dict(backend=gs.cpu, theme="dumb")
    if args.quiet:
        init_kwargs["logging_level"] = logging.WARNING
    gs.init(**init_kwargs)

    # Create a scene with 10 ms timestep
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=bool(args.vis),
    )

    # Add a quadrotor (Crazyflie CF2X) from built-in assets
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

    # Optional: SPH rain must be added BEFORE build
    emitter = None
    emit_cfg = None
    if args.mode == "sph":
        emitter, emit_cfg = setup_sph_rain(scene)

    # Optional recording camera (must be added BEFORE build)
    cam = None
    if args.record:
        cam = scene.add_camera(
            res=tuple(map(int, args.res)),
            pos=tuple(map(float, args.cam_pos)),
            lookat=tuple(map(float, args.cam_lookat)),
            fov=40,
            GUI=False,
        )

    # Build the scene for a single environment
    scene.build(n_envs=1)
    if cam is not None:
        cam.start_recording()

    # Constant wind force (Newtons), applied at the drone's COM link
    wind_force = torch.tensor([[list(args.wind)]], device=gs.device, dtype=gs.tc_float)

    # Lightweight model parameters (unused if mode == 'sph')
    rain_down_force = torch.tensor([[[0.0, 0.0, -abs(args.rain_down)]]], device=gs.device, dtype=gs.tc_float)
    drag_coeff = float(args.drag)  # set >0 (e.g., 0.2) to enable simple velocity-proportional drag
    # Use the base link as application point
    com_link = [drone.base_link_idx]

    # Run the simulation
    for step in range(int(args.steps)):
        # Apply wind force to the COM link
        scene.sim.rigid_solver.apply_links_external_force(
            force=wind_force, links_idx=com_link, ref="link_com", local=False
        )

        if args.mode == "light":
            # Constant downward "rain" force
            scene.sim.rigid_solver.apply_links_external_force(
                force=rain_down_force, links_idx=com_link, ref="link_com", local=False
            )

            # Optional: simple linear drag proportional to current velocity
            if drag_coeff > 0.0:
                cur_vel = drone.get_vel()  # shape: (B, 3)
                drag_force = (-drag_coeff * cur_vel).unsqueeze(1).contiguous()  # (B,1,3)
                scene.sim.rigid_solver.apply_links_external_force(
                    force=drag_force, links_idx=com_link, ref="link_com", local=False
                )
        else:
            # Emit SPH rain particles every k steps (keep rate modest for performance)
            if step % max(1, int(args.sph_interval)) == 0:
                emitter.emit(
                    droplet_shape=emit_cfg["droplet_shape"],
                    droplet_size=float(args.sph_size),
                    pos=tuple(args.sph_pos),
                    direction=emit_cfg["direction"],
                    speed=float(args.sph_speed),
                )

        # Step physics
        scene.step()

        # Capture a frame if recording
        if cam is not None:
            cam.render(rgb=True, depth=False, segmentation=False, normal=False)

        # Read and print state
        pos = drone.get_pos()[0].cpu().numpy()
        vel = drone.get_vel()[0].cpu().numpy()
        print(f"{step:04d}: position={pos}, velocity={vel}")

    # Finalize recording
    if cam is not None:
        cam.stop_recording(save_to_filename=args.record, fps=int(args.fps))


if __name__ == "__main__":
    main()
