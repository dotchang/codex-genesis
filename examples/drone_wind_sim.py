"""Drone flight under constant wind using Genesis.

This example uses the upstream Genesis API (Scene + Drone morph) to spawn a
Crazyflie CF2X quadrotor and apply a constant wind force at each step.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    # Initialize Genesis; default to CPU backend on Windows
    # Use a simple theme to avoid Unicode box characters on some consoles
    gs.init(backend=gs.cpu, theme="dumb")

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

    # Build the scene for a single environment
    scene.build(n_envs=1)

    # Constant wind force (Newtons), applied at the drone's COM link
    wind_force = torch.tensor([[[2.0, 0.0, 0.0]]], device=gs.device, dtype=gs.tc_float)

    # Lightweight model parameters (unused if mode == 'sph')
    rain_down_force = torch.tensor([[[0.0, 0.0, -0.8]]], device=gs.device, dtype=gs.tc_float)
    drag_coeff = 0.0  # set >0 (e.g., 0.2) to enable simple velocity-proportional drag
    # Use the base link as application point
    com_link = [drone.base_link_idx]

    # Run the simulation
    for step in range(1000):
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
            # Emit SPH rain particles each step (light rate to keep it fast)
            emitter.emit(
                droplet_shape=emit_cfg["droplet_shape"],
                droplet_size=emit_cfg["droplet_size"],
                pos=emit_cfg["pos"],
                direction=emit_cfg["direction"],
                speed=emit_cfg["speed"],
            )

        # Step physics
        scene.step()

        # Read and print state
        pos = drone.get_pos()[0].cpu().numpy()
        vel = drone.get_vel()[0].cpu().numpy()
        print(f"{step:04d}: position={pos}, velocity={vel}")


if __name__ == "__main__":
    main()
