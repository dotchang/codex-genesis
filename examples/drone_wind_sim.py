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
import json
import logging
import math
import os
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

    # Wind field modes
    parser.add_argument(
        "--wind-mode",
        choices=["const", "grid", "noise"],
        default="const",
        help="Wind model: const (uniform), grid (3D lattice), noise (procedural)",
    )
    # Grid parameters
    parser.add_argument("--wind-grid-size", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=(8, 8, 4))
    parser.add_argument("--wind-grid-spacing", type=float, nargs=3, metavar=("SX", "SY", "SZ"), default=(10.0, 10.0, 10.0))
    parser.add_argument("--wind-grid-origin", type=float, nargs=3, metavar=("OX", "OY", "OZ"), default=(-40.0, -40.0, 0.0))
    parser.add_argument("--wind-grid-default", type=float, nargs=3, metavar=("GX", "GY", "GZ"), default=(2.0, 0.0, 0.0))
    parser.add_argument("--wind-grid-file", type=str, help="Path to npz/json defining wind field (shape [NX,NY,NZ,3])")
    parser.add_argument(
        "--wind-grid-interp",
        choices=["nearest", "linear", "idwdir", "idwdir4"],
        default="linear",
    )
    parser.add_argument("--wind-idw-power", type=float, default=2.0, help="IDW power for 'idwdir' interpolation")
    parser.add_argument("--wind-grid-scale", type=float, default=1.0, help="Scale factor applied to loaded/grid vectors")
    # Noise parameters
    parser.add_argument("--wind-noise-amp", type=float, default=2.0, help="Amplitude for noise mode (N)")
    parser.add_argument("--wind-noise-seed", type=int, default=0, help="RNG seed for noise mode")

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
    parser.add_argument("--multi-cam", action="store_true", help="Record from two viewpoints (front and top)")
    parser.add_argument("--snap-dir", type=str, help="Directory to save PNG snapshots (every k steps)")
    parser.add_argument("--snap-interval", type=int, default=0, help="Save a PNG every k steps (0 disables)")
    parser.add_argument("--snap-prefix", type=str, default="frame", help="Snapshot filename prefix")

    # Camera following (keep drone centered)
    parser.add_argument("--follow", action="store_true", help="Move camera to keep the drone centered")
    parser.add_argument("--follow-smooth", type=float, default=0.9, help="EMA smoothing factor [0..1) for follow")
    parser.add_argument(
        "--follow-fixed-z",
        type=float,
        default=None,
        help="Fix camera Z while following (omit to follow in Z as well)",
    )

    # Waypoint autopilot (simple PID-based)
    parser.add_argument("--auto", action="store_true", help="Enable waypoint autopilot")
    parser.add_argument("--wp", type=float, nargs=3, action="append", metavar=("X","Y","Z"), help="Waypoint")
    parser.add_argument("--wp-file", type=str, help="JSON file with [[x,y,z], ...]")
    parser.add_argument("--wp-radius", type=float, default=0.25, help="Waypoint acceptance radius (m)")
    parser.add_argument("--base-rpm", type=float, default=14500.0, help="Base RPM hover around ~14.5k")
    parser.add_argument("--pid-pos", type=float, nargs=3, default=(1.2, 0.0, 0.8), help="PID gains kp ki kd for position loop (applies per-axis)")
    parser.add_argument("--pid-vel", type=float, nargs=3, default=(0.6, 0.0, 0.3), help="PID gains kp ki kd for velocity loop (applies per-axis)")
    parser.add_argument("--pid-att", type=float, nargs=3, default=(0.15, 0.0, 0.05), help="PID gains kp ki kd for attitude (roll/pitch/yaw) loop")

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

    # Optional offscreen cameras (must be added BEFORE build)
    cams = []
    need_camera = bool(args.record) or (args.snap_dir is not None and int(args.snap_interval) > 0)
    if need_camera:
        # Primary/front camera from CLI
        cam_main = scene.add_camera(
            res=tuple(map(int, args.res)),
            pos=tuple(map(float, args.cam_pos)),
            lookat=tuple(map(float, args.cam_lookat)),
            fov=40,
            GUI=False,
        )
        cams.append(cam_main)
        # Optional top-down camera
        if args.multi_cam:
            cam_top = scene.add_camera(
                res=tuple(map(int, args.res)),
                pos=(0.0, 0.0, 5.0),
                lookat=(0.0, 0.0, 0.5),
                fov=50,
                GUI=False,
            )
            cams.append(cam_top)

    # If following is requested, set cameras to follow the drone entity (before build)
    if args.follow and (need_camera):
        fixed_axis = (None, None, args.follow_fixed_z) if args.follow_fixed_z is not None else (None, None, None)
        for c in cams:
            c.follow_entity(drone, fixed_axis=fixed_axis, smoothing=float(args.follow_smooth), fix_orientation=False)

    # Build the scene for a single environment
    scene.build(n_envs=1)
    # Start recording if requested
    if args.record and cams:
        for c in cams:
            c.start_recording()

    # Wind model setup
    wind_const = torch.tensor([[list(args.wind)]], device=gs.device, dtype=gs.tc_float)

    def load_wind_grid_from_file(path: str):
        p = Path(path)
        if p.suffix.lower() == ".npz":
            data = np.load(p)
            if "U" in data:
                arr = np.array(data["U"], dtype=np.float32)
            else:
                # take first array
                key = list(data.keys())[0]
                arr = np.array(data[key], dtype=np.float32)
        else:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            arr = np.array(obj["U"] if isinstance(obj, dict) and "U" in obj else obj, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise SystemExit(f"Invalid wind grid shape: {arr.shape}. Expect [NX,NY,NZ,3].")
        return arr

    def make_wind_grid():
        nx, ny, nz = args.wind_grid_size
        base = np.array(args.wind_grid_default, dtype=np.float32)
        if args.wind_grid_file:
            arr = load_wind_grid_from_file(args.wind_grid_file)
        elif args.wind_mode == "noise":
            rng = np.random.default_rng(args.wind_noise_seed)
            arr = rng.uniform(-1.0, 1.0, size=(nx, ny, nz, 3)).astype(np.float32) * float(args.wind_noise_amp)
            arr += base
        else:
            arr = np.zeros((nx, ny, nz, 3), dtype=np.float32)
            arr[...] = base
        if args.wind_grid_scale != 1.0:
            arr *= float(args.wind_grid_scale)
        return torch.tensor(arr, device=gs.device, dtype=gs.tc_float)

    wind_grid = None
    if args.wind_mode in ("grid", "noise"):
        from pathlib import Path  # ensure Path is available in this scope
        wind_grid = make_wind_grid()
        grid_origin = torch.tensor(list(args.wind_grid_origin), device=gs.device, dtype=gs.tc_float)
        grid_spacing = torch.tensor(list(args.wind_grid_spacing), device=gs.device, dtype=gs.tc_float)

        def sample_wind_at(pos_b3: torch.Tensor, heading_b3: torch.Tensor | None = None) -> torch.Tensor:
            """Sampling wind at world positions (B,3). Returns (B,3).
            Modes: nearest, linear, idwdir (8-node IDW on mag+dir), idwdir4 (2x2 at nearest-Z using heading)."""
            # map to grid coords
            g = (pos_b3 - grid_origin) / grid_spacing.clamp_min(1e-6)
            nx, ny, nz = wind_grid.shape[:3]
            if args.wind_grid_interp == "nearest":
                ix = torch.clamp(torch.round(g[:, 0]).to(torch.int64), 0, nx - 1)
                iy = torch.clamp(torch.round(g[:, 1]).to(torch.int64), 0, ny - 1)
                iz = torch.clamp(torch.round(g[:, 2]).to(torch.int64), 0, nz - 1)
                return wind_grid[ix, iy, iz, :]
            if args.wind_grid_interp == "idwdir":
                # 8 neighbor node indices
                x = torch.clamp(g[:, 0], 0.0, nx - 1.000001)
                y = torch.clamp(g[:, 1], 0.0, ny - 1.000001)
                z = torch.clamp(g[:, 2], 0.0, nz - 1.000001)
                x0 = torch.floor(x).to(torch.int64); x1 = torch.clamp(x0 + 1, 0, nx - 1)
                y0 = torch.floor(y).to(torch.int64); y1 = torch.clamp(y0 + 1, 0, ny - 1)
                z0 = torch.floor(z).to(torch.int64); z1 = torch.clamp(z0 + 1, 0, nz - 1)

                v000 = wind_grid[x0, y0, z0, :]
                v100 = wind_grid[x1, y0, z0, :]
                v010 = wind_grid[x0, y1, z0, :]
                v110 = wind_grid[x1, y1, z0, :]
                v001 = wind_grid[x0, y0, z1, :]
                v101 = wind_grid[x1, y0, z1, :]
                v011 = wind_grid[x0, y1, z1, :]
                v111 = wind_grid[x1, y1, z1, :]
                V = torch.stack([v000, v100, v010, v110, v001, v101, v011, v111], dim=1)  # (B,8,3)

                idxs = torch.stack([
                    torch.stack([x0, y0, z0], dim=1),
                    torch.stack([x1, y0, z0], dim=1),
                    torch.stack([x0, y1, z0], dim=1),
                    torch.stack([x1, y1, z0], dim=1),
                    torch.stack([x0, y0, z1], dim=1),
                    torch.stack([x1, y0, z1], dim=1),
                    torch.stack([x0, y1, z1], dim=1),
                    torch.stack([x1, y1, z1], dim=1),
                ], dim=1)  # (B,8,3)
                node_pos = grid_origin + idxs.to(grid_spacing.dtype) * grid_spacing  # (B,8,3)

                eps = 1e-6
                d = torch.linalg.norm(pos_b3.unsqueeze(1) - node_pos, dim=2).clamp_min(eps)  # (B,8)
                w = (1.0 / (d ** float(args.wind_idw_power))).unsqueeze(2)  # (B,8,1)

                mag = torch.linalg.norm(V, dim=2)  # (B,8)
                dir = V / (mag.unsqueeze(2).clamp_min(eps))  # (B,8,3)
                mag_i = (w.squeeze(2) * mag).sum(dim=1) / w.squeeze(2).sum(dim=1).clamp_min(eps)  # (B,)
                dir_i = (w * dir).sum(dim=1)
                dir_i = dir_i / torch.linalg.norm(dir_i, dim=1, keepdim=True).clamp_min(eps)  # (B,3)
                return dir_i * mag_i.unsqueeze(1)
            if args.wind_grid_interp == "idwdir4":
                # 2x2 neighbors on nearest Z plane, selected relative to heading
                x = torch.clamp(g[:, 0], 0.0, nx - 1.000001)
                y = torch.clamp(g[:, 1], 0.0, ny - 1.000001)
                z = torch.clamp(g[:, 2], 0.0, nz - 1.000001)
                x0 = torch.floor(x).to(torch.int64)
                y0 = torch.floor(y).to(torch.int64)
                z_near = torch.clamp(torch.round(z).to(torch.int64), 0, nz - 1)

                if heading_b3 is None:
                    heading_b3 = torch.zeros_like(pos_b3); heading_b3[:, 0] = 1.0
                hx = heading_b3[:, 0]
                hy = heading_b3[:, 1]
                xi0 = torch.where(hx >= 0, x0, torch.clamp(x0 - 1, 0, nx - 1))
                xi1 = torch.where(hx >= 0, torch.clamp(x0 + 1, 0, nx - 1), x0)
                yi0 = torch.where(hy >= 0, y0, torch.clamp(y0 - 1, 0, ny - 1))
                yi1 = torch.where(hy >= 0, torch.clamp(y0 + 1, 0, ny - 1), y0)

                v00 = wind_grid[xi0, yi0, z_near, :]
                v10 = wind_grid[xi1, yi0, z_near, :]
                v01 = wind_grid[xi0, yi1, z_near, :]
                v11 = wind_grid[xi1, yi1, z_near, :]
                V = torch.stack([v00, v10, v01, v11], dim=1)  # (B,4,3)

                idxs = torch.stack([
                    torch.stack([xi0, yi0, z_near], dim=1),
                    torch.stack([xi1, yi0, z_near], dim=1),
                    torch.stack([xi0, yi1, z_near], dim=1),
                    torch.stack([xi1, yi1, z_near], dim=1),
                ], dim=1)  # (B,4,3)
                node_pos = grid_origin + idxs.to(grid_spacing.dtype) * grid_spacing  # (B,4,3)

                eps = 1e-6
                d = torch.linalg.norm(pos_b3.unsqueeze(1) - node_pos, dim=2).clamp_min(eps)  # (B,4)
                w = (1.0 / (d ** float(args.wind_idw_power))).unsqueeze(2)  # (B,4,1)

                mag = torch.linalg.norm(V, dim=2)  # (B,4)
                dir = V / (mag.unsqueeze(2).clamp_min(eps))  # (B,4,3)
                mag_i = (w.squeeze(2) * mag).sum(dim=1) / w.squeeze(2).sum(dim=1).clamp_min(eps)  # (B,)
                dir_i = (w * dir).sum(dim=1)
                dir_i = dir_i / torch.linalg.norm(dir_i, dim=1, keepdim=True).clamp_min(eps)  # (B,3)
                return dir_i * mag_i.unsqueeze(1)
            # linear
            x = torch.clamp(g[:, 0], 0.0, nx - 1.000001)
            y = torch.clamp(g[:, 1], 0.0, ny - 1.000001)
            z = torch.clamp(g[:, 2], 0.0, nz - 1.000001)
            x0 = torch.floor(x).to(torch.int64)
            y0 = torch.floor(y).to(torch.int64)
            z0 = torch.floor(z).to(torch.int64)
            x1 = torch.clamp(x0 + 1, 0, nx - 1)
            y1 = torch.clamp(y0 + 1, 0, ny - 1)
            z1 = torch.clamp(z0 + 1, 0, nz - 1)
            fx = (x - x0.to(x.dtype)).unsqueeze(1)
            fy = (y - y0.to(y.dtype)).unsqueeze(1)
            fz = (z - z0.to(z.dtype)).unsqueeze(1)
            # gather corners
            c000 = wind_grid[x0, y0, z0, :]
            c100 = wind_grid[x1, y0, z0, :]
            c010 = wind_grid[x0, y1, z0, :]
            c110 = wind_grid[x1, y1, z0, :]
            c001 = wind_grid[x0, y0, z1, :]
            c101 = wind_grid[x1, y0, z1, :]
            c011 = wind_grid[x0, y1, z1, :]
            c111 = wind_grid[x1, y1, z1, :]
            c00 = c000 * (1 - fx) + c100 * fx
            c10 = c010 * (1 - fx) + c110 * fx
            c01 = c001 * (1 - fx) + c101 * fx
            c11 = c011 * (1 - fx) + c111 * fx
            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy
            c = c0 * (1 - fz) + c1 * fz
            return c


    # Lightweight model parameters (unused if mode == 'sph')
    rain_down_force = torch.tensor([[[0.0, 0.0, -abs(args.rain_down)]]], device=gs.device, dtype=gs.tc_float)
    drag_coeff = float(args.drag)  # set >0 (e.g., 0.2) to enable simple velocity-proportional drag
    # Use the base link as application point
    com_link = [drone.base_link_idx]

    # Waypoints autopilot setup
    def _load_waypoints() -> list[torch.Tensor]:
        wps: list[torch.Tensor] = []
        if args.wp_file:
            with open(args.wp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for p in data:
                wps.append(torch.tensor(p, device=gs.device, dtype=gs.tc_float))
        if args.wp:
            for w in args.wp:
                wps.append(torch.tensor(w, device=gs.device, dtype=gs.tc_float))
        return wps

    class _PID:
        def __init__(self, kp, ki, kd):
            self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
            self.i = 0.0
            self.prev = 0.0
        def step(self, err, dt):
            self.i += err * dt
            d = (err - self.prev) / dt
            self.prev = err
            return self.kp * err + self.ki * self.i + self.kd * d

    def _make_ctrl(dt, base_rpm, kpki_kd_pos, kpki_kd_vel, kpki_kd_att):
        return {
            'dt': dt,
            'base': float(base_rpm),
            'pos': [_PID(*kpki_kd_pos), _PID(*kpki_kd_pos), _PID(*kpki_kd_pos)],
            'vel': [_PID(*kpki_kd_vel), _PID(*kpki_kd_vel), _PID(*kpki_kd_vel)],
            'att': [_PID(*kpki_kd_att), _PID(*kpki_kd_att), _PID(*kpki_kd_att)],
        }

    def _ctrl_step(ctrl, drone_entity, target_xyz):
        dt = ctrl['dt']
        pos = drone_entity.get_pos()[0]
        vel = drone_entity.get_vel()[0]
        quat = drone_entity.get_quat()[0]
        att = gs.utils.geom.quat_to_xyz(quat.unsqueeze(0), rpy=True, degrees=True)[0]
        err_pos = target_xyz - pos
        vel_des = torch.tensor([
            ctrl['pos'][0].step(float(err_pos[0]), dt),
            ctrl['pos'][1].step(float(err_pos[1]), dt),
            ctrl['pos'][2].step(float(err_pos[2]), dt),
        ], device=gs.device, dtype=gs.tc_float)
        err_vel = vel_des - vel
        vel_del = torch.tensor([
            ctrl['vel'][0].step(float(err_vel[0]), dt),
            ctrl['vel'][1].step(float(err_vel[1]), dt),
            ctrl['vel'][2].step(float(err_vel[2]), dt),
        ], device=gs.device, dtype=gs.tc_float)
        att_del = torch.tensor([
            ctrl['att'][0].step(float(-att[0]), dt),
            ctrl['att'][1].step(float(-att[1]), dt),
            ctrl['att'][2].step(float(-att[2]), dt),
        ], device=gs.device, dtype=gs.tc_float)
        thrust = vel_del[2]
        roll = att_del[0]
        pitch = att_del[1]
        yaw = att_del[2]
        x_vel = vel_del[0]
        y_vel = vel_del[1]
        base = ctrl['base']
        m1 = base + (thrust - roll - pitch - yaw - x_vel + y_vel)
        m2 = base + (thrust - roll + pitch + yaw + x_vel + y_vel)
        m3 = base + (thrust + roll + pitch - yaw + x_vel - y_vel)
        m4 = base + (thrust + roll - pitch + yaw - x_vel - y_vel)
        rpms = torch.stack([m1, m2, m3, m4])
        return torch.clamp(rpms, 0.0, 22000.0)

    _waypoints = _load_waypoints()
    _wp_idx = 0
    _wp_tol = float(args.wp_radius)
    _ctrl = _make_ctrl(
        dt=0.01,
        base_rpm=float(args.base_rpm),
        kpki_kd_pos=tuple(args.pid_pos),
        kpki_kd_vel=tuple(args.pid_vel),
        kpki_kd_att=tuple(args.pid_att),
    ) if args.auto and _waypoints else None

    # Run the simulation
    last_heading = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
    last_heading[0, 0] = 1.0
    for step in range(int(args.steps)):
        # Wind force application (const/grid/noise)
        if args.wind_mode == "const":
            wf = wind_const
        else:
            # sample at current drone COM (use entity position as proxy)
            pos_b3 = drone.get_pos()  # (B,3)
            wf_b3 = None
            if args.wind_grid_interp == "idwdir4":
                vel_b3 = drone.get_vel()
                spd = torch.linalg.norm(vel_b3, dim=1, keepdim=True)
                heading_b3 = torch.where(
                    spd > 1e-6,
                    vel_b3 / spd,
                    last_heading.expand_as(vel_b3),
                )
                last_heading = heading_b3.detach()
                wf_b3 = sample_wind_at(pos_b3, heading_b3)
            else:
                wf_b3 = sample_wind_at(pos_b3)
            wf = wf_b3.unsqueeze(1).contiguous()
        scene.sim.rigid_solver.apply_links_external_force(force=wf, links_idx=com_link, ref="link_com", local=False)

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

        # Simple waypoint controller
        if _ctrl is not None and _wp_idx < len(_waypoints):
            tgt = _waypoints[_wp_idx]
            pos_now = drone.get_pos()[0]
            if torch.linalg.norm(tgt - pos_now) < _wp_tol:
                _wp_idx += 1
            else:
                rpms = _ctrl_step(_ctrl, drone, tgt)
                # expect (B,N) -> make (1,4)
                drone.set_propellels_rpm(rpms.unsqueeze(0))

        # Step physics
        scene.step()

        # Capture a frame for recording/snapshots
        if cams:
            for idx, c in enumerate(cams):
                rgb_arr, *_ = c.render(rgb=True, depth=False, segmentation=False, normal=False)
                # Save snapshots if enabled and interval reached
                if args.snap_dir and args.snap_interval and (step % int(args.snap_interval) == 0):
                    out_dir = args.snap_dir
                    os.makedirs(out_dir, exist_ok=True)
                    img = rgb_arr if isinstance(rgb_arr, np.ndarray) else rgb_arr.detach().cpu().numpy()
                    # Convert from (H,W,3) float [0,1] or uint8 as returned by renderer
                    gs.tools.save_img_arr(img, os.path.join(out_dir, f"{args.snap_prefix}_{idx}_{step:04d}.png"))

        # Read and print state
        pos = drone.get_pos()[0].cpu().numpy()
        vel = drone.get_vel()[0].cpu().numpy()
        print(f"{step:04d}: position={pos}, velocity={vel}")

    # Finalize recording (write one MP4 per camera)
    if args.record and cams:
        base = args.record
        root, ext = os.path.splitext(base)
        for idx, c in enumerate(cams):
            filename = base if len(cams) == 1 else f"{root}_cam{idx}{ext or '.mp4'}"
            c.stop_recording(save_to_filename=filename, fps=int(args.fps))


if __name__ == "__main__":
    main()
