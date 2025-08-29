# codex-genesis

This repository contains small code snippets for the Genesis physics simulator.

## Examples
- `examples/drone_wind_sim.py`: Quadrotor (CF2X) under wind + rain, using the upstream Genesis API.

  Modes and usage:
  - Light (default): constant wind + downward rain force (+ optional linear drag)
  - SPH: particle-based rain (SPH Liquid + emitter)

  Commands (PowerShell):
  - `python examples\drone_wind_sim.py`  # light mode
  - `python examples\drone_wind_sim.py --mode sph`  # SPH rain
  - Options:
    - `--steps N` (default 1000)
    - `--wind WX WY WZ` (default `2 0 0`)
    - `--rain-down Fz` (default `0.8`), `--drag C` (default `0.0`)
    - `--sph-size S` (default `0.03`), `--sph-speed V` (default `4.0`)
    - `--sph-pos PX PY PZ` (default `0 0 2.5`), `--sph-interval K` (default `1`)
    - `-v/--vis` to show the viewer, `--quiet` to reduce logging noise

  Notes:
  - Windows console may not render Unicode emojis/box characters; the example sets `theme="dumb"` and supports `--quiet` to reduce banners.
  - SPH mode is more physically/visually detailed but heavier than the light mode. Start with defaults before increasing particle rate.
### Presets
You can quickly try typical scenarios with `-p/--preset`:
- `calm` (light): light wind, gentle rain force, mild drag
- `breezy` (light): medium wind, moderate rain force, moderate drag
- `storm` (light): strong wind, heavy rain force, stronger drag
- `drizzle-sph` (sph): sparse particle rain, slower speed, less frequent emission
- `downpour-sph` (sph): small droplets, fast speed, frequent emission

Examples:
- `python examples\drone_wind_sim.py --preset calm --steps 300 --quiet`
- `python examples\drone_wind_sim.py --preset drizzle-sph --steps 300 --quiet`

### Recording (MP4/GIF)
- Save MP4 while simulating (offscreen camera):
  - `python examples\drone_wind_sim.py --preset breezy --steps 600 --record out.mp4 --fps 60 --res 640 480 --quiet`
  - Camera pose options: `--cam-pos X Y Z`, `--cam-lookat X Y Z`
- Convert MP4 to GIF (Python, moviepy):
  ```python
  from moviepy import VideoFileClip
  VideoFileClip('out.mp4').write_gif('out.gif', fps=30)
  ```

### Multi-camera & Snapshots
- Record two viewpoints simultaneously:
  - `--multi-cam` creates a top view in addition to the front camera.
  - Example: `python examples\drone_wind_sim.py --preset storm --steps 600 --record out.mp4 --multi-cam --quiet`
    - Outputs: `out_cam0.mp4` (front), `out_cam1.mp4` (top)
- Periodic PNG snapshots:
  - `--snap-dir snaps --snap-interval 10 --snap-prefix storm`
  - Saves `snaps/storm_<camIdx>_####.png` every 10 steps for each active camera.

### PLATEAU City Loader
- Load glTF/GLB/OBJ/PLY tiles from a directory and place them in a Genesis scene.
- Auto-centers geometry with a sampled scan so models appear near the origin.
- Supports MP4 recording and periodic PNG snapshots.

Usage:
- `python examples\plateau_city_loader.py --data-dir C:\data\PLATEAU\tiles --max-files 200 --vis --quiet`
- Disable centering: `--center none`
- Manual offset: `--offset 1000 2000 0`
- Record: `--record city.mp4 --fps 60 --res 1280 720`
- Snapshots: `--snap-dir snaps --snap-interval 20`
### Obstacle-Aware Path Planning
- Built-in RRT planner (default fallback) and optional OMPL (if available).
- Spherical obstacles and a planning bounding box can be specified via CLI.

Examples:
- Plan with RRT and a single spherical obstacle, then fly with autopilot:
  - `python examples\drone_wind_sim.py --auto --plan rrt --bounds -1 -1 0 2 2 1.5 --obs-sphere 0.5 0.0 0.5 0.25 --wp 1.2 0 0.5 --steps 1500 --quiet`
- Use OMPL (if installed, e.g., via conda-forge) for planning:
  - `python examples\drone_wind_sim.py --auto --plan ompl --bounds -2 -2 0 2 2 2 --obs-sphere 0.0 0.5 0.5 0.3 --wp 1.5 0 0.5 --steps 1500 --quiet`

Options:
- `--plan {none,rrt,ompl}`: choose planner (fallback is RRT if OMPL not present)
- `--bounds minx miny minz maxx maxy maxz`: planning workspace bounds
- `--obs-sphere x y z r`: add spherical obstacle (repeatable)
- `--plan-time T`: time budget for planning (sec), `--rrt-step s`: RRT step size (m)

Notes:
- Obstacles are visualized in the scene:
  - Spheres: red spheres
  - Boxes (AABB): orange boxes
  - Mesh files: blue AABB boxes (mesh bounds)
- The resulting path is fed to the waypoint autopilot (`--auto`).
