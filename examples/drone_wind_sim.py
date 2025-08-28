"""Drone flight simulation under wind using Genesis.

This example demonstrates how to extend a basic Genesis quadrotor example by
applying a constant wind force. The script will create a simulation, spawn a
simple quadrotor model, and step the physics while applying wind at each step.

The code relies on the external ``genesis`` package. Install it via
``pip install genesis-sim`` before running this example.
"""

from __future__ import annotations

import numpy as np

try:
    import genesis as gs
except ImportError as exc:  # pragma: no cover - library not installed in CI
    raise SystemExit(
        "This example requires the 'genesis' package. Install it before running."
    ) from exc


def main() -> None:
    """Run the wind‑affected drone simulation."""

    # Create the Genesis simulation world with a 10 ms timestep
    sim = gs.Simulation(dt=0.01)

    # Spawn a basic quadrotor from the standard assets
    drone = sim.create_quadrotor()

    # Define a constant wind force vector (Newtons)
    wind_force = np.array([2.0, 0.0, 0.0])

    # Run the simulation for 1000 steps
    for step in range(1000):
        # Apply the wind force at the drone's center of mass
        drone.apply_external_force(wind_force)

        # Advance the physics by one step
        sim.step()

        # Query and print the drone's current state
        pos = drone.position()
        vel = drone.velocity()
        print(f"{step:04d}: position={pos}, velocity={vel}")


if __name__ == "__main__":
    main()
