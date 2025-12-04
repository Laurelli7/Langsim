import os
import time
from omni.isaac.kit import SimulationApp

# 1. Start Simulation App (MUST BE FIRST)
simulation_app = SimulationApp({
    "headless": False,
    "extensions": [
        "isaacsim.ros2.bridge",
        "isaacsim.sensors.camera"
    ]
})

import omni.graph.core as og
from omni.isaac.core.utils.extensions import enable_extension

from .constants import SCENE_DIR
from .core.sim_manager import SimManager


def run_sim(scene_id: str):
    # A. Resolve stage path
    scene_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}.usd")
    if not os.path.exists(scene_path):
        print(f"[SIM] Scene not found: {scene_path}")
        return

    print(f"[SIM] Loading scene: {scene_path}")
    # # Ensure extension is enabled (redundant if added to SimulationApp, but safe)
    enable_extension("isaacsim.ros2.bridge")

    sim = SimManager(scene_path)

    # (Optional) any initialization that needs to happen before stepping
    # e.g., reset robot pose, start ROS bridge, etc.
    sim.world.reset()

    print("[SIM] Starting main stepping loop...")
    try:
        # Standard Isaac pattern
        while simulation_app.is_running():
            sim.step()
            # You usually don’t want extra sleep here; SimManager.step() should
            # handle real-time vs sim-time, but keep if needed.
            # time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("[SIM] Interrupted by user.")
    finally:
        print("[SIM] Shutting down SimulationApp.")
        simulation_app.close()


if __name__ == "__main__":
    # e.g. `python sim_runner.py`
    run_sim("0000")
