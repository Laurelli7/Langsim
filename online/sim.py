import os
import time
from isaacsim.simulation_app import SimulationApp

# Start Simulation App
simulation_app = SimulationApp(
    {
        "headless": False,
    },
)

simulation_app.update()

from .constants import SCENE_DIR
from .core.sim_manager import SimManager


def run_sim(scene_id: str):
    scene_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}.usd")
    if not os.path.exists(scene_path):
        print(f"[SIM] Scene not found: {scene_path}")
        return

    print(f"[SIM] Loading scene: {scene_path}")

    sim = SimManager(scene_path)

    print("[SIM] Starting main stepping loop...")
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        run_sim("0000")
    except KeyboardInterrupt:
        print("[SIM] Interrupted by user.")
    finally:
        print("[SIM] Shutting down SimulationApp.")
        simulation_app.close()
