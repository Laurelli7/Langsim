import os

import rclpy
from rclpy.node import Node

from simulation_interfaces.srv import (
    ResetSimulation,
    SetSimulationState,
    LoadWorld,
)
from simulation_interfaces.msg import SimulationState as SimStateMsg


from .constants import SCENE_DIR
from .panner_node import PlannerNode


# ------------------------------------------------------------------------------
# Simulation Control Node
# ------------------------------------------------------------------------------
class SimControlNode(Node):
    def __init__(self):
        super().__init__("sim_control_node")

        self.set_state_client = self.create_client(
            SetSimulationState, "/set_simulation_state"
        )
        self.reset_client = self.create_client(ResetSimulation, "/reset_simulation")
        self.load_world_client = self.create_client(LoadWorld, "/load_world")

        # Wait for services
        for client, name in [
            (self.set_state_client, "/set_simulation_state"),
            (self.reset_client, "/reset_simulation"),
            (self.load_world_client, "/load_world"),
        ]:
            if not client.wait_for_service(timeout_sec=10.0):
                self.get_logger().warn(f"Service {name} not available.")

    def _call_and_wait(self, client, request, timeout=10.0, desc="service"):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            self.get_logger().error(f"{desc} call failed or timed out.")
            return None
        return future.result()

    def set_state(self, state: int):
        req = SetSimulationState.Request()
        req.state.state = state
        self._call_and_wait(self.set_state_client, req, desc="SetSimulationState")
        self.get_logger().info(f"[SIM] Set simulation state to {state}")

    def reset_simulation(self):
        req = ResetSimulation.Request()
        self._call_and_wait(self.reset_client, req, desc="ResetSimulation")
        self.get_logger().info("[SIM] Reset simulation to initial state")

    def load_world(self, uri: str):
        req = LoadWorld.Request()
        req.uri = uri
        self._call_and_wait(self.load_world_client, req, desc="LoadWorld")
        self.get_logger().info(f"[SIM] Loaded world from URI: {uri}")


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def main():
    rclpy.init()

    # Configure your scenes here:
    # e.g. ["0000", "0001", "0002"]
    scene_ids = [f"{i:04d}" for i in range(10, 15)]
    runs_per_scene = 2

    sim_control = SimControlNode()

    try:
        for scene_id in scene_ids:
            # Infer world URI from SCENE_DIR; adjust if your worlds live elsewhere
            world_uri = os.path.join(SCENE_DIR, f"scene_{scene_id}.usd")

            sim_control.get_logger().info(
                f"[SIM] Preparing scene {scene_id} with world URI: {world_uri}"
            )

            # Ensure simulation is stopped, then load the world
            sim_control.set_state(SimStateMsg.STATE_STOPPED)
            sim_control.load_world(world_uri)

            for run_idx in range(runs_per_scene):
                sim_control.get_logger().info(
                    f"[SIM] === Scene {scene_id}, Run {run_idx} ==="
                )

                # Reset environment to initial state
                sim_control.reset_simulation()

                # Make sure simulation is playing so sensors and physics update
                sim_control.set_state(SimStateMsg.STATE_PLAYING)

                # Run a single Planner episode
                node = PlannerNode(scene_id=scene_id, run_idx=run_idx)
                node.run_episode()
                node.destroy_node()

            # Stop simulation after finishing all runs for this scene
            sim_control.set_state(SimStateMsg.STATE_STOPPED)

    finally:
        sim_control.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
