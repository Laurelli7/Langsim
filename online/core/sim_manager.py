import os
import math
import numpy as np

# Isaac Sim Imports (Must happen after SimulationApp starts in main)
import omni.usd
from omni.syntheticdata import helpers
from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.world import World
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles


class SimManager:
    def __init__(self, scene_path):
        open_stage(scene_path)
        self.world = World(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        self.setup_scene()

    def setup_scene(self):
        # Setup Cameras
        self.ego_cam = Camera(
            prim_path="/World/turtlebot4/oakd_link/Camera",
            name="ego_cam",
            resolution=(250, 250),
        )
        self.top_cam = Camera(
            prim_path="/World/Camera", name="top_cam", resolution=(1280, 720)
        )
        self.robot_prim = XFormPrim(prim_path="/World/turtlebot4", name="robot_base")

        self.world.reset()
        self.ego_cam.initialize()
        self.top_cam.initialize()
        self.robot_prim.initialize()

        # Warmup
        for _ in range(25):
            self.world.step(render=True)

    def step(self):
        self.world.step(render=True)

    def get_images(self):
        ego_data = self.ego_cam.get_rgba()
        top_data = self.top_cam.get_rgba()

        ego_img = ego_data[:, :, :3].copy() if ego_data.ndim >= 3 else None
        top_img = top_data[:, :, :3].copy() if top_data.ndim >= 3 else None
        return ego_img, top_img

    def get_robot_state(self):
        pos, quat = self.robot_prim.get_world_pose()
        return pos, quat

    def close(self):
        # We don't close app here, just local cleanup if needed
        pass
