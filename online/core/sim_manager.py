import time
import os
import math
import numpy as np

# Isaac Sim imports (AFTER SimulationApp is created in your main script)
import omni.usd
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import SingleXFormPrim as XFormPrim


class SimManager:
    def __init__(self, scene_path: str):
        # Load USD stage
        open_stage(scene_path)
        self.world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")

        # Get stage
        self.stage = omni.usd.get_context().get_stage()
        self.setup_scene()

    def setup_scene(self):
        # --- Cameras ---
        self.ego_cam = Camera(
            prim_path="/World/turtlebot4/oakd_link/Camera",
            name="ego_cam",
            resolution=(250, 250),
        )

        self.top_cam = Camera(
            prim_path="/World/Camera",
            name="top_cam",
            resolution=(1280, 720),
        )

        # --- Robot base prim ---
        self.robot_prim = XFormPrim(
            prim_path="/World/turtlebot4",
            name="robot_base",
        )

        # World reset will initialize scene objects after creation
        self.world.reset()

        # Explicit initialization is still good practice
        self.ego_cam.initialize()
        self.top_cam.initialize()
        self.robot_prim.initialize()

        # Warmup frames to stabilize rendering
        for _ in range(25):
            self.world.step(render=True)

    def step(self):
        self.world.step(render=True)

    def get_images(self):
        # isaacsim.sensors.camera.Camera still supports get_rgba()
        ego_data = self.ego_cam.get_rgba()
        top_data = self.top_cam.get_rgba()

        ego_img = (
            ego_data[:, :, :3].copy()
            if ego_data is not None and ego_data.ndim >= 3
            else None
        )
        top_img = (
            top_data[:, :, :3].copy()
            if top_data is not None and top_data.ndim >= 3
            else None
        )

        return ego_img, top_img

    def get_robot_state(self):
        # SingleXFormPrim keeps a single-prim get_world_pose() API
        pos, quat = self.robot_prim.get_world_pose()
        return pos, quat

    def close(self):
        # Local cleanup if needed; SimulationApp is closed by the caller
        pass
