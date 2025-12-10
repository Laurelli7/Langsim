import math
import os
import json
import time
import base64
import random

import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message

from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist
from simulation_interfaces.srv import GetEntityState

from cv_bridge import CvBridge
import cv2
import numpy as np

from PIL import Image as PILImage

from .constants import OUTPUT_DIR, MAX_ROUNDS, SCENE_DIR
from .utils.colors import get_color_name
from .core.agents import PlannerAgent, HumanOracle
from .core.executor import execute_ros2_code


class PlannerNode(Node):
    def __init__(self, scene_id: str, run_idx: int = 0):
        super().__init__("planner_node")

        self.scene_id = scene_id
        self.run_idx = run_idx
        self.bridge = CvBridge()

        # -----------------------
        # Initialize Simulation Control Client for pose
        # -----------------------
        self.sim_client = self.create_client(GetEntityState, "/get_entity_state")
        if not self.sim_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/get_entity_state service not available yet.")

        # -----------------------
        # Load GT and pick target
        # -----------------------
        gt_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}_gt.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT not found: {gt_path}")

        with open(gt_path, "r", encoding="utf-8") as f:
            self.gt_data = json.load(f)

        target_cylinder = random.choice(self.gt_data["cylinders"])
        target_rgb = target_cylinder.get("color", [1.0, 1.0, 1.0])
        target_desc = (
            f"Find the {get_color_name(target_rgb)} cylinder and move close to it."
        )

        self.target_cylinder_data = target_cylinder
        self.target_pos = self.target_cylinder_data["position"]  # [x, y, z]

        self.get_logger().info(f"[PLANNER] Target: {target_desc}")

        # -----------------------
        # Initialize agents
        # -----------------------
        self.planner = PlannerAgent(goal_description=target_desc)
        self.human = HumanOracle(self.gt_data, target_cylinder, human_mode=False)

        # -----------------------
        # Internal state
        # -----------------------
        self.round_idx = 0
        self.hint = None
        self.episode_done = False

        # Define success threshold (in meters)
        self.success_threshold = 0.8

        # Per-episode log data
        self.log_data = {
            "scene": scene_id,
            "run_idx": run_idx,
            "dialogue": [],
            "steps": [],
            "success": False,
            "final_distance": None,
            "rounds_taken": 0,
        }

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info(
            f"[PLANNER] Node initialized for scene {scene_id}, run {run_idx}."
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------
    # Helper: block until a camera image is received
    # ----------------------------------------------------------------------
    def get_latest_image(self, topic="/camera_rgb", timeout_sec=1.0):
        _, msg = wait_for_message(RosImage, self, topic, time_to_wait=timeout_sec)
        return msg

    # ----------------------------------------------------------------------
    # Encode image → base64
    # ----------------------------------------------------------------------
    def encode_image_to_base64(self, img_rgb: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def fetch_robot_pose(self, entity_name="/World/turtlebot4/base_link"):
        """
        Calls the /get_entity_state service provided by Isaac Sim.
        Returns [x, y, z] or None if failed.
        """
        if not self.sim_client.service_is_ready():
            self.get_logger().warn("Sim service not ready.")
            return None

        request = GetEntityState.Request()
        request.entity = entity_name

        future = self.sim_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        try:
            response = future.result()
            if response:
                pos = response.state.pose.position
                return [round(pos.x, 2), round(pos.y, 2), round(pos.z, 2)]
            else:
                self.get_logger().warn(f"Failed to get state for {entity_name}")
                return None
        except Exception as e:
            self.get_logger().error(f"Service call exception: {e}")
            return None

    # ----------------------------------------------------------------------
    # Main episode logic (single run)
    # ----------------------------------------------------------------------
    def run_episode(self):

        self.get_logger().info("--- Starting Episode ---")

        while rclpy.ok() and not self.episode_done:

            robot_pos = self.fetch_robot_pose(entity_name="/World/turtlebot4/base_link")

            current_dist = float("inf")
            if robot_pos is not None and robot_pos != "Unknown":
                dx = robot_pos[0] - self.target_pos[0]
                dy = robot_pos[1] - self.target_pos[1]
                current_dist = math.sqrt(dx * dx + dy * dy)

                self.get_logger().info(
                    f"[METRICS] Target Pos: {self.target_pos[:2]}, "
                    f"Robot Pos: {robot_pos[:2]}, Dist: {current_dist:.2f}"
                )

                # Termination condition
                if current_dist < self.success_threshold:
                    self.get_logger().info(
                        f"[SUCCESS] Robot is close enough ({current_dist:.2f}m). Stopping."
                    )
                    self.episode_done = True
                    self.log_data["success"] = True
                    self.log_data["final_distance"] = current_dist
                    break

            # Stop if max rounds reached
            if self.round_idx >= MAX_ROUNDS:
                self.get_logger().info("[PLANNER] Max rounds reached.")
                self.log_data["final_distance"] = (
                    current_dist if current_dist != float("inf") else None
                )
                break

            # -----------------------
            # Retrieve latest camera frames
            # -----------------------
            img_msg = self.get_latest_image(topic="/camera_rgb", timeout_sec=1.0)
            top_msg = self.get_latest_image(topic="/camera_top_rgb", timeout_sec=1.0)

            if img_msg is None or top_msg is None:
                self.get_logger().warn(
                    "No image received from /camera_rgb or /camera_top_rgb..."
                )
                time.sleep(0.1)
                continue

            # Convert ROS → CV2
            ego_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            ego_img_rgb = cv2.cvtColor(ego_img, cv2.COLOR_BGR2RGB)
            top_img = self.bridge.imgmsg_to_cv2(top_msg, desired_encoding="bgr8")
            top_img_rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)

            # Save debug images
            ego_path = f"{OUTPUT_DIR}/{self.scene_id}_run{self.run_idx}_r{self.round_idx}_ego.jpg"
            top_path = f"{OUTPUT_DIR}/{self.scene_id}_run{self.run_idx}_r{self.round_idx}_top.jpg"
            PILImage.fromarray(ego_img_rgb).save(ego_path)
            PILImage.fromarray(top_img_rgb).save(top_path)

            # Encode to base64
            ego_b64 = self.encode_image_to_base64(ego_img_rgb)
            top_b64 = self.encode_image_to_base64(top_img_rgb)

            # -----------------------
            # Planner step
            # -----------------------
            self.get_logger().info(f"[PLANNER] Round {self.round_idx + 1}: thinking...")

            decision = self.planner.think(ego_b64, human_hint=self.hint)

            step_log = {
                "round": self.round_idx,
                "robot_pos": robot_pos,
                "distance_to_target": (
                    current_dist if current_dist != float("inf") else None
                ),
            }
            self.log_data["steps"].append(step_log)

            # -----------------------
            # Handle "ask human"
            # -----------------------
            if decision["action"] == "ask":
                question = decision["content"]
                self.get_logger().info(f"[PLANNER] Asking human: {question}")

                self.hint = self.human.get_hint(top_b64, question, robot_pos)
                self.get_logger().info(f"[HUMAN] Hint: {self.hint}")

            # -----------------------
            # Handle "code" action
            # -----------------------
            elif decision["action"] == "code":
                self.get_logger().info("[PLANNER] Generated code → executing")
                filename = f"{OUTPUT_DIR}/{self.scene_id}_run{self.run_idx}_r{self.round_idx}_code.py"
                execute_ros2_code(decision["content"], filename=filename)
                # Reset robot commands after execution
                stop_msg = Twist()
                self.cmd_pub.publish(stop_msg)
                self.hint = (
                    "We are still working towards the goal. "
                    "Please ask for more information if needed."
                )

            self.round_idx += 1

        self.finish_episode()

    # ----------------------------------------------------------------------
    # Write logs (no shutdown here)
    # ----------------------------------------------------------------------
    def finish_episode(self):
        self.log_data["dialogue"] = self.planner.history
        self.log_data["rounds_taken"] = self.round_idx

        log_path = f"{OUTPUT_DIR}/log_{self.scene_id}_run{self.run_idx}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2)

        self.get_logger().info(f"[PLANNER] Episode finished. Saved log to: {log_path}")
