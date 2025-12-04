import os
import json
import time
import base64
import random
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image as RosImage

from cv_bridge import CvBridge
import cv2
import numpy as np

from PIL import Image as PILImage

from .constants import OUTPUT_DIR, MAX_ROUNDS, SCENE_DIR
from .utils.colors import get_color_name
from .core.agents import PlannerAgent, HumanOracle
from .core.executor import execute_ros2_code


class PlannerNode(Node):
    def __init__(self, scene_id: str):
        super().__init__("planner_node")

        self.scene_id = scene_id
        self.bridge = CvBridge()

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
        target_desc = f"Find the {get_color_name(target_rgb)} cylinder and navigate to right in front of it."

        self.get_logger().info(f"[PLANNER] Target: {target_desc}")

        # -----------------------
        # Initialize agents
        # -----------------------
        self.planner = PlannerAgent(goal_description=target_desc)
        self.human = HumanOracle(self.gt_data, target_cylinder, human_mode=True)

        # -----------------------
        # Internal state
        # -----------------------
        self.round_idx = 0
        self.hint = None
        self.episode_done = False

        self.log_data = {"scene": scene_id, "dialogue": []}

        # -----------------------
        # We do NOT create a subscription.
        # We will manually pull images using wait_for_message()
        # -----------------------
        self.get_logger().info("[PLANNER] Node initialized (no callbacks).")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------
    # Helper: block until a camera image is received
    # ----------------------------------------------------------------------
    def get_latest_ego_image(self, timeout_sec=1.0):
        _, msg = wait_for_message(RosImage, self, "/camera_rgb", time_to_wait=timeout_sec)
        return msg

    # ----------------------------------------------------------------------
    # Encode image → base64 (same as original behavior)
    # ----------------------------------------------------------------------
    def encode_image_to_base64(self, img_rgb: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    # ----------------------------------------------------------------------
    # Main episode logic (equivalent to your original run_episode)
    # ----------------------------------------------------------------------
    def run_episode(self):

        self.get_logger().info("--- Starting Episode ---")

        while rclpy.ok() and not self.episode_done:

            if self.round_idx >= MAX_ROUNDS:
                self.get_logger().info("[PLANNER] Max rounds reached.")
                break

            # -----------------------
            # 1. Retrieve latest camera frame on demand
            # -----------------------
            img_msg = self.get_latest_ego_image(timeout_sec=1.0)

            if img_msg is None:
                self.get_logger().warn("No /camera_rgb image received yet...")
                time.sleep(0.1)
                continue
            
            # Convert ROS → CV2
            ego_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            ego_img_rgb = cv2.cvtColor(ego_img, cv2.COLOR_BGR2RGB)

            # Save debug image
            PILImage.fromarray(ego_img_rgb).save(
                f"{OUTPUT_DIR}/{self.scene_id}_r{self.round_idx}_ego.jpg"
            )

            # -----------------------
            # 2. Encode for LLM
            # -----------------------
            ego_b64 = self.encode_image_to_base64(ego_img_rgb)

            # -----------------------
            # 3. Planner step
            # -----------------------
            self.get_logger().info(
                f"[PLANNER] Round {self.round_idx+1}: thinking..."
            )

            decision = self.planner.think(ego_b64, human_hint=self.hint)

            step_log = {
                "round": self.round_idx,
                "planner_action": decision["action"],
                "planner_content": decision["content"],
            }

            # -----------------------
            # 4. Handle "ask human"
            # -----------------------
            if decision["action"] == "ask":
                question = decision["content"]
                self.get_logger().info(f"[PLANNER] Asking human: {question}")

                # Pass ego image (we don't have top image now)
                top_b64 = ego_b64
                self.hint = self.human.get_hint(top_b64, question, [])
                self.get_logger().info(f"[HUMAN] Hint: {self.hint}")

                step_log["human_hint"] = self.hint

            # -----------------------
            # 5. Handle "code" action
            # -----------------------
            elif decision["action"] == "code":
                self.get_logger().info("[PLANNER] Generated code → executing")

                filename = f"{OUTPUT_DIR}/{self.scene_id}_r{self.round_idx}_code.py"
                result = execute_ros2_code(decision["content"], filename=filename)

                step_log["result"] = result
                self.log_data["dialogue"].append(step_log)

                self.episode_done = True
                break

            self.log_data["dialogue"].append(step_log)
            self.round_idx += 1

        self.finish_episode()

    # ----------------------------------------------------------------------
    # Write logs + shutdown
    # ----------------------------------------------------------------------
    def finish_episode(self):
        log_path = f"{OUTPUT_DIR}/log_{self.scene_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2)

        self.get_logger().info(f"[PLANNER] Episode finished. Saved log to: {log_path}")
        rclpy.shutdown()


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    rclpy.init()
    scene_id = "0000"
    node = PlannerNode(scene_id)

    # Run your episode logic in foreground thread.
    node.run_episode()


if __name__ == "__main__":
    main()
