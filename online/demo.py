import os
import sys
import json
import time
import base64
import argparse
import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from .constants import OUTPUT_DIR, MAX_ROUNDS
from .core.agents import PlannerAgent, HumanOracle
from .core.executor import execute_ros2_code

# ------------------------------------------------------------------
# Configuration via Environment Variables
# ------------------------------------------------------------------
TOPIC_CMD_VEL = os.getenv("TOPIC_CMD_VEL", "/cmd_vel")
TOPIC_SCAN = os.getenv("TOPIC_SCAN", "/scan")
TOPIC_CAMERA = os.getenv("TOPIC_CAMERA", "/camera_rgb")
ROBOT_FRONT_ANGLE = float(os.getenv("ROBOT_FRONT_ANGLE", "0.0"))  # in radians


class InteractivePlannerNode(Node):
    def __init__(
        self, goal_description: str, scene_id: str = "real", run_idx: int = 0
    ):
        super().__init__("interactive_planner_node")

        self.scene_id = scene_id
        self.run_idx = run_idx
        self.bridge = CvBridge()
        self.goal_description = goal_description

        # -----------------------
        # Initialize Planner
        # -----------------------
        self.get_logger().info(f"[PLANNER] Goal: {self.goal_description}")
        self.get_logger().info(f"[CONFIG] Camera: {TOPIC_CAMERA}, Cmd: {TOPIC_CMD_VEL}")
        self.planner = PlannerAgent(goal_description=self.goal_description)

        # -----------------------
        # Initialize Human Oracle (Real Mode)
        # -----------------------
        self.human = HumanOracle(gt_data=None, target_cylinder=None, human_mode=True)

        # -----------------------
        # Internal state
        # -----------------------
        self.round_idx = 0
        self.hint = None
        self.episode_done = False

        # Per-episode log data
        self.log_data = {
            "scene": scene_id,
            "run_idx": run_idx,
            "goal": goal_description,
            "config": {
                "topic_cmd": TOPIC_CMD_VEL,
                "topic_cam": TOPIC_CAMERA,
                "front_angle": ROBOT_FRONT_ANGLE,
            },
            "dialogue": [],
            "steps": [],
            "success": False,
            "rounds_taken": 0,
        }

        # Publisher for stopping the robot between rounds
        self.cmd_pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.get_logger().info("[PLANNER] Interactive Node initialized.")

    def get_latest_image(self, topic=TOPIC_CAMERA, timeout_sec=2.0):
        """
        Blocking call to get the latest image from the configured topic.
        """
        try:
            _, msg = wait_for_message(RosImage, self, topic, time_to_wait=timeout_sec)
            return msg
        except Exception:
            return None

    def encode_image_to_base64(self, img_rgb: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def run_episode(self):
        self.get_logger().info("--- Starting Real World Episode ---")
        self.get_logger().info("Press Ctrl+C in terminal to abort if needed.")

        while rclpy.ok() and not self.episode_done:

            if self.round_idx >= MAX_ROUNDS:
                self.get_logger().info("[PLANNER] Max rounds reached.")
                break

            # -----------------------
            # Retrieve Ego Camera Frame
            # -----------------------
            img_msg = self.get_latest_image(topic=TOPIC_CAMERA, timeout_sec=2.0)

            if img_msg is None:
                self.get_logger().warn(
                    f"No image received from {TOPIC_CAMERA}. Retrying..."
                )
                time.sleep(0.5)
                continue

            # Convert ROS → CV2
            try:
                ego_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                ego_img_rgb = cv2.cvtColor(ego_img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")
                continue

            # Save debug image
            ego_path = f"{OUTPUT_DIR}/{self.scene_id}_run{self.run_idx}_r{self.round_idx}_ego.jpg"
            PILImage.fromarray(ego_img_rgb).save(ego_path)

            # Encode to base64
            ego_b64 = self.encode_image_to_base64(ego_img_rgb)

            # -----------------------
            # Planner step
            # -----------------------
            self.get_logger().info(f"[PLANNER] Round {self.round_idx + 1}: Thinking...")

            decision = self.planner.think(ego_b64, human_hint=self.hint)

            step_log = {
                "round": self.round_idx,
                "action": decision["action"],
                "content": decision["content"],
            }
            self.log_data["steps"].append(step_log)

            # -----------------------
            # Execute Decision
            # -----------------------

            if decision["action"] == "ask":
                question = decision["content"]
                self.get_logger().info(f"[PLANNER] Question: {question}")

                # Interactive Input
                self.hint = self.human.get_hint(
                    top_down_b64=None, question=question, robot_pos=None
                )

            elif decision["action"] == "code":
                self.get_logger().info("[PLANNER] Generated code → executing...")

                filename = f"{OUTPUT_DIR}/{self.scene_id}_run{self.run_idx}_r{self.round_idx}_code.py"

                try:
                    execute_ros2_code(decision["content"], filename=filename)
                except Exception as e:
                    self.get_logger().error(f"Execution failed: {e}")
                    self.hint = f"The previous code failed to execute with error: {e}"

                # Safety Stop
                stop_msg = Twist()
                self.cmd_pub.publish(stop_msg)

                self.hint = self.human.get_hint(
                    top_down_b64=None,
                    question="Any further instructions?",
                    robot_pos=None,
                )

            self.round_idx += 1

        self.finish_episode()

    def finish_episode(self):
        self.log_data["dialogue"] = self.planner.history
        self.log_data["rounds_taken"] = self.round_idx

        log_path = f"{OUTPUT_DIR}/log_{self.scene_id}_run{self.run_idx}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2)

        self.get_logger().info(f"[PLANNER] Episode finished. Log saved to {log_path}")


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description="Run the Interactive VLM Planner on a Real Robot"
    )
    parser.add_argument(
        "--goal", type=str, required=True, help="Natural language goal description"
    )
    parser.add_argument(
        "--scene", type=str, default="real", help="Identifier for the environment"
    )
    parser.add_argument("--run", type=int, default=0, help="Run index for logging")

    # Parse known args to allow ROS args to pass through if needed
    parsed_args, _ = parser.parse_known_args()

    node = InteractivePlannerNode(
        goal_description=parsed_args.goal,
        scene_id=parsed_args.scene,
        run_idx=parsed_args.run,
    )

    try:
        node.run_episode()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
