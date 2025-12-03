import os
import json
import time
import base64
import subprocess
import numpy as np
import random
import io
import re
import webcolors
from PIL import Image

# Isaac Sim Core Imports
from omni.isaac.kit import SimulationApp

# START SIMULATION APP
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.world import World
from omni.isaac.sensor import Camera
from openai import OpenAI


# --- CONFIGURATION (ENV VARS) ---
# These are used in the Python script AND passed to the LLM prompt
ENV_CONFIG = {
    "ROBOT_FRONT_ANGLE": os.getenv(
        "ROBOT_FRONT_ANGLE", "0.0"
    ),  # Radians adjustment if camera isn't perfectly front
    "TOPIC_CMD_VEL": os.getenv("TOPIC_CMD_VEL", "/cmd_vel"),
    "TOPIC_SCAN": os.getenv("TOPIC_SCAN", "/scan"),
    "TOPIC_CAMERA_RGB": os.getenv("TOPIC_CAMERA_RGB", "/camera_rgb"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

OPENAI_CLIENT = OpenAI()
MAX_ROUNDS = 3
OUTPUT_DIR = "dataset_logs"
SCENE_DIR = "scene_snapshots"


# --- HELPER: COLOR CLASSIFIER (WEBCOLORS) ---
def closest_color(requested_rgb):
    min_colors = {}
    for name in webcolors.names():
        r, g, b = webcolors.name_to_rgb(name)
        min_colors[
            (r - requested_rgb[0]) ** 2
            + (g - requested_rgb[1]) ** 2
            + (b - requested_rgb[2]) ** 2
        ] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_vector):
    """
    Classifies a normalized RGB vector [0-1, 0-1, 0-1] using webcolors.
    """
    # Scale to 0-255 int
    rgb_255 = (
        int(rgb_vector[0] * 255),
        int(rgb_vector[1] * 255),
        int(rgb_vector[2] * 255),
    )

    try:
        return webcolors.rgb_to_name(rgb_255)
    except ValueError:
        return closest_color(rgb_255)


# --- HELPER: ENCODE IMAGE ---
def encode_image_to_base64(numpy_img):
    if numpy_img.shape[-1] == 4:
        numpy_img = numpy_img[:, :, :3]
    img = Image.fromarray(numpy_img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --- AGENT: HUMAN ORACLE ---
class HumanOracle:
    def __init__(self, gt_data, target_cylinder):
        self.gt_data = gt_data
        self.target = target_cylinder

    def get_hint(self, top_down_b64, question, robot_pos):
        cylinders_info = json.dumps(self.gt_data["cylinders"], indent=2)
        target_info = json.dumps(self.target, indent=2)

        system_prompt = (
            "You are a helpful human observer looking at a robot from a ceiling camera. "
            f"Robot Location: {robot_pos}. "
            f"Map Data: {cylinders_info}. "
            f"TARGET OBJECT: {target_info}. "
            "Guide the robot to the target. Give brief, natural language directions."
        )

        response = OPENAI_CLIENT.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"The robot asks: {question}"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{top_down_b64}",
                        },
                    ],
                },
            ],
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        return response.output_text


# --- AGENT: PLANNER ---
class PlannerAgent:
    def __init__(self, goal_description):
        self.history = []
        self.goal_description = goal_description

        # Inject Env Vars into prompt so LLM writes correct topic names
        self.code_template = """
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

TOPIC_CMD_VEL = os.getenv("TOPIC_CMD_VEL", "/cmd_vel")
TOPIC_SCAN = os.getenv("TOPIC_SCAN", "/scan")
TOPIC_CAMERA_RGB = os.getenv("TOPIC_CAMERA_RGB", "/camera_rgb")
ROBOT_FRONT_ANGLE = float(os.getenv("ROBOT_FRONT_ANGLE", "0.0"))

class FindCylinder(Node):
    # --- YOUR LOGIC HERE ---

def main():
    rclpy.init()
    node = FindCylinder()
    node.run_action()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

        self.system_prompt = (
            "You are a Robot Planner using ROS 2. "
            "You view the world through a camera. You are helping a human operator to achieve the following goal. "
            f"GOAL: {goal_description}. "
            "DECISION PROTOCOL:\n"
            "1. IF you cannot see the target or need help: Describe the problem in one sentence and ask for more hints in another. Be friendly to the users!\n"
            "2. IF you know how to move: Output a Python code block using ```python ... ```.\n"
            "   You MUST complete the following template to perform the move:\n"
            f"   {self.code_template}"
        )

    def think(self, ego_image_b64, human_hint=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        for entry in self.history:
            messages.append(entry)

        user_content = None

        if human_hint:
            user_content = [{"type": "input_text", "text": human_hint}]
            self.history.append({"role": "user", "content": human_hint})
        else:
            user_content = [
                {"type": "input_text", "text": "Current camera view."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{ego_image_b64}",
                },
            ]

        messages.append({"role": "user", "content": user_content})

        # No JSON format enforcement
        response = OPENAI_CLIENT.responses.create(
            model="gpt-5-mini",
            input=messages,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )

        result_text = response.output_text
        self.history.append({"role": "assistant", "content": result_text})

        # PARSING LOGIC
        if "```python" in result_text:
            # Extract code between fences
            try:
                code_match = re.search(r"```python(.*?)```", result_text, re.DOTALL)
                if code_match:
                    code_content = code_match.group(1).strip()
                    return {"action": "code", "content": code_content}
            except Exception:
                pass
            return {"action": "ask", "content": "Error parsing code block."}
        else:
            # Treat strictly as dialogue
            return {"action": "ask", "content": result_text}


# --- EXECUTION ENGINE ---
def execute_ros2_code(code_str):
    filename = "generated_move.py"
    if not code_str or "rclpy" not in code_str:
        return "Error: Invalid ROS2 code."

    with open(filename, "w") as f:
        f.write(code_str)

    print(">>> Executing ROS 2 Code...")
    try:
        # Pass current environment variables to the subprocess
        # This ensures the ROS 2 node sees the same ENV vars as the script
        env = os.environ.copy()
        env.update(ENV_CONFIG)

        subprocess.run(["python", filename], env=env, timeout=15, check=True)
        return "Success"
    except subprocess.TimeoutExpired:
        return "Code timed out (Safety Stop)"
    except subprocess.CalledProcessError as e:
        return f"Code failed with error: {e}"


# --- MAIN LOOP ---
def run_episode(scene_id):
    # 1. Load Ground Truth
    gt_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}_gt.json")
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found {gt_path}")
        return

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    # 2. Select Target & Colors
    if not gt_data.get("cylinders"):
        return
    target_cylinder = random.choice(gt_data["cylinders"])

    target_pos = target_cylinder["position"]
    target_rgb = target_cylinder.get("color", [1.0, 1.0, 1.0])
    color_name = get_color_name(target_rgb)

    target_desc = f"Find the {color_name} cylinder and center it in the camera view."
    print(f"TARGET SELECTED: {target_desc} (GT Color RGB: {target_rgb})")

    # 3. Load Stage (DO THIS FIRST)
    scene_usd_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}.usd")
    open_stage(scene_usd_path)

    # --- NEW LOCATION: Initialize World AFTER opening the stage ---
    world = World(stage_units_in_meters=1.0)

    # 4. Cameras
    # Note: Ensure prim_paths exist in the loaded USD
    ego_cam = Camera(prim_path="/World/turtlebot4/oakd_link/Camera", name="ego_cam", resolution=(250, 250))
    top_cam = Camera(prim_path="/World/Camera", name="top_cam", resolution=(640, 480))

    # It is often safer to initialize the world *before* initializing sensors that depend on it
    # though Camera usually works independently.
    world.reset()

    ego_cam.initialize()
    top_cam.initialize()

    # Sensors often return empty/invalid data on the very first frame.
    # We step the physics and renderer a few times to let buffers populate.
    print("Warming up simulation...")
    for _ in range(5):
        world.step(render=True)

    # Agents
    planner = PlannerAgent(goal_description=target_desc)
    human = HumanOracle(gt_data, target_cylinder)

    log_data = {
        "scene": scene_id,
        "target": target_desc,
        "env_vars": ENV_CONFIG,
        "dialogue": [],
    }

    print("--- Starting Episode ---")

    for round_idx in range(MAX_ROUNDS):
        world.step(render=True)

        # Get Images
        ego_data = ego_cam.get_rgba()
        top_data = top_cam.get_rgba()

        # Check if data is valid (H, W, 4) before slicing
        if ego_data.ndim < 3 or top_data.ndim < 3:
            print(
                f"Warning: Invalid camera frame dimensions (Ego: {ego_data.shape}, Top: {top_data.shape}). Retrying step."
            )
            continue

        ego_img = ego_data[:, :, :3]
        top_img = top_data[:, :, :3]
        robot_pos = [0, 0, 0]  # Replace with actual odometry if available

        ego_b64 = encode_image_to_base64(ego_img)
        top_b64 = encode_image_to_base64(top_img)

        # Planner Decision
        print(f"Round {round_idx+1}: Planner Thinking...")
        decision = planner.think(ego_b64)

        # Log Entry
        step_log = {
            "round": round_idx,
            "planner_view_path": f"{OUTPUT_DIR}/{scene_id}_r{round_idx}_ego.jpg",
            "planner_action": decision["action"],
            "planner_content": decision["content"],
        }
        Image.fromarray(ego_img).save(step_log["planner_view_path"])

        # Execute Logic
        if decision["action"] == "ask":
            print(f"Planner asks: {decision['content']}")
            hint = human.get_hint(top_b64, decision["content"], robot_pos)
            print(f"Human replies: {hint}")

            step_log["human_hint"] = hint
            step_log["human_view_path"] = (
                f"{OUTPUT_DIR}/{scene_id}_r{round_idx}_top.jpg"
            )
            Image.fromarray(top_img).save(step_log["human_view_path"])

            planner.think(ego_b64, human_hint=hint)  # Update history

        elif decision["action"] == "code":
            print("Planner generated code. Executing...")
            result = execute_ros2_code(decision["content"])
            step_log["execution_result"] = result
            log_data["dialogue"].append(step_log)
            break

        log_data["dialogue"].append(step_log)
        time.sleep(1)

    with open(f"{OUTPUT_DIR}/log_{scene_id}.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print("Episode Complete.")
    simulation_app.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        run_episode("0000")
    except Exception as e:
        print(f"Sim Error: {e}")
        simulation_app.close()
