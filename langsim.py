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
import math
from PIL import Image
import cv2  # OpenCV for drawing the arrow

# Isaac Sim Core Imports
from omni.isaac.kit import SimulationApp

# START SIMULATION APP
# set headless to True if you do not need the GUI
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.syntheticdata import helpers
from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.world import World
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from openai import OpenAI


# --- CONFIGURATION (ENV VARS) ---
ENV_CONFIG = {
    "ROBOT_FRONT_ANGLE": os.getenv("ROBOT_FRONT_ANGLE", "0.0"),
    "TOPIC_CMD_VEL": os.getenv("TOPIC_CMD_VEL", "/cmd_vel"),
    "TOPIC_SCAN": os.getenv("TOPIC_SCAN", "/scan"),
    "TOPIC_CAMERA_RGB": os.getenv("TOPIC_CAMERA_RGB", "/camera_rgb"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

OPENAI_CLIENT = OpenAI()
MAX_ROUNDS = 3
OUTPUT_DIR = "dataset_logs"
SCENE_DIR = "scene_snapshots"


# --- HELPER: 3D TO 2D PROJECTION ---
def project_world_to_screen(point_3d, view_mtx, proj_mtx, img_width, img_height):
    """
    Projects a 3D world coordinate to 2D screen pixel coordinates.
    """
    # 1. Homogeneous coordinates
    point_h = np.append(point_3d, 1.0)

    # 2. World -> Camera Space (Row-major multiplication)
    point_cam = point_h @ view_mtx

    # 3. Camera -> Clip Space
    point_clip = point_cam @ proj_mtx

    # 4. Perspective Division (Clip -> NDC)
    if abs(point_clip[3]) < 1e-5:
        return None
    point_ndc = point_clip[:3] / point_clip[3]

    # 5. Check bounds (optional safety)
    if point_clip[3] < 0:  # Point is behind camera
        return None

    # 6. NDC -> Pixel Coordinates
    # NDC x [-1, 1] -> u [0, W]
    # NDC y [-1, 1] -> v [H, 0] (Flip Y)
    u = int((point_ndc[0] + 1) * 0.5 * img_width)
    v = int((-point_ndc[1] + 1) * 0.5 * img_height)

    return (u, v)


# --- HELPER: COLOR CLASSIFIER ---
def closest_color(requested_rgb):
    min_colors = {}
    for name in webcolors.names("html4"):
        r, g, b = webcolors.name_to_rgb(name)
        min_colors[
            (r - requested_rgb[0]) ** 2
            + (g - requested_rgb[1]) ** 2
            + (b - requested_rgb[2]) ** 2
        ] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_vector):
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
    
    # Ensure contiguous array for Pillow (OpenCV can disrupt this)
    numpy_img = np.ascontiguousarray(numpy_img)
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
            "You are a helpful human observer looking at a turtlebot 4 from a ceiling camera. "
            "The RED ARROW in the image indicates the robot's position and facing direction. "
            f"Robot Location: {robot_pos}. "
            f"Map Data: {cylinders_info}. "
            f"TARGET OBJECT: {target_info}. "
            "Guide the robot to the target. Give brief, natural language directions."
        )

        response = OPENAI_CLIENT.responses.create(
            model="gpt-4o",
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
        )
        return response.output_text


# --- AGENT: PLANNER ---
class PlannerAgent:
    def __init__(self, goal_description):
        self.history = []
        self.goal_description = goal_description
        self.code_template = """
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

TOPIC_CMD_VEL = os.getenv("TOPIC_CMD_VEL", "/cmd_vel")

class FindCylinder(Node):
    # --- YOUR LOGIC HERE ---
    # Example:
    # def __init__(self):
    #     super().__init__('find_cylinder_node')
    #     self.pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)
    # def run_action(self):
    #     pass

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
            f"GOAL: {goal_description}. "
            "DECISION PROTOCOL:\n"
            "1. IF you think you still need extra information or human help: Ask for a hint.\n"
            "2. IF you know how to move based on the camera view or human hint: Output Python code using ```python ... ```.\n"
            f"   Use this template:\n{self.code_template}"
        )

    def think(self, ego_image_b64, human_hint=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        for entry in self.history:
            messages.append(entry)

        if human_hint:
            messages.append({"role": "user", "content": human_hint})
            self.history.append({"role": "user", "content": human_hint})
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Current camera view."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{ego_image_b64}"},
                ]
            })
            
        print(messages)

        response = OPENAI_CLIENT.responses.create(
            model="gpt-4o",
            input=messages
        )
        result_text = response.output_text
        self.history.append({"role": "assistant", "content": result_text})

        if "```python" in result_text:
            try:
                code_match = re.search(r"```python(.*?)```", result_text, re.DOTALL)
                if code_match:
                    return {"action": "code", "content": code_match.group(1).strip()}
            except Exception:
                pass
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
        env = os.environ.copy()
        env.update(ENV_CONFIG)
        subprocess.run(["python", filename], env=env, timeout=15, check=True)
        return "Success"
    except subprocess.TimeoutExpired:
        return "Code timed out"
    except Exception as e:
        return f"Code failed: {e}"


# --- MAIN LOOP ---
def run_episode(scene_id):
    # 1. Load Ground Truth
    gt_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}_gt.json")
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found {gt_path}")
        return

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    if not gt_data.get("cylinders"):
        return
    target_cylinder = random.choice(gt_data["cylinders"])
    target_rgb = target_cylinder.get("color", [1.0, 1.0, 1.0])
    color_name = get_color_name(target_rgb)
    target_desc = f"Find the {color_name} cylinder."
    print(f"TARGET: {target_desc}")

    # 2. Load Stage
    scene_usd_path = os.path.abspath(f"{SCENE_DIR}/scene_{scene_id}.usd")
    open_stage(scene_usd_path)
    
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 3. Setup Cameras & Robot Tracker
    ego_cam_path = "/World/turtlebot4/oakd_link/Camera"
    top_cam_path = "/World/Camera"
    
    ego_cam = Camera(prim_path=ego_cam_path, name="ego_cam", resolution=(250, 250))
    top_cam = Camera(prim_path=top_cam_path, name="top_cam", resolution=(1280, 720))
    
    # Initialize the Robot Prim for Tracking (Adjust path if needed)
    robot_prim = XFormPrim(prim_path="/World/turtlebot4", name="robot_base")

    world.reset()
    ego_cam.initialize()
    top_cam.initialize()
    robot_prim.initialize()

    # Warmup
    for _ in range(5):
        world.step(render=True)

    planner = PlannerAgent(goal_description=target_desc)
    human = HumanOracle(gt_data, target_cylinder)
    log_data = {"scene": scene_id, "dialogue": []}

    print("--- Starting Episode ---")

    hint = None

    for round_idx in range(MAX_ROUNDS):
        world.step(render=True)

        # A. Capture Data
        ego_data = ego_cam.get_rgba()
        top_data = top_cam.get_rgba()

        if ego_data.ndim < 3 or top_data.ndim < 3:
            continue

        ego_img = ego_data[:, :, :3].copy()
        top_img = top_data[:, :, :3].copy()

        # B. Get Robot Pose & Draw Arrow
        # Get absolute position and orientation
        pos, rot = robot_prim.get_world_pose()
        
        # Calculate yaw
        _, _, yaw = quat_to_euler_angles(rot)
        
        # Define arrow geometry in 3D
        arrow_len = 0.6
        start_3d = pos + np.array([0, 0, 0.1]) # Lift slightly
        end_3d = start_3d + np.array([np.cos(yaw) * arrow_len, np.sin(yaw) * arrow_len, 0])

        # --- REPLACED: Custom Projection Matrix Logic ---
        # 1. Get View Matrix (Keeping ROS convention for View if Camera is setup that way)
        view_mtx = top_cam.get_view_matrix_ros()
        
        # 2. Compute Projection Matrix Manually
        # Get actual prim to read attributes
        camera_prim = stage.GetPrimAtPath(top_cam_path)
        
        # Read USD attributes
        focal_length = camera_prim.GetAttribute("focalLength").Get()
        horiz_aperture = camera_prim.GetAttribute("horizontalAperture").Get()
        # Using sensor resolution instead of viewport_api to ensure sync with image data
        width, height = top_cam.get_resolution() 
        near, far = camera_prim.GetAttribute("clippingRange").Get()
        
        # Calculations
        aspect_ratio = width / height
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        
        # Helper to compute projection matrix
        # Note: helpers returns a list, we need numpy array
        proj_list = helpers.get_projection_matrix(fov, aspect_ratio, near, far)
        proj_mtx = np.array(proj_list).reshape(4, 4)
        
        # -----------------------------------------------

        # Project
        p_start = project_world_to_screen(start_3d, view_mtx, proj_mtx, width, height)
        p_end = project_world_to_screen(end_3d, view_mtx, proj_mtx, width, height)

        # Draw
        if p_start and p_end:
            # Color is RGB (Red)
            cv2.arrowedLine(top_img, p_start, p_end, (255, 0, 0), 4, tipLength=0.3)
        else:
            print("Warning: Robot out of top-down view.")

        # C. Encode
        ego_b64 = encode_image_to_base64(ego_img)
        top_b64 = encode_image_to_base64(top_img) # Contains arrow now

        # D. Planner Think
        print(f"Round {round_idx+1}: Planner Thinking...")
        decision = planner.think(ego_b64, human_hint=hint)

        step_log = {
            "round": round_idx,
            "robot_pos": pos.tolist(),
            "planner_action": decision["action"],
            "planner_content": decision["content"]
        }

        # E. Execute or Ask
        if decision["action"] == "ask":
            print(f"Planner asks: {decision['content']}")
            hint = human.get_hint(top_b64, decision["content"], pos.tolist())
            print(f"Human replies: {hint}")
            
            # Save visual log
            Image.fromarray(top_img).save(f"{OUTPUT_DIR}/{scene_id}_r{round_idx}_top.jpg")
            
            step_log["human_hint"] = hint

        elif decision["action"] == "code":
            print("Planner generated code.")
            result = execute_ros2_code(decision["content"])
            step_log["result"] = result
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
