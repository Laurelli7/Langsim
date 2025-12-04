import json
import re
from ..constants import OPENAI_CLIENT

import json


class HumanOracle:
    def __init__(self, gt_data, target_cylinder, human_mode=False):
        """
        human_mode=True → bypass LLM and ask a human via terminal
        """
        self.gt_data = gt_data
        self.target = target_cylinder
        self.human_mode = human_mode

    def get_hint(self, top_down_b64, question, robot_pos):
        # If human mode is enabled, ask operator through terminal
        if self.human_mode:
            print("\n=== HUMAN ORACLE MODE ===")
            print(f"Robot Position: {robot_pos}")
            print("Question:", question)
            print("Target:", json.dumps(self.target, indent=2))
            print("Cylinders:", json.dumps(self.gt_data["cylinders"], indent=2))
            print("\n(Top-down image omitted; base64 available if needed.)")
            return input("Enter your advice to the robot: ")

        # Otherwise, use the LLM mode
        cylinders_info = json.dumps(self.gt_data["cylinders"], indent=2)
        target_info = json.dumps(self.target, indent=2)

        system_prompt = (
            "You are a helpful human observer looking at a turtlebot 4 from a ceiling camera. "
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
        )
        return response.output_text


class PlannerAgent:
    def __init__(self, goal_description):
        self.history = []
        self.goal_description = goal_description
        self.code_template = """
```python
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

# The following topics are available in the environment
TOPIC_CMD_VEL = os.getenv("TOPIC_CMD_VEL", "/cmd_vel")
TOPIC_SCAN = os.getenv("TOPIC_SCAN", "/scan")
TOPIC_CAMERA = os.getenv("TOPIC_CAMERA", "/camera_rgb")

class FindCylinder(Node):
    # --- YOUR LOGIC HERE ---

def main():
    rclpy.init()
    node = FindCylinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
"""
        self.system_prompt = (
            "You are a Robot Planner using ROS 2. "
            f"GOAL: {goal_description}. "
            "1. You are encouraged to ask for extra information or human help: Describe your problem in one sentence and ask for help in another. "
            "2. If you think you have enough information from the observations and human hints: Output Python code, make sure you use code fences ```python ... ```."
            f" Use this template:\n{self.code_template}"
            "When generating code, ensure you make only necessary efforts to accomplish the goal, and always exit the program if the goal cannot be accomplished with existing information."
        )

    def think(self, ego_image_b64, human_hint=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)

        if human_hint:
            messages.append({"role": "user", "content": human_hint})
            self.history.append({"role": "user", "content": human_hint})
        else:
            msg = {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Current camera view."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{ego_image_b64}",
                    },
                ],
            }
            messages.append(msg)

        response = OPENAI_CLIENT.responses.create(model="gpt-5-mini", input=messages)
        result_text = response.output_text
        self.history.append({"role": "assistant", "content": result_text})

        if "```python" in result_text:
            code_match = re.search(r"```python(.*?)```", result_text, re.DOTALL)
            if code_match:
                return {"action": "code", "content": code_match.group(1).strip()}

        return {"action": "ask", "content": result_text}
