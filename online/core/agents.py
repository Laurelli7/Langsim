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
            return input("Enter your advice to the robot: ")

        # Otherwise, use the LLM mode
        cylinders_info = json.dumps(self.gt_data["cylinders"], indent=2)
        target_info = json.dumps(self.target, indent=2)

        system_prompt = (
            "You are a helpful human observer looking at a turtlebot 4 from a ceiling camera. "
            f"Robot position and direction is shown as the red arrow in the top down view. "
            f"Position: {robot_pos}. "
            f"Objects Data: {cylinders_info}. "
            f"TARGET OBJECT: {target_info}. "
            "- Guide the robot to the target. Give brief, one-sentence, natural language directions as a human operator."
            "- Nav planning is priority, low-level control advice is supplemental. "
            "- If the target is blocked, suggest how to reach the target step by step. In particularly complex scenarios, suggest robot to come back for more feedback after a step. "
            "- The robot doesn't have the ceiling camera view, so give advice relative to robot position and direction instead of global up-down and left-right."
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from helpers import *

# The following topics are available in the environment
TOPIC_CMD_VEL = os.getenv("TOPIC_CMD_VEL", "/cmd_vel")
TOPIC_SCAN = os.getenv("TOPIC_SCAN", "/scan")
TOPIC_CAMERA = os.getenv("TOPIC_CAMERA", "/camera_rgb")
ROBOT_FRONT_ANGLE = float(os.getenv("ROBOT_FRONT_ANGLE", "0.0"))  # in radians

class FindCylinder(Node):
    def __init__(self):
        super().__init__('find_cylinder')

        self.cmd_vel_pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)
        self._latest_image = None
        self.image_width = None

        self.create_subscription(
            Image,
            TOPIC_CAMERA,
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        
        # Your state machine and logic goes here
        self.state = ""
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9  # seconds

    def image_callback(self, msg: Image):
        try:
            self._latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.image_width is None and self._latest_image is not None:
                self.image_width = self._latest_image.shape[1]
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def get_latest_image(self):
        return self._latest_image
    
    # Your additional methods go here


def main(args=None):
    rclpy.init(args=args)
    node = FindCylinder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```
"""

        self.system_prompt = f"""
You are a Robot Planner using ROS 2.
GOAL: {goal_description}.

1. **You are encouraged to ask for extra information whenever you're unsure how to achieve the goal**:
   - Describe your problem in one sentence and ask for help in another.
   - Prioritize asking high-level nav planning and problem-solving questions, only ask specific low-level questions if absolutely necessary.

2. If you think you have enough information from the observations and human hints:
   - Output Python code, and make sure you use code fences: `python ...`.
   - Use this template:
{self.code_template}
   - When generating code, ensure you make only necessary efforts to accomplish the goal, each state should have a time limit such as 60 seconds.

You have access to the following helper functions (imported via `from helpers import *`):

1. move_forward(cmd_vel_pub, speed=0.2)
   - Sends a single Twist command to move the robot forward with linear.x = speed and angular.z = 0.0.
   - Non-blocking: call this from a timer or loop at a fixed rate; the caller is responsible for timing/duration.

2. rotate(cmd_vel_pub, angular_speed=0.5, clockwise=True)
   - Sends a single Twist command to rotate in place.
   - angular.z = -abs(angular_speed) if clockwise, else +abs(angular_speed).
   - Non-blocking: call repeatedly for as long as you want to rotate.

3. stop(cmd_vel_pub)
   - Sends a Twist command with linear.x = 0.0 and angular.z = 0.0 to stop the robot.

4. find_colored_blob(cv_image, lower_hsv, upper_hsv, min_area=200) -> (found, cx, cy, area)
   - Takes a BGR cv2 image and HSV bounds and returns whether a blob was found.
   - If found is True, cx and cy are the centroid pixel coordinates of the largest blob, and area is its contour area.
   - If no valid blob is found, returns (False, None, None, 0.0).

5. rotate_and_search_step(cmd_vel_pub, get_latest_image, lower_hsv, upper_hsv,
                          angular_speed=0.4, clockwise=False, min_area=200, active=True)
   - ONE CONTROL STEP of rotating in place while searching for a colored object.
   - If active is True, publishes a rotation Twist and looks at the latest image from get_latest_image().
   - Returns (found, cx, area):
       * found: True if the target color blob is detected in this step.
       * cx: horizontal pixel coordinate of the blob center (or None).
       * area: blob area (0 if not found).
   - Non-blocking: caller must manage timing, number of steps, and state transitions.

6. approach_colored_object_step(cmd_vel_pub, get_latest_image, lower_hsv, upper_hsv,
                                image_width, target_area=20000, linear_speed=0.18,
                                k_angular=0.004, min_area=200, active=True)
   - ONE CONTROL STEP of approaching a colored object while trying to keep it centered in the image.
   - The camera resolution is 250x250 pixels and a target area of 20000 roughly means being 0.8 meters away.
   - Uses the latest camera image to find the target blob and computes a Twist with forward motion plus
     an angular correction proportional to horizontal pixel error.
   - Returns (reached, found, area):
       * reached: True if area >= target_area on this step (object is close enough).
       * found: True if the target blob is visible in this step.
       * area: current blob area (0 if not found).
   - Non-blocking: caller must handle timeouts, stopping conditions, and switching states.

Important constraints:
- These helper functions do NOT handle time.sleep or durations internally; they are single-step controllers.
- In your generated code, use ROS 2 timers or a main loop to call these functions at a fixed rate and implement
  higher-level behaviors (search, approach, etc.) as a simple state machine.
- When using lidar scans, make sure to ignore invalid readings outside of [0.2, 10.0] meters, such as NaN, inf or -1.0.

Color detection helpers:
- You are given loose HSV bounds for several HTML4 color names, suitable for OpenCV-based color detection.
- HSV is in OpenCV format: H in [0, 179], S in [0, 255], V in [0, 255].
- You can use these bounds directly when calling find_colored_blob, rotate_and_search_step, or
  approach_colored_object_step.

```python
COLOR_BOUNDS = {{
    'aqua':    {{'lower': (70, 205, 205), 'upper': (110, 255, 255)}},
    'black':   {{'lower': (0,   0,   0),  'upper': (180, 255, 80)}},
    'blue':    {{'lower': (100, 205, 205),'upper': (140, 255, 255)}},
    'fuchsia': {{'lower': (130, 205, 205),'upper': (170, 255, 255)}},
    'green':   {{'lower': (40,  105,   0),'upper': (80,  255, 178)}},
    'gray':    {{'lower': (0,   0,    0),'upper': (180, 130, 178)}},
    'lime':    {{'lower': (40,  205, 205),'upper': (80,  255, 255)}},
    'maroon':  {{'lower': (0,   105,   0),'upper': (20,  255, 178)}},
    'navy':    {{'lower': (100, 105,   0),'upper': (140, 255, 178)}},
    'olive':   {{'lower': (10,  105,   0),'upper': (50,  255, 178)}},
    'purple':  {{'lower': (130, 105,   0),'upper': (170, 255, 178)}},
    'red':     {{'lower': (0,   205, 205),'upper': (20,  255, 255)}},
    'silver':  {{'lower': (0,   0,   42), 'upper': (180, 130, 255)}},
    'teal':    {{'lower': (70,  105,   0),'upper': (110, 255, 178)}},
    'white':   {{'lower': (0,   0,  105), 'upper': (180, 130, 255)}},
    'yellow':  {{'lower': (10,  205, 205),'upper': (60,  255, 255)}},
}}
```
"""

        self.history.append({"role": "system", "content": self.system_prompt})

    def think(self, ego_image_b64, human_hint=None):
        messages = []
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
