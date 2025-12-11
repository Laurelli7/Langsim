import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
print("GPT init successfully!")

# Folder for JSON output
OUTPUT_DIR = "json_scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Python code template
code_template = """
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

goal_descriptions = [
    "Spin in a circle",
    "Turn and follow the nearest object",
    "Follow along the orange line on the ground",
]

SYSTEM_PROMPTS = []

for goal in goal_descriptions:
    prompt = f"""
You are a Robot Planner using ROS 2.
GOAL: {goal}.

Output format:
   - Output Python code, and put it inside a code fence (for example: three backticks with the word python, then your code, then three backticks to close).
   - Use this template as a starting point for your node:

{code_template}

   - When generating code, ensure you make only necessary efforts to accomplish the goal, and ensure each state has a time limit such as 60 seconds.
   - Avoid including any other remarks besides just the code 

You have access to the following helper functions (imported via "from helpers import *"):

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

7. get_lidar_ranges(scan_msg, max_val=10.0, min_safe=0.15) -> list
   - Returns a processed list of LIDAR distances from the raw LaserScan message.
   - Replaces values that are infinite or below min_safe (0.15 m) with max_val (default 10.0).
   - The resulting list typically contains 1080 values, where index 0 is the robot's Right and index 270 is Front.

8. get_nearest_front_distance(scan_msg, error=30) -> float
   - Finds the closest object in a specific slice directly in front of the robot.
   - Scans indices centered at 270 (Front) with a margin of +/- error (default 30 indices).
   - Returns the minimum valid distance found; returns 10.0 if all readings are infinite or too close (noise).

9. get_scan_range_by_degree(scan_msg, start_degree, end_degree, max_val=10.0, min_safe=0.15) -> list
   - Returns a cleaned list of LIDAR distances for a specific angular range (in degrees).
   - Maps 0 degrees to the robot's Right side, increasing counter-clockwise (approx. 3 indices per degree).
   - Applies the same cleaning logic as get_lidar_ranges to handle infinite or noisy readings.

Important constraints:
- These helper functions do not handle sleep or durations internally; they are single-step controllers.
- In your generated code, use ROS 2 timers or a main loop to call these functions at a fixed rate and implement
  higher-level behaviors (search, approach, etc.) as a simple state machine.
- When using lidar scans, make sure to ignore invalid readings outside of [0.2, 10.0] meters, such as NaN, inf or -1.0.

Color detection helpers:
- You are given loose HSV bounds for several HTML4 color names, suitable for OpenCV-based color detection.
- HSV is in OpenCV format: H in [0, 179], S in [0, 255], V in [0, 255].
- You can use these bounds directly when calling find_colored_blob, rotate_and_search_step, or
  approach_colored_object_step.

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
"""
    SYSTEM_PROMPTS.append(prompt)

# Main generation loop

for scenario_idx, prompt in enumerate(SYSTEM_PROMPTS):
    print(f"Scenario {scenario_idx + 1} ({goal_descriptions[scenario_idx]})")
    scenes = []

    for i in range(10):
        print(f"  Starting iteration {i}...")
        messages = [
            {"role": "system", "content": prompt},
        ]

        print("  Prompting GPT...")
        first_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        assistant_content = first_resp.choices[0].message.content

        scene_obj = {
            "scene": f"{i:04d}",
            "dialogue": [
                {
                    "role": "system",
                    "content": prompt.strip(),
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ],
            # Blank training fields (to stay consistent with Peter's json)
            "steps": [],
            "success": False,
            "final_distance": 0.0,
            "rounds_taken": 0,
        }

        scenes.append(scene_obj)
        print(f"  Iteration {i} added to scenario {scenario_idx + 1} JSON.")

    # Save one JSON file per scenario
    json_path = os.path.join(
        OUTPUT_DIR,
        f"scenario_{scenario_idx + 1:02d}.json"
    )
    with open(json_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"Scenario {scenario_idx + 1} saved to {json_path}!")

print("All scenarios generated and saved as JSON.")
