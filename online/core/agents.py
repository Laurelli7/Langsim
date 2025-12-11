import json
import re
from ..constants import OPENAI_CLIENT, MODEL

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
            if self.target and self.gt_data:
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

        if not MODEL.startswith("gpt-"):
            # Use Standard OpenAI Chat Completions API (e.g., for vLLM/Qwen)
            response = OPENAI_CLIENT.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The robot asks: {question}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{top_down_b64}"
                                },
                            },
                        ],
                    },
                ],
            )
            return response.choices[0].message.content
        else:
            # Use Custom/Legacy Client
            response = OPENAI_CLIENT.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"The robot asks: {question}",
                            },
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
        self.system_prompt = """
You are a robot planner programing a turtlebot4 running ros2 using Python.
Given the task and camera input, you need to determine if you need to ask for human help or write code to move the robot.

- You are encouraged to ask for human help if you're unsure, or need more information, just respond with a short question in 1-2 sentences.
- If you are sufficiently confident with writing the code to complete the task, respond with Python code that uses ROS2 to control the robot.

You will have access to the following robot messages. You need to use environment variables to choose the topic names. which are: 
    - Subscribe to RGB Camera with topic os.getenv('IMAGE_TOPIC')
    - Subscribe to lidar with topic os.getenv('LIDAR_TOPIC')
    - Subscribe to odometry with topic os.getenv('ODOM_TOPIC')
    - Publish to robot movement with os.getenv('CMD_VEL_TOPIC')
Note that the lidar values have 1080 values, with 0 on the right inreasing counter clockwise, just like a unit circle. 
This means that the front direction is index 270 of the data range, for example. 
Also there are objects near the lidar that are part of the robot, so we should filter nearby values out. Only values > 0.15 for example.
When calculating front distance, take the minimum value in a range around index 270, for example from index 250 to 290; or an average of a tigher range like 260 to 280.

Here is an example implementation of the imports and code formats that uses all three messages. 
Note that the robot is very sensitive. The example code breaks up the process into steps. 
First it aligns itself and confirms its aligned, then it moves towards the destination. 
There is real-world error and lag. Thus, if it didn't first align properly and instead moved immediately towards the color, the result
would result in missing the target. Therefore, your code should assume these errors, and make sure your solutions breaks it down into 
steps and verifies precision like the example script. 

For every subsequent task we need to evaluate the success of the task completion. 
If you complete the task (for example, found and reached a blue cylinder) print "TASK_RESULT:SUCCESS" using self.get_logger().info(''). 
If the task cannot be completed within a reasonable amount of time (for example, cannot find a blue cylinder within 90 seconds), 
print "TASK_RESULT:FAIL" using self.get_logger().info('') and stop. 

Always generate Python code in a code block.

Example color ranges are:

"pink": {"lower": (140,  50,  50), "upper": (179, 255, 255)},
"aqua": {"lower": (70, 205, 205), "upper": (110, 255, 255)},
"black": {"lower": (0, 0, 0), "upper": (180, 255, 80)},
"blue": {"lower": (80, 50, 50), "upper": (140, 255, 255)},
"fuchsia": {"lower": (130, 205, 205), "upper": (170, 255, 255)},
"green": {"lower": (30, 105, 100), "upper": (80, 255, 255)},
"gray": {"lower": (0, 0, 0), "upper": (180, 130, 178)},
"lime": {"lower": (40, 205, 205), "upper": (80, 255, 255)},
"maroon": {"lower": (0, 105, 0), "upper": (20, 255, 178)},
"navy": {"lower": (100, 105, 0), "upper": (140, 255, 178)},
"olive": {"lower": (10, 105, 0), "upper": (50, 255, 178)},
"purple": {"lower": (130, 105, 0), "upper": (170, 255, 178)},
"red": {"lower": (0, 205, 205), "upper": (20, 255, 255)},
"silver": {"lower": (0, 0, 42), "upper": (180, 130, 255)},
"teal": {"lower": (70, 105, 0), "upper": (110, 255, 178)},
"white": {"lower": (0, 0, 105), "upper": (180, 130, 255)},
"yellow": {"lower": (15, 100, 100), "upper": (50, 255, 255)},
"orange": {"lower": (4, 45, 232), "upper": (10, 217, 247)}

Example Code:

```python
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import cv2
import cv_bridge
import numpy as np
import os

import time

# Load topic names from environment variables
from dotenv import load_dotenv
load_dotenv()

class ExecutePolicy(Node):

    def __init__(self):
        super().__init__('example_move')

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # initialize the debugging window
        cv2.namedWindow("window", 1)

        # Set attribute for phase, state, consecutive ready count, etc
        self.phase = 0
        self.state = 0
        self.consec_ready = 0
        self.front_dist = 0
        self.find_color = "pink"

        # subscribe to the robot's RGB camera data stream
        self.image_sub = self.create_subscription(
            Image,
            os.getenv('IMAGE_TOPIC'),
            self.image_callback,
            10
        )
        # subscribe to the robot's lidar
        self.lidar_sub = self.create_subscription(
            LaserScan,
            os.getenv('LIDAR_TOPIC'),
            self.lidar_callback,
            10
        )
        # Create a publisher to the robots command velocity
        self.publisher_ = self.create_publisher(
            Twist, 
            os.getenv('CMD_VEL_TOPIC'),
            10
        )

    # Most of the logic will be done in image_callback. lidar_callback just sets some distance param
    def lidar_callback(self, msg):
        # We find the minimum distance in the near-front direction
        # Ignore values of 0.15 or less since there are objects near the lidar
        err = 40
        start = 270 - err
        stop = 270 + err
        self.front_dist = 10
        for i in range(start, stop):
            if 0.15 < msg.ranges[i] < self.front_dist:
                self.front_dist = msg.ranges[i]

    def image_callback(self, msg):
        match self.phase:
            case 0:
                self.phase0_find_color(msg)
            case 1:
                self.phase1_to_color(msg)
    
    def phase0_find_color(self, msg):
        # converts the incoming ROS message to OpenCV format and HSV
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # pink
        if self.find_color == "pink":
            lower_color = np.array([156.38, 109.86, 150.35])
            upper_color = np.array([162.66, 255.00, 165.65])
        # green
        elif self.find_color == "green":
            lower_color = np.array([35, 90, 110])
            upper_color = np.array([41, 140, 210])
        # blue
        elif self.find_color == "blue":
            lower_color = np.array([93.64, 56.98, 146.35])
            upper_color = np.array([99.93, 227.83, 161.65])

        # erase all pixels that aren't that color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # limit the search to slice out the top 1/4
        h, w, d = image.shape
        search_top = int(h/4)
        search_bot = h
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # draw a rectangle around the search box
        cv2.rectangle(image, (0, search_top), (w, search_bot), (0, 0, 0), 1)

        # calculate moments to find the center of color pixels
        M = cv2.moments(mask)

        # if there are any pixels found
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # draw a red circle at the center
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)

            # Turn towards the color
            err_x = cx - (w // 2)
            msg1 = Twist()
            msg1.angular.z = -0.001 * err_x

            self.publisher_.publish(msg1)
            self.get_logger().info(f'Found color {self.find_color}! With error {err_x}.')
            # Increment consecutive ready if low enough error
            if abs(err_x) <= 10:
                self.consec_ready += 1
        # else, we keep turning until we find something
        else:
            msg1 = Twist()
            msg1.angular.z = 0.3
            self.publisher_.publish(msg1)
            self.get_logger().info(f'Searching for {self.find_color}...')
            self.consec_ready = 0
        
        # Move on to next phase if facing the color 20 iterations in a row (means ready)
        if self.consec_ready >= 20:
            self.next_phase()
            self.get_logger().info(f'phase0_find_color complete! Moving on to phase 1...')
            self.consec_ready = 0

        # show debugging window
        cv2.imshow("window", image)
        cv2.waitKey(3)
    
    def phase1_to_color(self, msg):
        # converts the incoming ROS message to OpenCV format and HSV
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # pink
        if self.find_color == "pink":
            lower_color = np.array([140,  40,  80], dtype=np.uint8)
            upper_color = np.array([179, 255, 255], dtype=np.uint8)
        # green
        elif self.find_color == "green":
            lower_color = np.array([35, 90, 110])
            upper_color = np.array([41, 140, 210])
        # blue
        elif self.find_color == "blue":
            lower_color = np.array([93.64, 56.98, 146.35])
            upper_color = np.array([99.93, 227.83, 161.65])

        # erase all pixels that aren't that color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # limit the search to slice out the top 1/4
        h, w, d = image.shape
        search_top = int(h/4)
        search_bot = h
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # draw a rectangle around the search box
        cv2.rectangle(image, (0, search_top), (w, search_bot), (0, 0, 0), 1)

        # calculate moments to find the center of color pixels
        M = cv2.moments(mask)

        # if there are any pixels found
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # draw a red circle at the center
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)

            # Turn towards the color
            err_x = cx - (w // 2)
            msg1 = Twist()
            msg1.angular.z = -0.004 * err_x
            msg1.linear.x = (self.front_dist - 0.35) / 2.5

            self.publisher_.publish(msg1)
            self.get_logger().info(f'Moving towards {self.find_color} with speed {msg1.linear.x}! Distance: {self.front_dist}.')
            # Increment consecutive ready if low enough error
            if abs(msg1.linear.x) <= 0.02:
                self.consec_ready += 1
        # else, we keep turning until we find something
        else:
            msg1 = Twist()
            msg1.angular.z = 0.3
            self.publisher_.publish(msg1)
            self.get_logger().info(f'Searching for {self.find_color}...')
            self.consec_ready = 0
        
        # Move on to next phase if facing the color 20 iterations in a row (means ready)
        if self.consec_ready >= 20:
            self.next_phase()
            self.get_logger().info(f'phase1_to_color complete! Moving on to phase 2...')
            self.consec_ready = 0

        # show debugging window
        cv2.imshow("window", image)
        cv2.waitKey(3)
    
    def next_phase(self):
        self.phase = self.phase + 1
    
    def back_phase(self):
        self.phase = self.phase - 1

def main(args=None):
    rclpy.init(args=args)
    execute_policy_node = ExecutePolicy()
    try:
        rclpy.spin(execute_policy_node)
    except KeyboardInterrupt:
        pass
    finally:
        execute_policy_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```
"""

        self.history.extend(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.goal_description},
            ]
        )

    def think(self, ego_image_b64, human_hint=None):
        messages = []
        messages.extend(self.history)

        if human_hint:
            messages.append({"role": "user", "content": human_hint})
            self.history.append({"role": "user", "content": human_hint})
        else:
            # Differentiate formatting between standard ChatCompletions and custom client
            if not MODEL.startswith("gpt-"):
                # Standard Chat Completion Format (vLLM/OpenAI Standard)
                msg = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current camera view."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ego_image_b64}"
                            },
                        },
                    ],
                }
            else:
                # Legacy/Custom Client Format
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

        # Call the API
        if not MODEL.startswith("gpt-"):
            response = OPENAI_CLIENT.chat.completions.create(
                model=MODEL, messages=messages
            )
            result_text = response.choices[0].message.content
        else:
            response = OPENAI_CLIENT.responses.create(model=MODEL, input=messages)
            result_text = response.output_text

        self.history.append({"role": "assistant", "content": result_text})

        if "```python" in result_text:
            code_match = re.search(r"```python(.*?)```", result_text, re.DOTALL)
            if code_match:
                return {"action": "code", "content": code_match.group(1).strip()}

        return {"action": "ask", "content": result_text}
