import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Use env var for API key
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
print("GPT init successfully!")

# Folders for the two sets of scripts
print("Creating folders...")
os.makedirs("scenario_1", exist_ok=True)
os.makedirs("scenario_2", exist_ok=True)

SYSTEM_PROMPT = """
                 You are generating code for a turtlebot4 running ros2 using Python.
                 Do not generate anything else besides Python. You will have access to 3 robot messages.
                 You need to use environment variables to choose the topic names. which are: 
                    - Subscribe to RGB Camera with topic os.getenv('IMAGE_TOPIC')
                    - Subscribe to lidar with topic os.getenv('LIDAR_TOPIC')
                    - Publish to robot movement with os.getenv('CMD_VEL_TOPIC')
                Note that the lidar values have 1080 values, with 0 on the right inreasing counter clockwise, just like a unit circle. 
                This means that the front direction is index 270 of the data range, for example. 
                Also there are objects near the lidar that are part of the robot, so we should filter nearby values out. Only values > 0.15 for example.

                Here is an example implementation of the imports and code formats that uses all three messages. 
                Note that the robot is very sensitive. The example code breaks up the process into steps. 
                First it aligns itself and confirms its aligned, then it moves towards the destination. 
                There is real-world error and lag. Thus, if it didn't first align properly and instead moved immediately towards the color, the result
                would result in missing the target. Therefore, your code should assume these errors, and make sure your solutions breaks it down into 
                steps and verifies precision like the example script. 

                For every subsequent task we need to evaluate the success of the task completion. 
                If you complete the task (for example, found and reached a blue cylinder) print "TASK_RESULT:SUCCESS" using self.get_logger().info(''). 
                If the task cannot be completed within a reasonable amount of time (for example, cannot find a blue cylinder within 20 seconds), 
                print "TASK_RESULT:FAIL" using self.get_logger().info('') and stop. 

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
            self.get_logger().info(f'Found color \{self.find_color\}! With error \{err_x\}.')
            # Increment consecutive ready if low enough error
            if abs(err_x) <= 10:
                self.consec_ready += 1
        # else, we keep turning until we find something
        else:
            msg1 = Twist()
            msg1.angular.z = 0.3
            self.publisher_.publish(msg1)
            self.get_logger().info(f'Searching for \{self.find_color\}...')
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
            self.get_logger().info(f'Moving towards \{self.find_color\} with speed \{msg1.linear.x\}! Distance: \{self.front_dist\}.')
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
"""

FOLLOWUP_PROMPT = """The code wasn't able to find the blue cylinder, it was not in sight. 
                     Here's a tip to locate the blue cylinder: 
                     """

INITIAL_PROMPTS = ["Find and move to the blue cylinder",
                  "Detect the blue cylinder, and go to it",
                  "Locate the blue cylinder than move to it",
                  "Go to blue cylinder",
                  "Write code to go to the blue cylinder",
                  "Write code to locate and move to the blue cyliinder",
                  "Generate code to move to the blue cylinder",
                  "Generate code to find and move to the blue cylinder"]

FOLLOWUP_PROMPTS_2 = ["If you move to the pink cylinder first, then search from there, you should be able to find and reach the blue cylinder.",
                      "If you go to the pink cylinder first, then you can search for the blue cylinder and find it",
                      "The blue cylinder is locatable from where the pink cylinder is at. Move to the pink first, then you can find the blue",
                      "The pink cylinder has a good angle of where the blue cylinder is, try moving there first then see if you can go to the blue cylinder",
                      "I can see the blue cylinder from where the pink cylinder is at, try moving there first, then searching for the blue cylinder",
                      "If you can't see the blue cylinder, try moving to the pink one first. That has a better angle of the blue cylinder",
                      "Find the pink cylinder, move to it, find the blue cylinder, then move to it",
                      "Take this in steps: Find then move to the pink cylinder for a better angle, then try to find and align to the blue cylinder, then move to it"]

def extract_code(text):
    # If there are no ```, just return it
    if "```" not in text:
        return text
    
    parts = text.split("```")
    if len(parts) < 3:
        # Something weird, fall back to original
        print("Warning: Something weird with text")
        return text

    # The code block is usually in parts[1]
    code_block = parts[1]

    # Remove possible "python" on first line
    lines = code_block.splitlines()
    if lines and lines[0].strip().lower().startswith("python"):
        lines = lines[1:]

    return "\n".join(lines).strip() + "\n"

for i in range(100):
    print(f"Starting iteration {i}...")
    # Ask to find blue cylinder
    print("Prompting first prompt to GPT...")
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (INITIAL_PROMPTS[i % 8])}
    ]

    first_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=initial_messages,
    )

    first_code = first_resp.choices[0].message.content
    first_code = extract_code(first_code)

    file1_path = f"scenario_1/script_{i:03d}.py"
    with open(file1_path, "w") as f:
        f.write(first_code)
    print(f"Scenario-1 iteration-{i} generated and saved!")

    print("Prompting followup prompt to GPT...")
    # Can't see blue cylinder scenario (always runs)
    followup_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (INITIAL_PROMPTS[i % 8])},
        {"role": "assistant", "content": first_code},
        {"role": "user", "content": (FOLLOWUP_PROMPT + FOLLOWUP_PROMPTS_2[i % 8])},
    ]

    second_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=followup_messages,
    )

    second_code = second_resp.choices[0].message.content
    second_code = extract_code(second_code)

    file2_path = f"scenario_2/script_{i:03d}.py"
    with open(file2_path, "w") as f:
        f.write(second_code)
    print(f"Scenario-2 iteration-{i} generated and saved!")


