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
