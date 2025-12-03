#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np


class ApproachCylinderCamera(Node):
    def __init__(self):
        super().__init__('approach_cylinder_camera')

        # ===== Parameters you’ll probably want to tune =====
        # Stop when cylinder fills at least this many pixels in the image
        self.min_area_to_stop = 800         # (pixels) adjust for your camera/distance
        self.max_linear_speed = 0.20           # m/s
        self.angular_gain = 0.003              # rad per pixel error
        self.linear_gain = 0.00001             # speed scaling vs area error

        # HSV color range for the cylinder (example: dark/black object)
        # You MUST tune these for your cylinder’s color & lighting.
        self.lower_hsv = np.array([0, 0, 0])   # lower bound
        self.upper_hsv = np.array([180, 255, 60])  # upper bound (dark colors)

        # ==================================================

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image,
            '/camera_rgb',
            self.image_callback,
            10
        )

        self.get_logger().info("ApproachCylinderCamera node started. Using camera area to approach cylinder.")

    def image_callback(self, msg: Image):
        # Convert ROS Image -> OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w, _ = cv_image.shape

        # Convert to HSV for easier color thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Mask for the cylinder color
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Optional: clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()

        if len(contours) == 0:
            # Nothing detected – stop or slowly rotate to search
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return

        # Use the largest contour as the cylinder
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Compute centroid of the contour
        M = cv2.moments(largest)
        if M['m00'] == 0:
            # Degenerate case – skip this frame
            self.cmd_pub.publish(twist)
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Horizontal error (pixels) relative to image center
        error_x = cx - w / 2.0

        if area < self.min_area_to_stop:
            # Still far: move forward and turn to center cylinder

            # Angular velocity to center the object
            twist.angular.z = -self.angular_gain * error_x

            # Linear speed: faster when far, slower when close
            area_error = self.min_area_to_stop - area
            v = self.linear_gain * area_error
            v = max(0.0, min(self.max_linear_speed, v))
            twist.linear.x = v

            self.get_logger().debug(
                f"Approaching. Area={area:.1f}, v={twist.linear.x:.3f}, wz={twist.angular.z:.3f}"
            )
        else:
            # Area big enough: stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(f"Cylinder reached. Area={area:.1f} >= {self.min_area_to_stop}")

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ApproachCylinderCamera()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
