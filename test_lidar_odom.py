#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


class ScanOdomInspector(Node):
    def __init__(self):
        super().__init__('scan_odom_inspector')

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',          # change if your topic is different
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',          # change if your topic is different
            self.odom_callback,
            10
        )

        self.get_logger().info("ScanOdomInspector node started. Listening to /scan and /odom")

    # ---- LaserScan callback ----
    def scan_callback(self, msg: LaserScan):
        # Basic info
        num_points = len(msg.ranges)
        stamp = msg.header.stamp

        # Example: get the front measurement (approx middle index)
        if num_points > 0:
            mid_index = 1080 // 4
            front_distance = msg.ranges[mid_index]
        else:
            front_distance = float('nan')

        # Print a small summary (not spamming everything)
        self.get_logger().info(
            f"[LaserScan] t={stamp.sec}.{stamp.nanosec:09d}  "
            f"points={num_points}  "
            f"angle_min={msg.angle_min:.3f}  angle_max={msg.angle_max:.3f}  "
            f"front_dist={front_distance:.3f} m"
        )

        # If you want to inspect a few specific rays, uncomment:
        # for i in range(0, num_points, max(1, num_points // 10)):
        #     angle = msg.angle_min + i * msg.angle_increment
        #     r = msg.ranges[i]
        #     self.get_logger().info(f"  index={i:3d}, angle={angle:.3f} rad, range={r:.3f} m")

    # ---- Odometry callback ----
    def odom_callback(self, msg: Odometry):
        stamp = msg.header.stamp

        # Position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Orientation (quaternion -> yaw)
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        # Linear and angular velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        self.get_logger().info(
            f"[Odom] t={stamp.sec}.{stamp.nanosec:09d}  "
            f"pos=({x:.3f}, {y:.3f})  yaw={yaw:.3f} rad  "
            f"v=({vx:.3f}, {vy:.3f}) m/s  wz={wz:.3f} rad/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ScanOdomInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
