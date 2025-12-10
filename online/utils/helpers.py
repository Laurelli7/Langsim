import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

COLOR_BOUNDS = {
    "aqua": {"lower": (70, 205, 205), "upper": (110, 255, 255)},
    "black": {"lower": (0, 0, 0), "upper": (180, 255, 80)},
    "blue": {"lower": (100, 205, 205), "upper": (140, 255, 255)},
    "fuchsia": {"lower": (130, 205, 205), "upper": (170, 255, 255)},
    "green": {"lower": (40, 105, 0), "upper": (80, 255, 178)},
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
    "yellow": {"lower": (10, 205, 205), "upper": (60, 255, 255)},
}


def move_forward(cmd_vel_pub, speed=0.2):
    """
    Issue a single forward command.

    Time/duration is handled by the caller: call this at your desired rate
    for as long as you want to move forward.

    Args:
        cmd_vel_pub: ROS2 publisher for geometry_msgs/Twist.
        speed: forward linear speed (m/s, positive = forward).
    """
    twist = Twist()
    twist.linear.x = float(speed)
    twist.angular.z = 0.0
    cmd_vel_pub.publish(twist)


def rotate(cmd_vel_pub, angular_speed=0.5, clockwise=True):
    """
    Issue a single rotation command.

    Time/duration is handled by the caller: call this at your desired rate
    for as long as you want to rotate.

    Args:
        cmd_vel_pub: ROS2 publisher for geometry_msgs/Twist.
        angular_speed: angular speed (rad/s, positive magnitude).
        clockwise: if True, rotate clockwise; else counter-clockwise.
    """
    twist = Twist()
    twist.linear.x = 0.0
    twist.angular.z = -abs(angular_speed) if clockwise else abs(angular_speed)
    cmd_vel_pub.publish(twist)


def stop(cmd_vel_pub):
    """
    Convenience function: stop all motion.
    """
    twist = Twist()
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    cmd_vel_pub.publish(twist)


def find_colored_blob(cv_image, lower_hsv, upper_hsv, min_area=500):
    """
    Find the largest blob within an HSV range.

    Args:
        cv_image: BGR OpenCV image (from cv_bridge).
        lower_hsv: np.array([H, S, V]) lower bound.
        upper_hsv: np.array([H, S, V]) upper bound.
        min_area: minimum contour area to consider a valid detection.

    Returns:
        (found, cx, cy, area)
        found: bool, True if a blob is found.
        cx, cy: centroid pixel coordinates (ints) of largest blob.
        area: contour area (float) of largest blob.
    """
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Optional: morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return False, None, None, area

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return False, None, None, area

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return True, cx, cy, area


def rotate_and_search_step(
    cmd_vel_pub,
    get_latest_image,
    lower_hsv,
    upper_hsv,
    angular_speed=0.4,
    clockwise=False,
    min_area=800,
    active=True,
):
    """
    ONE CONTROL STEP: Rotate in place while searching for a colored object.

    This function is non-blocking and does NOT handle time.
    Call it repeatedly from your main loop (or ROS timer) while you are
    in a "search" state. The caller decides when to stop based on its own
    timing or logic.

    Args:
        cmd_vel_pub: ROS2 publisher for Twist.
        get_latest_image: callable -> returns cv_image or None.
        lower_hsv, upper_hsv: color bounds for detection.
        angular_speed: rotation speed (rad/s).
        clockwise: rotation direction.
        min_area: minimum blob area to accept a detection.
        active: if False, this will command zero velocity.

    Returns:
        (found, cx, area)
        found: bool
        cx: horizontal pixel coordinate of blob center (or None)
        area: blob area (or 0)
    """
    twist = Twist()

    if not active:
        # Ensure we are stopped if search is inactive
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        return False, None, 0.0

    # Default: rotate
    twist.linear.x = 0.0
    twist.angular.z = -abs(angular_speed) if clockwise else abs(angular_speed)

    found = False
    cx = None
    area = 0.0

    cv_image = get_latest_image()
    if cv_image is not None:
        found, cx, cy, area = find_colored_blob(
            cv_image, lower_hsv, upper_hsv, min_area
        )
        if found:
            # We detected the target: stop rotation and let caller switch state
            twist.angular.z = 0.0

    cmd_vel_pub.publish(twist)
    return found, cx, area


def approach_colored_object_step(
    cmd_vel_pub,
    get_latest_image,
    lower_hsv,
    upper_hsv,
    image_width,
    target_area=36000,
    linear_speed=0.18,
    k_angular=0.004,
    min_area=800,
    active=True,
):
    """
    ONE CONTROL STEP: Approach a colored object while keeping it centered.

    This function is non-blocking and does NOT handle timeouts or durations.
    The caller is responsible for:
      - calling this at a fixed rate,
      - enforcing a max_duration / watchdog, and
      - deciding what to do if 'reached' never becomes True.

    Args:
        cmd_vel_pub: ROS2 publisher for Twist.
        get_latest_image: callable -> returns cv_image or None.
        lower_hsv, upper_hsv: HSV bounds for the target color.
        image_width: width of the image in pixels.
        target_area: when blob area >= this, we consider it "reached".
        linear_speed: forward speed.
        k_angular: proportional gain for angular correction.
        min_area: minimum area to treat as valid detection.
        active: if False, this will command zero velocity.

    Returns:
        (reached, found, area)
        reached: bool, True if target_area reached on this step.
        found: bool, True if target detected on this step.
        area: float, blob area (0 if not found).
    """
    twist = Twist()

    if not active:
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        return False, False, 0.0

    cv_image = get_latest_image()
    if cv_image is None:
        # No image yet, just stop this step
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        return False, False, 0.0

    found, cx, cy, area = find_colored_blob(cv_image, lower_hsv, upper_hsv, min_area)

    if not found:
        # Lost target: stop for this step
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        return False, False, 0.0

    # Check if we’re close enough
    if area >= target_area:
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        return True, True, area

    # Steering: error is horizontal offset from center
    center_x = image_width / 2.0
    error_x = cx - center_x
    angular_z = -k_angular * error_x  # negative to turn toward the error

    twist.linear.x = linear_speed
    twist.angular.z = float(angular_z)

    cmd_vel_pub.publish(twist)

    return False, True, area
