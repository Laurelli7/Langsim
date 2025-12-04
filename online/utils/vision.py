import numpy as np
import cv2
import io
import base64
from PIL import Image


def project_world_to_screen(robot_pos, cam):
    """Projects a 3D world coordinate to 2D screen pixel coordinates."""

    # Project to pixel coordinates
    points_3d = np.array([robot_pos])  # shape (N, 3) with N=1
    uv = cam.get_image_coords_from_world_points(points_3d)
    u, v = uv[0]  # pixel coords in [0,width]x[0,height]

    return u, v


def draw_robot_direction(top_img, robot_pos, robot_quat, cam):
    """Draws an arrow on the top-down image indicating the robot's position and orientation."""
    u, v = project_world_to_screen(robot_pos, cam)

    # Compute heading direction
    heading_vector = np.array([1.0, 0.0, 0.0])
    R = cv2.Rodrigues(np.array([robot_quat[0], robot_quat[1], robot_quat[2]]))[0]
    heading_world = R @ heading_vector
    heading_end_pos = robot_pos + heading_world * 0.5  # Scale for visibility
    u_end, v_end = project_world_to_screen(heading_end_pos, cam)

    # Draw arrow
    img_with_arrow = top_img.copy()
    start_point = (int(u), int(v))
    end_point = (int(u_end), int(v_end))
    cv2.arrowedLine(
        img_with_arrow,
        start_point,
        end_point,
        color=(255, 0, 0),
        thickness=2,
        tipLength=0.2,
    )

    return img_with_arrow


def encode_image_to_base64(numpy_img):
    """Converts a numpy image to a base64 string."""
    if numpy_img.shape[-1] == 4:
        numpy_img = numpy_img[:, :, :3]

    numpy_img = np.ascontiguousarray(numpy_img)
    img = Image.fromarray(numpy_img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
