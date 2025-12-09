"""
generate_room_with_plot.py

Requirements:
    pip install openai pydantic matplotlib numpy python-dotenv

Make sure OPENAI_API_KEY is set in your environment (or a .env file).
"""

import os
import json
from typing import List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydantic import BaseModel, Field
from openai import OpenAI

# ---------------------------------------------------------------------
# 1. Structured output schema (Pydantic)
# ---------------------------------------------------------------------

class Pose(BaseModel):
    x: float = Field(..., description="X position in meters within the room")
    y: float = Field(..., description="Y position in meters within the room")
    yaw: float = Field(..., description="Heading in radians, 0 along +X, CCW positive")


class CylinderDimensions(BaseModel):
    radius: float
    height: float


class BoxDimensions(BaseModel):
    length_x: float
    width_y: float
    height: float


class Object(BaseModel):
    id: str
    type: Literal["cylinder", "box"]
    category: str = Field(..., description="semantic label such as wall, table, chair, target")
    color: str = Field(..., description="basic color name (e.g. red, green, blue, gray)")
    pose: Pose

    # Only one of these will be used depending on type
    cylinder: Optional[CylinderDimensions] = None
    box: Optional[BoxDimensions] = None
    is_obstacle: bool = True


class Robot(BaseModel):
    name: str
    type: Literal["differential_drive", "omni", "holonomic"]
    pose: Pose
    footprint_radius: float


class Room(BaseModel):
    frame_id: str = "map"
    name: str
    length_x: float
    width_y: float
    robot: Robot
    objects: List[Object]


class RoomConfig(BaseModel):
    room: Room


# ---------------------------------------------------------------------
# 2. Call OpenAI Responses API with structured output
# ---------------------------------------------------------------------

def generate_room_config(
    model: str = "gpt-4o-mini",
) -> RoomConfig:
    """
    Uses OpenAI structured outputs to generate a single RoomConfig instance.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_instructions = """
    You are a robotics simulation planner.
    Generate a single simple indoor room layout for a mobile robot.

    Requirements:
    - Room is axis-aligned rectangle with origin (0,0) at bottom-left.
    - Room length_x between 6 and 12 meters, width_y between 5 and 10 meters.
    - Exactly one robot.
      * Place the robot at least 0.5 m away from any wall or obstacle.
    - Exactly three cylinders with distinct 'color' fields (e.g. red, green, blue).
      * Mark these cylinders as is_obstacle = false and category = "target".
    - Include at least 2 box obstacles with categories chosen from:
      ["wall", "table", "chair", "cabinet"].
      * Mark these as is_obstacle = true.
    - All coordinates must be inside the room bounds.
    - yaw is in radians, 0 along +X, counter-clockwise positive.
    """

    # responses.parse + text_format=Pydantic model -> structured output. :contentReference[oaicite:1]{index=1}
    response = client.responses.parse(
        model=model,
        instructions=system_instructions,
        input=[
            {
                "role": "user",
                "content": "Produce one valid room configuration in the RoomConfig schema.",
            }
        ],
        text_format=RoomConfig,  # this tells SDK to use structured outputs
        max_output_tokens=1024,
    )

    # response.output_parsed is already a RoomConfig instance
    return response.output_parsed


# ---------------------------------------------------------------------
# 3. Plotting utilities (top-down view with robot yaw)
# ---------------------------------------------------------------------

def plot_room(room_cfg: RoomConfig, show: bool = True, save_path: Optional[str] = None):
    room = room_cfg.room
    robot = room.robot

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    # Room boundary
    rect = patches.Rectangle(
        (0.0, 0.0),
        room.length_x,
        room.width_y,
        fill=False,
        linewidth=1.5,
    )
    ax.add_patch(rect)

    # Robot footprint
    ax.add_patch(
        patches.Circle(
            (robot.pose.x, robot.pose.y),
            radius=robot.footprint_radius,
            fill=False,
            linewidth=1.5,
        )
    )
    ax.text(robot.pose.x, robot.pose.y + 0.3, robot.name, ha="center")

    # Robot yaw as an arrow (length proportional to footprint)
    arrow_len = robot.footprint_radius * 2.0
    dx = arrow_len * np.cos(robot.pose.yaw)
    dy = arrow_len * np.sin(robot.pose.yaw)
    ax.arrow(
        robot.pose.x,
        robot.pose.y,
        dx,
        dy,
        head_width=robot.footprint_radius * 0.6,
        length_includes_head=True,
    )

    # Objects
    for obj in room.objects:
        if obj.type == "cylinder" and obj.cylinder is not None:
            ax.add_patch(
                patches.Circle(
                    (obj.pose.x, obj.pose.y),
                    radius=obj.cylinder.radius,
                    fill=False,
                    linewidth=1.0,
                )
            )
        elif obj.type == "box" and obj.box is not None:
            ax.add_patch(
                patches.Rectangle(
                    (
                        obj.pose.x - obj.box.length_x / 2.0,
                        obj.pose.y - obj.box.width_y / 2.0,
                    ),
                    obj.box.length_x,
                    obj.box.width_y,
                    fill=False,
                    linewidth=1.0,
                )
            )

        label = f"{obj.category} ({obj.id})"
        ax.text(obj.pose.x, obj.pose.y + 0.2, label, ha="center")

    ax.set_xlim(-0.5, room.length_x + 0.5)
    ax.set_ylim(-0.5, room.width_y + 0.5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Top-down layout: {room.name}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. End-to-end entry point
# ---------------------------------------------------------------------

def main():
    output_dir = "scene_json"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        cfg = generate_room_config()
        with open(f"{output_dir}/{i}.json", "w") as f:
            f.write(json.dumps(cfg.model_dump()))
            
        plot_room(cfg, show=False, save_path=f"{output_dir}/{i}.png")


if __name__ == "__main__":
    main()
