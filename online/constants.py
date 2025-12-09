import os
from openai import OpenAI

# Environment Variables
ENV_CONFIG = {
    "ROBOT_FRONT_ANGLE": os.getenv("ROBOT_FRONT_ANGLE", "0.0"),
    "TOPIC_CMD_VEL": os.getenv("TOPIC_CMD_VEL", "/cmd_vel"),
    "TOPIC_SCAN": os.getenv("TOPIC_SCAN", "/scan"),
    "TOPIC_CAMERA_RGB": os.getenv("TOPIC_CAMERA_RGB", "/camera_rgb"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

# Global Constants
MAX_ROUNDS = 5
OUTPUT_DIR = "dataset_logs"
SCENE_DIR = "/home/dotin13/isaac-proj/scene_snapshots"

# Initialize Client (Lazy initialization or global access)
try:
    OPENAI_CLIENT = OpenAI()
except Exception:
    OPENAI_CLIENT = None
    print("Warning: OpenAI Client could not be initialized.")