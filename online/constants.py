import os
from openai import OpenAI

# Global Constants
MAX_ROUNDS = 5
OUTPUT_DIR = "dataset_logs"
SCENE_DIR = "/home/dotin13/isaac-proj/scene_snapshots"

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen3-vl-8b"

# BASE_URL = None
# MODEL = "gpt-5-mini"

# Initialize Client
try:
    OPENAI_CLIENT = OpenAI(base_url=BASE_URL)
except Exception:
    OPENAI_CLIENT = None
    print("Warning: OpenAI Client could not be initialized.")
