import subprocess
import os
import sys


def execute_ros2_code(code_str, filename="generated_code.py", timeout_sec=120.0):
    """
    Run a ROS 2 script in a subprocess.
    While it runs, keep stepping the Isaac simulation via `sim.step()`.

    Args:
        code_str: The generated Python ROS 2 code (must import/use rclpy).
        sim:      Your SimManager / world wrapper that has a .step() method.
        filename: Where to write the temporary ROS 2 script.
        timeout_sec: Max wall-time to let the ROS 2 process run.

    Returns:
        "Success", "Code timed out", or "Code failed: <err>".
    """
    if not code_str or "rclpy" not in code_str:
        return "Error: Invalid ROS2 code."

    # Write code to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code_str)

    print(">>> Executing ROS 2 Code...")
    
    env = os.environ.copy()

    try:
        subprocess.run(
            [sys.executable, filename],
            text=True,
            timeout=timeout_sec,
            check=True,
            env=env,
        )

    except Exception as e:
        return f"Code failed: {e}"
