import subprocess
import os
import sys
import time
import signal
import select
import termios
import tty


class NonBlockingConsole:
    """
    Context manager to switch terminal to raw mode (read keys instantly)
    and restore it afterwards.
    """

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def is_data(self):
        """Checks if there is input waiting on stdin."""
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def read_key(self):
        """Reads a single character."""
        if self.is_data():
            return sys.stdin.read(1)
        return None


def execute_ros2_code(code_str, filename="generated_code.py", timeout_sec=120.0):
    """
    Run a ROS 2 script in a subprocess.
    Allows pressing 'q' to stop execution gracefully.
    """
    if not code_str or "rclpy" not in code_str:
        return "Error: Invalid ROS2 code."

    # Write code to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code_str)

    print(">>> Executing ROS 2 Code. Press 'q' to stop early...")

    env = os.environ.copy()

    # Start the process non-blocking
    proc = subprocess.Popen(
        [sys.executable, filename], text=True, env=env, preexec_fn=os.setsid
    )

    start_time = time.time()
    result_msg = "Success"

    try:
        # Use our context manager to handle raw input
        with NonBlockingConsole() as console:
            while True:
                # 1. Check if process is still running
                ret_code = proc.poll()
                if ret_code is not None:
                    # Process finished naturally
                    if ret_code != 0:
                        result_msg = f"Code failed with return code {ret_code}"
                    break

                # 2. Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    print("\n[Host] Timeout reached. Killing process...")
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    result_msg = "Code timed out."
                    break

                # 3. Check for 'q' key
                key = console.read_key()
                if key == "q":
                    print("\n[Host] 'q' pressed. Stopping process...")
                    # Send SIGINT (Ctrl+C) to the subprocess group
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    result_msg = "Execution interrupted by user."

                    # Wait a moment for it to shut down cleanly
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    break

                # Small sleep to prevent CPU spiking
                time.sleep(0.1)

    except Exception as e:
        # Failsafe: ensure process is dead if Python crashes
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return f"Code failed: {e}"

    return result_msg
