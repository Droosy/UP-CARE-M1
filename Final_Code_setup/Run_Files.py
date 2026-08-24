"""
Launches code.py (sensor logger) and Rtsp_zone_tracker_updated2.py (camera
tracker) together, and stops both cleanly on a single Ctrl+C.

Run this instead of running the two scripts separately:
    python run_both.py

Place this file in the same folder as code.py and Rtsp_zone_tracker_updated2.py
(D:\\CoE 199\\Final_Code_setup), or adjust the paths below.
"""

import subprocess
import sys
import time
import signal

PYTHON = sys.executable  # uses whatever python/venv you're running this launcher with

# Adjust paths/args as needed
SENSOR_SCRIPT = [r"D:\CoE 199\Final_Code_setup\Final_V1.py"]
TRACKER_SCRIPT = [r"D:\CoE 199\Extra codes\199 occ count code with masking\rtsp_zone_tracker_updated2.py"]


def main():
    print("Starting sensor logger (code.py)...")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    sensor_proc = subprocess.Popen([PYTHON] + SENSOR_SCRIPT, creationflags=creationflags)

    print("Starting camera tracker (Rtsp_zone_tracker_updated2.py)...")
    tracker_proc = subprocess.Popen([PYTHON] + TRACKER_SCRIPT, creationflags=creationflags)

    procs = [sensor_proc, tracker_proc]

    try:
        # Just wait until either process exits on its own, or user hits Ctrl+C
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"\nProcess {p.args} exited on its own (code {p.returncode}). Stopping the other one too.")
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping both scripts...")
        for p in procs:
            if p.poll() is None:  # still running
                try:
                    if sys.platform == "win32":
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        p.send_signal(signal.SIGINT)
                except Exception as e:
                    print(f"Could not send stop signal to {p.args}: {e}")

        # Give them a few seconds to shut down gracefully (your scripts already
        # handle KeyboardInterrupt / Ctrl+C for cleanup)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing {p.args}...")
                p.kill()

        print("Both stopped.")


if __name__ == "__main__":
    main()