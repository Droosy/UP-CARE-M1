"""
Launches code.py (sensor logger) and Rtsp_zone_tracker_updated2.py (camera
tracker) together, and stops both cleanly on a single Ctrl+C.

If either script crashes or exits on its own, it is automatically restarted
in place -- the other script keeps running undisturbed. A script that keeps
crashing immediately (a "crash loop") is retried with growing backoff delays
(2s, 4s, 8s, ... up to 60s) instead of being restarted in a tight loop; if a
restarted script manages to stay up for a while, the backoff resets back to
2s the next time it dies, since that's treated as a fresh, unrelated problem
rather than a continuation of the crash loop.

Run this instead of running the two scripts separately:
    python Run_Files.py

Place this file in the same folder as code.py and Rtsp_zone_tracker_updated2.py
(D:\\CoE 199\\Final_Code_setup), or adjust the paths below.
"""

import datetime
import signal
import subprocess
import sys
import time

PYTHON = sys.executable  # uses whatever python/venv you're running this launcher with

# Adjust paths/args as needed
SENSOR_SCRIPT = [r"C:\PlatformIO\Projects\Final_Code_setup\Final_V3.py"]
TRACKER_SCRIPT = [r"C:\PlatformIO\Projects\Final_Code_setup\199 occ count code with masking\rtsp_zone_tracker_updated4.py"]

SCRIPTS = {
    "sensor logger": SENSOR_SCRIPT,
    "camera tracker": TRACKER_SCRIPT,
}

# --- restart behavior -------------------------------------------------------
# If a script stays up at least this long before dying, the next crash is
# treated as a fresh issue and the backoff delay resets to INITIAL_BACKOFF.
MIN_UPTIME_FOR_HEALTHY = 30.0
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 60.0


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


class ManagedProcess:
    """One launched script, plus enough bookkeeping to detect crashes and
    restart it on its own without touching the other managed process."""

    def __init__(self, name, cmd):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.start_time = None
        self.backoff = INITIAL_BACKOFF
        self.restart_count = 0

    def start(self):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([PYTHON] + self.cmd, creationflags=creationflags)
        self.start_time = time.time()
        print(f"[{ts()}] Started {self.name} (pid={self.proc.pid}).")

    def poll(self):
        """Returns the exit code if the process has ended, else None."""
        return self.proc.poll() if self.proc is not None else None

    def uptime(self):
        return time.time() - self.start_time if self.start_time else 0.0

    def stop(self, timeout=10):
        """Ask the process to shut down gracefully (Ctrl+C-equivalent), then force-kill if it doesn't."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
        except Exception as e:
            print(f"Could not send stop signal to {self.name}: {e}")
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Force killing {self.name}...")
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def handle_exit(m: ManagedProcess):
    """A managed process just died. Log it, apply crash-loop backoff if it
    died quickly, then restart it. Never touches any other process."""
    code = m.poll()
    uptime = m.uptime()
    print(f"[{ts()}] {m.name} exited (code={code}) after {uptime:.1f}s uptime.")

    if uptime >= MIN_UPTIME_FOR_HEALTHY:
        # Ran fine for a while before dying -- treat as a fresh problem, not
        # a continuation of a crash loop, so restart immediately next time.
        m.backoff = INITIAL_BACKOFF
    else:
        print(f"    {m.name} died quickly (possible crash loop) -- "
              f"waiting {m.backoff:.0f}s before restarting it.")
        time.sleep(m.backoff)
        m.backoff = min(MAX_BACKOFF, m.backoff * 2)

    m.restart_count += 1
    print(f"[{ts()}] Restarting {m.name} (restart #{m.restart_count})...")
    m.start()


def main():
    managed = [ManagedProcess(name, cmd) for name, cmd in SCRIPTS.items()]

    print("Starting sensor logger and camera tracker...")
    for m in managed:
        m.start()

    try:
        # Watch both processes. If one exits (crash or otherwise), restart
        # just that one -- the other keeps running the whole time. The only
        # way both stop is Ctrl+C (below).
        while True:
            for m in managed:
                if m.poll() is not None:
                    handle_exit(m)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C received -- stopping both scripts...")
        for m in managed:
            m.stop()
        print("Both stopped.")


if __name__ == "__main__":
    main()
