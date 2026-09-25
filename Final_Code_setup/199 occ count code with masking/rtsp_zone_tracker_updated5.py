#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# IMPORTANT: this must be set BEFORE cv2 is imported / any VideoCapture is
# created. Without a socket timeout, OpenCV's FFMPEG backend will block
# forever inside cap.read() when an RTSP stream stalls (camera reboots, PoE
# blip, Wi-Fi drop). The capture thread then hangs silently and the main loop
# keeps reporting the last count it ever saw -- this is the classic "count is
# frozen at 27 for hours" symptom.
#   rtsp_transport;tcp  -> TCP is far more reliable than UDP for H.264 RTSP
#   stimeout / timeout  -> socket read timeout in MICROseconds (5s here)
# 'stimeout' is the old FFmpeg name, 'timeout' the newer one; setting both
# keeps this working across OpenCV/FFmpeg builds.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|timeout;5000000|max_delay;500000",
)

import cv2
import numpy as np

# Try to import torch for device detection; if unavailable, we fall back to CPU.
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except ImportError as e:
    print("Please install ultralytics: pip install ultralytics. Error:", e)
    sys.exit(1)


# Camera aliases (user-provided RTSP links)
CAM_SOURCES = {
    "cam1": "rtsp://admin:++smartilab2023@10.158.71.241:554/Streaming/channels/101",
    "cam2": "rtsp://admin:++smartilab2023@10.158.71.240:554/Streaming/channels/101",
}

# Camera aliases (vids)
#CAM_SOURCES = {
#    "cam1": r"C:/Users/julia/yolo_project/videos/Mar30_front_cam.mp4",
#    "cam2": r"C:/Users/julia/yolo_project/videos/Mar30_rear_cam.mp4",
#}

# ---------------------------------------------------------------------------
# Zone mask configuration.
#
# Each camera now has TWO zones, each defined by its own white-on-black mask
# PNG (white = inside the zone). Zones carry globally unique labels so the
# room total is unambiguous:
#
#     cam1 -> zone1, zone2
#     cam2 -> zone3, zone4
#
# The order within each camera's list is also the DE-DUPLICATION PRIORITY: if
# a single detected person falls inside more than one of that camera's zone
# masks (i.e. the masks overlap), they are counted ONCE, in the first
# matching zone in this list. See assign_boxes_to_zones().
#
# Each entry is (zone_label, mask_path). Paths may be absolute, relative, or
# None to fall back to scanning the ./masks/ directory next to this script.
ZONE_MASKS = {
    "cam1": [
        ("zone1", r"C:/Users/julia/yolo_project/zone-tracker/masks/cam1-final_zone1.png"),
        ("zone3", r"C:/Users/julia/yolo_project/zone-tracker/masks/cam1-final_zone3.png"),
    ],
    "cam2": [
        ("zone2", r"C:/Users/julia/yolo_project/zone-tracker/masks/cam2-final_zone2.png"),
        ("zone4", r"C:/Users/julia/yolo_project/zone-tracker/masks/cam2-final_zone4.png"),
    ],
}

# Flat, ordered list of every zone label -- used for CSV columns, JSON keys
# and console output so the ordering is consistent everywhere.
ALL_ZONE_LABELS = [label for cam in ("cam1", "cam2") for label, _ in ZONE_MASKS[cam]]

# On-screen color per zone (BGR), used when --display or --out is active.
ZONE_COLORS = {
    "zone1": (255, 128, 0),    # blue
    "zone2": (0, 200, 255),    # amber
    "zone3": (0, 220, 0),      # green
    "zone4": (200, 0, 255),    # magenta
}
OUT_OF_ZONE_COLOR = (0, 0, 255)  # red: detected but in no zone -> not counted

# ---------------------------------------------------------------------------
# CSV logging configuration
# Flip ENABLE_CSV_LOGGING to True/False here to turn CSV output on/off by
# default with a single line of code. This is also exposed as --csv-output
# on the command line (which overrides this default), and the destination
# can be overridden with --csv-path.
ENABLE_CSV_LOGGING = False
CSV_OUTPUT_PATH = r"C:/Users/julia/yolo_project/zone-tracker/CSV results/results mar 30.csv"

# ---------------------------------------------------------------------------
# JSON "bridge" file configuration
# Flip ENABLE_JSON_BRIDGE to True/False here to turn the JSON bridge file
# on/off by default. This is also exposed as --json-output on the command
# line (which overrides this default), and the destination can be overridden
# with --json-path. Must match whatever path the downstream reader
# (e.g. code.py) expects.
ENABLE_JSON_BRIDGE = True
PERSON_COUNT_FILE = r"D:\CoE 199\Final_Code_setup\person_count_latest.json"

# ---------------------------------------------------------------------------
# Processing mode
# "live"     - RTSP / live camera streams. Summary rows (console + CSV) are
#              emitted every N seconds, controlled by --output-interval.
# "recorded" - Pre-recorded video files. Exactly 1 of every N frames is
#              decoded and analyzed, controlled by --frame-interval; a
#              summary/CSV row is emitted for each sampled frame.
# This is also exposed as --mode on the command line.
DEFAULT_MODE = "live"


@dataclass
class ZoneState:
    """One counting zone: a labelled binary mask plus its latest count."""
    label: str                                # globally unique, e.g. "zone1"
    mask_path: Optional[str] = None
    mask_bin: Optional[np.ndarray] = None     # binary mask resized to camera frame
    last_count: int = 0


@dataclass
class CameraState:
    name: str
    source: str
    cap: Optional[cv2.VideoCapture] = None
    frame_q: Queue = field(default_factory=lambda: Queue(maxsize=1))
    # Each camera now owns multiple zones instead of a single mask. Order in
    # this list is the de-duplication priority (see assign_boxes_to_zones).
    zones: list = field(default_factory=list)
    last_frame_counts: int = 0                # per-camera total across its zones
    last_zone_counts: dict = field(default_factory=dict)   # {zone_label: count}
    last_detection_time: float = 0.0
    last_output_counts: Optional[int] = None
    last_inference_ms: float = 0.0
    conf: float = 0.25
    imgsz: int = 1920
    stop_flag: threading.Event = field(default_factory=threading.Event)

    # --- stream health / freshness tracking (live mode) ---------------------
    # Wall-clock (time.time()) of the last frame actually pulled off the
    # camera, and of the last detection actually run. These are what let the
    # main loop tell "count really is 27" apart from "the stream died and 27
    # is just the last number we ever saw".
    last_frame_wall: float = 0.0
    last_detection_wall: float = 0.0
    frames_captured: int = 0          # monotonically increasing frame counter
    frames_at_last_summary: int = 0   # snapshot, to report per-interval throughput
    consecutive_read_failures: int = 0
    reconnects: int = 0
    # Set by the watchdog to ask the capture thread to tear down and reopen.
    force_reconnect: threading.Event = field(default_factory=threading.Event)
    cap_lock: threading.Lock = field(default_factory=threading.Lock)

    def seconds_since_frame(self) -> float:
        """Wall-clock age of the newest frame captured, or a huge number if none yet."""
        if self.last_frame_wall <= 0:
            return float("inf")
        return time.time() - self.last_frame_wall

    def seconds_since_detection(self) -> float:
        """Wall-clock age of the newest detection result, or a huge number if none yet."""
        if self.last_detection_wall <= 0:
            return float("inf")
        return time.time() - self.last_detection_wall


def str2bool(value: str) -> bool:
    """
    Parse a true/false-ish command line string into a bool. Accepts
    true/false, yes/no, y/n, t/f, 1/0 (case-insensitive).
    """
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value!r}")


def parse_args():
    p = argparse.ArgumentParser(description="Dual-camera detection using white masks.")
    p.add_argument("--model", default="yolov5nu.pt",
                   help="YOLO model path (default: yolov5n.pt)")
    p.add_argument("--zone1", help="Override mask PNG for zone1 (cam1). White area = in-zone.")
    p.add_argument("--zone2", help="Override mask PNG for zone2 (cam2). White area = in-zone.")
    p.add_argument("--zone3", help="Override mask PNG for zone3 (cam1). White area = in-zone.")
    p.add_argument("--zone4", help="Override mask PNG for zone4 (cam2). White area = in-zone.")
    p.add_argument("--out", help="Output video file path (optional).")
    p.add_argument("--display", action="store_true", help="Show live windows.")
    p.add_argument("--imgsz", type=int, default=1920, help="Inference image size (default: 1920)")
    p.add_argument("--conf1", type=float, default=0.1, help="Confidence threshold for cam1 (default 0.25)")
    p.add_argument("--conf2", type=float, default=0.05, help="Confidence threshold for cam2 (default 0.10)")
    p.add_argument("--device", default="0", help="Device for inference (0 for GPU0, cpu for CPU).")
    p.add_argument("--output-interval", type=float, default=10.0,
                   help="Live mode only: seconds between summary outputs (default 10s)")
    p.add_argument("--max-persons", type=int, default=33, help="Max detections per frame")

    # Live vs. recorded-footage processing mode
    p.add_argument("--mode", choices=["live", "recorded"], default=DEFAULT_MODE,
                   help="'live' processes RTSP/live streams and reports every --output-interval "
                        "seconds. 'recorded' deterministically analyzes exactly 1 of every "
                        f"--frame-interval frames of a saved video file. (default: {DEFAULT_MODE})")
    p.add_argument("--frame-interval", type=int, default=5,
                   help="Recorded mode only: analyze 1 out of every N frames per camera "
                        "(e.g. 30 -> frames 0, 30, 60, ...). A summary/CSV row is written for "
                        "each analyzed frame. (default: 30)")

    # --- stream health / anti-stale-count options (live mode) --------------
    p.add_argument("--stale-timeout", type=float, default=15.0,
                   help="Live mode: if a camera has produced no NEW detection within this many "
                        "seconds, its count is considered stale and is no longer reported as a "
                        "real occupancy value. Prevents the count freezing at an old number when "
                        "a stream dies. (default: 15s)")
    p.add_argument("--stale-behavior", choices=["zero", "hold", "last"], default="zero",
                   help="Live mode: what to report for a camera whose data has gone stale. "
                        "'zero' = report 0 for that camera (safest: a dead stream can't hold the "
                        "count up forever). 'hold'/'last' = keep reporting its last known value "
                        "(the old behavior). (default: zero)")
    p.add_argument("--reconnect-after", type=float, default=10.0,
                   help="Live mode: if no frame arrives from a camera within this many seconds, "
                        "tear down and reopen its RTSP connection. (default: 10s)")
    p.add_argument("--max-read-failures", type=int, default=20,
                   help="Live mode: consecutive failed cap.read() calls before forcing a "
                        "reconnect. (default: 20)")

    # CSV logging (timestamps, per-camera counts, fps, inference speed)
    p.add_argument("--csv-output", type=str2bool, default=ENABLE_CSV_LOGGING, metavar="{true,false}",
                   help=f"Log timestamp/counts/fps/inference-speed rows to a CSV file (default: {ENABLE_CSV_LOGGING})")
    p.add_argument("--csv-path", default=CSV_OUTPUT_PATH,
                   help=f"Destination CSV file when --csv-output is true (default: {CSV_OUTPUT_PATH})")

    # JSON bridge file (latest combined count only, for other scripts to poll)
    p.add_argument("--json-output", type=str2bool, default=ENABLE_JSON_BRIDGE, metavar="{true,false}",
                   help=f"Write the latest combined person count to a JSON bridge file on every "
                        f"summary row (default: {ENABLE_JSON_BRIDGE})")
    p.add_argument("--json-path", default=PERSON_COUNT_FILE,
                   help=f"Destination JSON file when --json-output is true (default: {PERSON_COUNT_FILE})")
    return p.parse_args()


def resolve_device(value: str):
    """
    Resolve the --device argument into a value acceptable to ultralytics:
      - 'cpu' -> 'cpu'
      - '0' (digit) -> int(0) if CUDA available, else 'cpu' (with warning)
      - other strings are returned as-is
    This prevents ValueError when users pass --device 0 on a machine without CUDA.
    """
    s = value.strip().lower()
    # explicit cpu requested
    if s == "cpu":
        return "cpu"
    # numeric requested (GPU index)
    if s.isdigit():
        gpu_index = int(s)
        # If torch is available, check for CUDA devices
        if torch is not None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Use the numeric GPU index
                return gpu_index
            else:
                warnings.warn(
                    f"No CUDA devices detected (torch.cuda.is_available()={torch.cuda.is_available()}). "
                    "Falling back to CPU. To force CPU explicitly, pass --device cpu."
                )
                return "cpu"
        else:
            warnings.warn(
                "PyTorch not importable; falling back to CPU. Install torch for GPU support."
            )
            return "cpu"
    # otherwise return as-is (e.g., '0,1' for multi-gpu or 'cuda:0' style if user passed it)
    return value


def load_mask_binary(path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Load a mask image (white on black). Resize to target_hw (h, w) and return binary uint8 mask (H,W) with 255 for white.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found or cannot be decoded: {path}")
    H, W = target_hw
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    _, binm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binm


def center_in_mask_count(mask_bin: np.ndarray, boxes_xyxy: np.ndarray, overlap_thresh: float = 0.0) -> int:
    """
    Count detections whose center point falls into the binary mask (nonzero).
    """
    H, W = mask_bin.shape[:2]
    count = 0
    for (x1, y1, x2, y2) in boxes_xyxy:
        bx1 = max(0, int(x1)); by1 = max(0, int(y1))
        bx2 = min(W - 1, int(x2)); by2 = min(H - 1, int(y2))
        if overlap_thresh <= 0.0:
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            cx = max(0, min(W - 1, cx))
            cy = max(0, min(H - 1, cy))
            if mask_bin[cy, cx] > 0:
                count += 1
        else:
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            roi = mask_bin[by1:by2, bx1:bx2]
            if roi.size == 0:
                continue
            ratio = float(np.count_nonzero(roi)) / box_area
            if ratio >= overlap_thresh:
                count += 1
    return count


def assign_boxes_to_zones(zones, boxes_xyxy):
    """
    Assign each detected box to AT MOST ONE zone, and return per-zone counts.

    This is what prevents double counting. Each detection's center point is
    tested against each zone mask in `zones` order; the FIRST zone that
    contains it wins and the search stops, so a person standing in an area
    where two masks overlap is counted exactly once (in the earlier zone in
    the list) rather than once per zone. A detection whose center falls
    outside every mask is not counted at all.

    Returns (counts, assignments):
      counts      -- {zone_label: int}, one entry per zone
      assignments -- list parallel to boxes_xyxy, each entry the winning zone
                     label or None if the box landed outside all zones
    """
    counts = {z.label: 0 for z in zones}
    assignments = []

    for (x1, y1, x2, y2) in boxes_xyxy:
        cx = (int(x1) + int(x2)) // 2
        cy = (int(y1) + int(y2)) // 2

        chosen = None
        for z in zones:
            if z.mask_bin is None:
                continue
            H, W = z.mask_bin.shape[:2]
            px = max(0, min(W - 1, cx))
            py = max(0, min(H - 1, cy))
            if z.mask_bin[py, px] > 0:
                chosen = z.label
                break  # first match wins -> counted once, never twice

        assignments.append(chosen)
        if chosen is not None:
            counts[chosen] += 1

    return counts, assignments


def camera_report(cam: CameraState, stale_timeout: float, stale_behavior: str):
    """
    Decide what counts to report for a camera, based on how fresh its last
    detection actually is.

    Returns (zone_counts, total, is_stale, age_seconds), where zone_counts is
    {zone_label: count} covering every zone this camera owns.

    This is the fix for the frozen-count bug. The old logic fell back to
    `cam.last_output_counts` whenever no new detection had happened since the
    previous summary -- which meant that once a stream died, the last number
    it ever produced was re-reported forever. Here, a camera that hasn't
    produced a detection within `stale_timeout` seconds is explicitly flagged,
    and by default contributes 0 (in every one of its zones) rather than
    propping the occupancy count up indefinitely.
    """
    age = cam.seconds_since_detection()
    is_stale = age > stale_timeout

    live_counts = {z.label: int(cam.last_zone_counts.get(z.label, 0) or 0) for z in cam.zones}

    if is_stale and stale_behavior not in ("hold", "last"):
        zeroed = {label: 0 for label in live_counts}
        return zeroed, 0, True, age

    return live_counts, sum(live_counts.values()), is_stale, age


def write_person_count_file(filepath, ts, zone_counts, cam1_total, cam2_total, combined,
                            stale1: bool = False, stale2: bool = False):
    """
    Atomically write the latest counts to a small JSON bridge file so other
    scripts (e.g. code.py) can pick them up.

    Payload now carries each zone separately plus the room total. 'combined'
    is kept as the room total so existing readers keep working unchanged.

    Uses a write-to-temp-then-rename pattern so a concurrent reader never
    sees a partially written file: os.replace() is atomic on the same
    filesystem, so any reader always sees either the previous complete file
    or the new complete file, never something half-written.
    """
    try:
        payload = {
            "timestamp": ts,
            # Per-zone counts (zone1/zone2 from cam1, zone3/zone4 from cam2).
            "zones": {label: int(zone_counts.get(label, 0)) for label in ALL_ZONE_LABELS},
            # Per-camera subtotals
            "cam1": int(cam1_total),
            "cam2": int(cam2_total),
            # Room total across all four zones (each person counted once).
            "combined": int(combined),
            "total": int(combined),
            # Freshness flags so the downstream reader can tell a real count
            # from one produced while a camera was offline. Also an epoch
            # stamp so a reader can detect that this file itself went stale
            # (e.g. this script crashed) rather than trusting it forever.
            "cam1_stale": bool(stale1),
            "cam2_stale": bool(stale2),
            "stale": bool(stale1 or stale2),
            "epoch": time.time(),
        }
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=out_dir if out_dir else None)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, filepath)  # atomic on same filesystem
    except Exception as e:
        print(f"⚠️ Could not write person count file: {e}")


def open_capture(source: str) -> Optional[cv2.VideoCapture]:
    """
    Open a VideoCapture for `source`, preferring the FFMPEG backend for RTSP
    (so the OPENCV_FFMPEG_CAPTURE_OPTIONS timeouts set at the top of this file
    actually apply) and requesting a 1-frame internal buffer so we always get
    the freshest frame rather than a growing backlog of stale ones.
    """
    is_stream = isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_stream else cv2.VideoCapture(source)
    except Exception:
        cap = cv2.VideoCapture(source)
    if cap is not None and cap.isOpened():
        # Keep OpenCV's internal buffer tiny. Without this, a slow consumer
        # causes frames to pile up and you end up doing inference on footage
        # that is many seconds old.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    if cap is not None:
        cap.release()
    return None


def camera_capture_loop(cam: CameraState, reconnect_after: float, max_read_failures: int):
    """
    Live-mode capture thread: reads frames from cam.cap and places the latest
    frame in cam.frame_q (size 1), dropping the previous one if not yet
    consumed.

    Unlike the previous version, a failed read is NOT retried forever against
    a dead capture object. After `max_read_failures` consecutive failures (or
    when the watchdog sets cam.force_reconnect), the capture is released and
    reopened. Every successful read stamps cam.last_frame_wall, which is what
    the watchdog and the summary logic use to detect a stalled stream.
    """
    backoff = 1.0
    while not cam.stop_flag.is_set():
        # (Re)open the capture if needed
        with cam.cap_lock:
            cap_ok = cam.cap is not None and cam.cap.isOpened()
        if not cap_ok:
            with cam.cap_lock:
                if cam.cap is not None:
                    try:
                        cam.cap.release()
                    except Exception:
                        pass
                    cam.cap = None
            new_cap = open_capture(cam.source)
            if new_cap is None:
                print(f"[{cam.name}] reconnect failed; retrying in {backoff:.0f}s")
                # Wait, but stay responsive to shutdown
                cam.stop_flag.wait(timeout=backoff)
                backoff = min(30.0, backoff * 2)
                continue
            with cam.cap_lock:
                cam.cap = new_cap
            cam.reconnects += 1
            cam.consecutive_read_failures = 0
            cam.force_reconnect.clear()
            backoff = 1.0
            print(f"[{cam.name}] stream (re)connected (reconnect #{cam.reconnects}).")

        # Watchdog asked us to recycle the connection
        if cam.force_reconnect.is_set():
            print(f"[{cam.name}] watchdog forced reconnect (no frames for "
                  f"{cam.seconds_since_frame():.0f}s).")
            with cam.cap_lock:
                if cam.cap is not None:
                    try:
                        cam.cap.release()
                    except Exception:
                        pass
                    cam.cap = None
            cam.force_reconnect.clear()
            continue

        with cam.cap_lock:
            cap = cam.cap
        if cap is None:
            continue

        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[{cam.name}] read() raised: {e}")
            ret, frame = False, None

        if not ret or frame is None:
            cam.consecutive_read_failures += 1
            if cam.consecutive_read_failures >= max_read_failures:
                print(f"[{cam.name}] {cam.consecutive_read_failures} consecutive read failures "
                      f"-- reconnecting.")
                with cam.cap_lock:
                    if cam.cap is not None:
                        try:
                            cam.cap.release()
                        except Exception:
                            pass
                        cam.cap = None
                continue
            # Brief pause so a hard-failing stream doesn't spin the CPU
            cam.stop_flag.wait(timeout=0.2)
            continue

        # Successful read
        cam.consecutive_read_failures = 0
        cam.last_frame_wall = time.time()
        cam.frames_captured += 1

        # Put the newest frame in the queue, dropping the older one if the
        # consumer hasn't taken it yet. Never block here: a blocked capture
        # thread is exactly what causes frozen counts.
        try:
            if cam.frame_q.full():
                try:
                    cam.frame_q.get_nowait()
                except Empty:
                    pass
            cam.frame_q.put_nowait(frame)
        except Exception:
            pass
    # release handled externally


def camera_watchdog_loop(cams, reconnect_after: float, stop_event: threading.Event):
    """
    Watchdog thread: if a camera hasn't delivered a frame in `reconnect_after`
    seconds, force its capture thread to reconnect.

    This is the backup for the case the FFMPEG socket timeout doesn't catch:
    cap.read() blocking indefinitely deep inside the decoder. Releasing the
    capture object from here unblocks that read so the capture thread can
    recover instead of hanging silently forever.
    """
    while not stop_event.wait(timeout=1.0):
        for cam in cams:
            age = cam.seconds_since_frame()
            if age > reconnect_after and not cam.force_reconnect.is_set():
                print(f"[{cam.name}] WATCHDOG: no frame for {age:.0f}s -- forcing reconnect.")
                cam.force_reconnect.set()
                # Releasing the capture unblocks a stuck read() in the capture thread.
                with cam.cap_lock:
                    if cam.cap is not None:
                        try:
                            cam.cap.release()
                        except Exception:
                            pass


def next_sampled_frame(cap: cv2.VideoCapture, frame_interval: int, frame_counter: int):
    """
    Recorded-mode helper: advance `cap` forward using grab() (cheap, no
    decode) until reaching the frame at the next 0-indexed position that's a
    multiple of frame_interval, then decode just that one frame with
    retrieve(). This gives deterministic sampling (e.g. frame_interval=30 ->
    frames 0, 30, 60, ...) without wasting time decoding skipped frames.

    frame_counter is the 0-indexed count of frames already consumed (grabbed)
    from this capture so far.

    Returns (frame_or_None, updated_frame_counter, eof_reached).
    """
    while True:
        ret = cap.grab()
        if not ret:
            return None, frame_counter, True
        is_wanted = (frame_counter % frame_interval == 0)
        frame_counter += 1
        if is_wanted:
            ok, frame = cap.retrieve()
            if not ok:
                return None, frame_counter, True
            return frame, frame_counter, False


def resolve_zone_mask_path(cam_name: str, zone_label: str, configured: Optional[str],
                           cli_override: Optional[str], script_dir: Path) -> Optional[str]:
    """
    Resolve the mask path for one zone, in priority order:
      1) explicit CLI override (--zone1 ... --zone4)
      2) the path configured in ZONE_MASKS
      3) ./masks/ next to this script: any filename mentioning BOTH the camera
         name and the zone suffix (e.g. 'cam1' + 'zone1')
      4) None if nothing found (caller handles the error)
    """
    candidates = []
    if cli_override:
        candidates.append(Path(cli_override))
    if configured:
        candidates.append(Path(configured))

    # Zone masks made by adjust_mask_gui.py are named like
    # 'cam1-mask-new_zone1.png'; the per-camera zone suffix is zone1/zone2.
    masks_dir = script_dir / "masks"
    cam_zone_suffix = "zone1" if zone_label in ("zone1", "zone3") else "zone2"
    candidates.append(masks_dir / f"{cam_name}-mask-new_{cam_zone_suffix}.png")
    if masks_dir.exists() and masks_dir.is_dir():
        for p in sorted(masks_dir.iterdir()):
            n = p.name.lower()
            if cam_name.lower() in n and cam_zone_suffix in n:
                candidates.append(p)

    print(f"Resolving mask for {cam_name}/{zone_label}. Candidates (in order):")
    for c in candidates:
        try:
            c_expanded = Path(str(c)).expanduser()
            c_try = c_expanded.resolve() if c_expanded.is_absolute() else (script_dir / c_expanded).resolve()
        except Exception:
            c_try = Path(str(c))
        print("  -", str(c_try))
        if c_try.exists():
            print(f"Resolved {cam_name}/{zone_label}: {c_try}")
            return str(c_try)

    print(f"Could not resolve mask for {cam_name}/{zone_label} "
          f"(checked {len(candidates)} candidates).")
    if masks_dir.exists() and masks_dir.is_dir():
        print(f"Files in {masks_dir}: {sorted(p.name for p in masks_dir.iterdir())}")
    else:
        print(f"No masks/ directory at {masks_dir}.")
    return None


def _run_detection(model, frame, cam, device, args):
    """
    Shared per-frame inference + visualization helper used by both modes.

    Runs YOLO ONCE on the frame, then assigns each detection to at most one of
    the camera's zones (see assign_boxes_to_zones) so overlapping masks can't
    double count the same person. Updates cam.last_zone_counts /
    cam.last_frame_counts / cam.last_inference_ms.

    Returns (zone_counts, cam_total, annotated_frame).
    """
    results = model.predict(
        source=frame,
        conf=cam.conf,
        iou=0.3,
        classes=[0],  # person
        device=device,
        imgsz=cam.imgsz,
        max_det=args.max_persons,
        verbose=False,
    )

    zone_counts = {z.label: 0 for z in cam.zones}
    vis = frame.copy()

    if len(results) > 0:
        r = results[0]

        speed = getattr(r, "speed", None)
        if speed:
            cam.last_inference_ms = float(speed.get("inference", cam.last_inference_ms))

        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = getattr(r.boxes, "xyxy", None)
            confs = getattr(r.boxes, "conf", None)
            if xyxy is not None:
                boxes = xyxy.cpu().numpy().astype(int)
                zone_counts, assignments = assign_boxes_to_zones(cam.zones, boxes)
                confs_np = confs.cpu().numpy() if confs is not None else np.ones(len(boxes))

                for (x1, y1, x2, y2), conf, zlabel in zip(boxes, confs_np, assignments):
                    # Color by the zone the detection was actually counted in;
                    # red means it fell outside every zone and wasn't counted.
                    color = ZONE_COLORS.get(zlabel, OUT_OF_ZONE_COLOR) if zlabel else OUT_OF_ZONE_COLOR
                    tag = zlabel if zlabel else "none"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{tag} {conf:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cam_total = sum(zone_counts.values())

    # Per-zone overlay banner so the split is visible on screen / in --out video
    banner = f"{cam.name}: " + "  ".join(f"{lbl}={zone_counts.get(lbl, 0)}" for lbl in
                                         [z.label for z in cam.zones]) + f"  total={cam_total}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for z in cam.zones:
        z.last_count = zone_counts.get(z.label, 0)
    cam.last_zone_counts = zone_counts
    cam.last_frame_counts = cam_total
    cam.last_detection_time = time.perf_counter()
    cam.last_detection_wall = time.time()
    return zone_counts, cam_total, vis


def _write_combined_frame(writer, vis_by_cam, cam1, cam2, h1, w1, h2, w2):
    """Compose whichever camera visualizations are available this cycle onto one canvas and write it."""
    if not writer or not vis_by_cam:
        return
    H_out, W_out = max(h1, h2), max(w1, w2)
    canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    if cam1.name in vis_by_cam:
        v = vis_by_cam[cam1.name]
        canvas[0:v.shape[0], 0:v.shape[1]] = v
    if cam2.name in vis_by_cam:
        v = vis_by_cam[cam2.name]
        x_off = W_out - v.shape[1]
        canvas[0:v.shape[0], x_off:x_off + v.shape[1]] = v
    writer.write(canvas)


def run_live_mode(cam1, cam2, model, device, args, writer, csv_writer, csv_file, output_interval, h1, w1, h2, w2):
    """
    Live/RTSP processing loop. Background capture threads (started by the
    caller) keep the newest frame per camera ready in a size-1 queue; here we
    opportunistically run inference on whichever camera has a frame ready,
    and print/log one summary row every `output_interval` seconds. If
    args.json_output is enabled, the same summary also refreshes the JSON
    bridge file.
    """
    fps_window = deque(maxlen=30)
    last_time = time.perf_counter()
    last_output_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        processed_any = False

        for cam in (cam1, cam2):
            try:
                frame = cam.frame_q.get(timeout=0.01)
            except Empty:
                continue  # no frame for this cam right now

            zone_counts, cam_total, vis = _run_detection(model, frame, cam, device, args)
            processed_any = True

            if args.display:
                cv2.imshow(cam.name, vis)

            if writer:
                _write_combined_frame(writer, {cam.name: vis}, cam1, cam2, h1, w1, h2, w2)

        now2 = time.perf_counter()
        fps_window.append(1.0 / max(1e-6, now2 - last_time))
        last_time = now2
        avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

        if now - last_output_time >= output_interval:
            # Freshness-aware reporting. A camera's counts are only reported as
            # real occupancy values if a detection actually ran recently. If the
            # stream died, we do NOT keep echoing the last numbers we ever saw.
            zc1, tot1, stale1, age1 = camera_report(cam1, args.stale_timeout, args.stale_behavior)
            zc2, tot2, stale2, age2 = camera_report(cam2, args.stale_timeout, args.stale_behavior)

            zone_counts = {}
            zone_counts.update(zc1)
            zone_counts.update(zc2)
            combined = tot1 + tot2
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Per-interval capture throughput: 0 new frames is the tell-tale
            # sign of a dead stream even before the stale timeout trips.
            new1 = cam1.frames_captured - cam1.frames_at_last_summary
            new2 = cam2.frames_captured - cam2.frames_at_last_summary
            cam1.frames_at_last_summary = cam1.frames_captured
            cam2.frames_at_last_summary = cam2.frames_captured

            tag1 = f" [STALE {age1:.0f}s]" if stale1 else ""
            tag2 = f" [STALE {age2:.0f}s]" if stale2 else ""
            zone_str = "  ".join(f"{lbl}={zone_counts.get(lbl, 0)}" for lbl in ALL_ZONE_LABELS)
            print(f"[{ts}] SUMMARY (every {int(output_interval)}s): {zone_str}  ||  "
                  f"cam1={tot1}{tag1}  cam2={tot2}{tag2}  ROOM TOTAL={combined}  "
                  f"FPS≈{avg_fps:.1f}  "
                  f"inf(cam1)≈{cam1.last_inference_ms:.1f}ms  inf(cam2)≈{cam2.last_inference_ms:.1f}ms  "
                  f"frames[{new1}/{new2}]  reconn[{cam1.reconnects}/{cam2.reconnects}]")

            if stale1 or stale2:
                which = ", ".join(n for n, s in ((cam1.name, stale1), (cam2.name, stale2)) if s)
                print(f"    WARNING: no fresh detections from {which}. "
                      f"Reported as {'0' if args.stale_behavior == 'zero' else 'last known value'} "
                      f"(--stale-behavior {args.stale_behavior}).")

            cam1.last_output_counts = tot1
            cam2.last_output_counts = tot2
            last_output_time = now

            if csv_writer:
                csv_writer.writerow([
                    ts,
                    *[zone_counts.get(lbl, 0) for lbl in ALL_ZONE_LABELS],
                    tot1, tot2, combined,
                    f"{avg_fps:.2f}", f"{cam1.last_inference_ms:.2f}", f"{cam2.last_inference_ms:.2f}",
                    int(stale1), int(stale2), new1, new2,
                ])
                csv_file.flush()

            if args.json_output:
                write_person_count_file(args.json_path, ts, zone_counts, tot1, tot2, combined,
                                        stale1=stale1, stale2=stale2)

        if args.display:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed, exiting.")
                break

        if not processed_any:
            time.sleep(0.01)


def run_recorded_mode(cam1, cam2, model, device, args, writer, csv_writer, csv_file, h1, w1, h2, w2):
    """
    Recorded-footage processing loop: no background threads. Reads frames
    directly and deterministically, decoding exactly 1 out of every
    --frame-interval frames per camera. One summary/CSV row (and, if
    args.json_output is enabled, one JSON bridge write) is produced per
    sampled pair of frames. Once a camera reaches real end-of-file it keeps
    reporting its last known count until the other camera finishes too, then
    the loop exits cleanly.
    """
    frame_interval = max(1, int(args.frame_interval))

    # main() already consumed frame 0 from each capture (read_first_frame(),
    # used only to measure resolution) -- rewind so sampling starts at frame 0.
    for cam in (cam1, cam2):
        cam.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps_window = deque(maxlen=30)
    last_time = time.perf_counter()
    frame_counters = {cam1.name: 0, cam2.name: 0}
    done = {cam1.name: False, cam2.name: False}
    sample_index = 0

    while not (done[cam1.name] and done[cam2.name]):
        row_zone_counts = {}
        vis_by_cam = {}

        for cam in (cam1, cam2):
            if done[cam.name]:
                # Camera finished earlier: carry its last known zone split.
                row_zone_counts.update({z.label: z.last_count for z in cam.zones})
                continue

            frame, frame_counters[cam.name], eof = next_sampled_frame(
                cam.cap, frame_interval, frame_counters[cam.name]
            )
            if eof or frame is None:
                done[cam.name] = True
                row_zone_counts.update({z.label: z.last_count for z in cam.zones})
                continue

            zone_counts, cam_total, vis = _run_detection(model, frame, cam, device, args)
            row_zone_counts.update(zone_counts)
            vis_by_cam[cam.name] = vis

            if args.display:
                cv2.imshow(cam.name, vis)

        if not vis_by_cam:
            # Neither camera produced a newly-sampled frame this cycle
            # (both just hit EOF) -- nothing left to report.
            break

        now2 = time.perf_counter()
        fps_window.append(1.0 / max(1e-6, now2 - last_time))
        last_time = now2
        avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

        tot1 = sum(row_zone_counts.get(z.label, 0) for z in cam1.zones)
        tot2 = sum(row_zone_counts.get(z.label, 0) for z in cam2.zones)
        combined = tot1 + tot2
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sample_index += 1

        zone_str = "  ".join(f"{lbl}={row_zone_counts.get(lbl, 0)}" for lbl in ALL_ZONE_LABELS)
        print(f"[{ts}] SAMPLE #{sample_index} (1 of every {frame_interval} frames -- "
              f"cam1 frame {frame_counters[cam1.name]}, cam2 frame {frame_counters[cam2.name]}): "
              f"{zone_str}  ||  cam1={tot1}  cam2={tot2}  ROOM TOTAL={combined}  FPS≈{avg_fps:.1f}  "
              f"inf(cam1)≈{cam1.last_inference_ms:.1f}ms  inf(cam2)≈{cam2.last_inference_ms:.1f}ms")

        if csv_writer:
            # Staleness doesn't apply to recorded files (frames are read
            # deterministically), so those columns are always 0/1-frame.
            csv_writer.writerow([
                ts,
                *[row_zone_counts.get(lbl, 0) for lbl in ALL_ZONE_LABELS],
                tot1, tot2, combined,
                f"{avg_fps:.2f}", f"{cam1.last_inference_ms:.2f}", f"{cam2.last_inference_ms:.2f}",
                0, 0, 1, 1,
            ])
            csv_file.flush()

        if args.json_output:
            write_person_count_file(args.json_path, ts, row_zone_counts, tot1, tot2, combined)

        _write_combined_frame(writer, vis_by_cam, cam1, cam2, h1, w1, h2, w2)

        if args.display:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed, exiting.")
                break

    print("End of video reached for both cameras. Wrapping up.")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    script_dir = Path(__file__).parent.resolve()

    print(f"Resolved device for inference: {device}")
    if isinstance(device, int):
        print(f"Attempting to use CUDA device index {device}.")
    else:
        print(f"Using device: {device}")

    # Build camera states for cam1 and cam2
    cam1_src = CAM_SOURCES["cam1"]
    cam2_src = CAM_SOURCES["cam2"]

    cam1 = CameraState(name="cam1", source=cam1_src, conf=args.conf1, imgsz=args.imgsz)
    cam2 = CameraState(name="cam2", source=cam2_src, conf=args.conf2, imgsz=args.imgsz)

    # Open VideoCaptures
    cam1.cap = cv2.VideoCapture(cam1.source)
    cam2.cap = cv2.VideoCapture(cam2.source)
    if not cam1.cap.isOpened():
        print(f"ERROR: cannot open cam1 source: {cam1.source}")
        return
    if not cam2.cap.isOpened():
        print(f"ERROR: cannot open cam2 source: {cam2.source}")
        return

    # Read one frame from each to get resolution; if fail, try a few frames
    def read_first_frame(cap):
        for _ in range(5):
            ret, f = cap.read()
            if ret:
                return f
            time.sleep(0.2)
        return None

    f1 = read_first_frame(cam1.cap)
    f2 = read_first_frame(cam2.cap)
    if f1 is None or f2 is None:
        print("ERROR: couldn't read initial frames from one or both cameras.")
        return

    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    print(f"cam1 resolution: {w1}x{h1}, cam2 resolution: {w2}x{h2}")

    # Build the zone list for each camera from ZONE_MASKS, resolve each mask
    # path, and load it at that camera's resolution.
    cli_overrides = {
        "zone1": args.zone1, "zone2": args.zone2,
        "zone3": args.zone3, "zone4": args.zone4,
    }

    for cam, (hh, ww) in ((cam1, (h1, w1)), (cam2, (h2, w2))):
        cam.zones = []
        for zone_label, configured_path in ZONE_MASKS[cam.name]:
            path = resolve_zone_mask_path(cam.name, zone_label, configured_path,
                                          cli_overrides.get(zone_label), script_dir)
            if path is None:
                print(f"ERROR: mask for {cam.name}/{zone_label} not found. "
                      f"Set it in ZONE_MASKS or pass --{zone_label}.")
                cam1.cap.release()
                cam2.cap.release()
                return
            try:
                mask = load_mask_binary(path, (hh, ww))
            except Exception as e:
                print(f"ERROR loading {cam.name}/{zone_label} mask ({path}): {e}")
                cam1.cap.release()
                cam2.cap.release()
                return
            white = int(np.count_nonzero(mask))
            if white == 0:
                print(f"WARNING: {cam.name}/{zone_label} mask is entirely black -- "
                      f"this zone will always count 0.")
            print(f"Loaded {cam.name}/{zone_label}: {path} "
                  f"({white} in-zone px, {100.0 * white / (hh * ww):.1f}% of frame)")
            cam.zones.append(ZoneState(label=zone_label, mask_path=path, mask_bin=mask))
            cam.last_zone_counts[zone_label] = 0

    # Report how much the two zones of each camera overlap. Overlap is allowed
    # -- a person in the overlap is counted once, in the earlier zone -- but a
    # large overlap usually means the masks were drawn wrong.
    for cam in (cam1, cam2):
        if len(cam.zones) == 2:
            a, b = cam.zones[0].mask_bin, cam.zones[1].mask_bin
            overlap = int(np.count_nonzero((a > 0) & (b > 0)))
            if overlap:
                union = int(np.count_nonzero((a > 0) | (b > 0))) or 1
                print(f"NOTE: {cam.name} zones overlap on {overlap} px "
                      f"({100.0 * overlap / union:.1f}% of their union). Detections there are "
                      f"counted once, in {cam.zones[0].label} (first in ZONE_MASKS order).")

    # Live mode needs background capture threads so the newest frame is
    # always ready; recorded mode reads frames directly and deterministically
    # in run_recorded_mode(), so no threads are started for it.
    watchdog_stop = threading.Event()
    if args.mode == "live":
        # Seed the frame timestamps so the watchdog doesn't fire immediately
        # on startup (read_first_frame() above already pulled a frame).
        cam1.last_frame_wall = time.time()
        cam2.last_frame_wall = time.time()

        t1 = threading.Thread(target=camera_capture_loop,
                              args=(cam1, args.reconnect_after, args.max_read_failures), daemon=True)
        t2 = threading.Thread(target=camera_capture_loop,
                              args=(cam2, args.reconnect_after, args.max_read_failures), daemon=True)
        t1.start()
        t2.start()
        wd = threading.Thread(target=camera_watchdog_loop,
                              args=((cam1, cam2), args.reconnect_after, watchdog_stop), daemon=True)
        wd.start()
        print("Capture threads started for cam1 and cam2.")
        print(f"Watchdog active: reconnect if no frames for {args.reconnect_after:.0f}s; "
              f"counts marked stale after {args.stale_timeout:.0f}s (--stale-behavior {args.stale_behavior}).")

    # Load single YOLO model instance (detection-only). Tracking call left commented for re-enable.
    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)

    OUTPUT_INTERVAL = max(1.0, float(args.output_interval))
    FRAME_INTERVAL = max(1, int(args.frame_interval))
    if args.mode == "live":
        print(f"Mode: live (summary/CSV every {OUTPUT_INTERVAL}s)")
    else:
        print(f"Mode: recorded (analyze 1 of every {FRAME_INTERVAL} frames per camera; "
              f"a summary/CSV row is written for each analyzed frame)")

    # Optional CSV logging of timestamp/counts/fps/inference-speed
    csv_file = None
    csv_writer = None
    if args.csv_output:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp",
            *[f"{lbl}_count" for lbl in ALL_ZONE_LABELS],
            "cam1_total", "cam2_total", "room_total",
            "fps", "cam1_inference_ms", "cam2_inference_ms",
            "cam1_stale", "cam2_stale", "cam1_new_frames", "cam2_new_frames",
        ])
        csv_file.flush()
        print(f"CSV logging enabled -> {csv_path}")
    else:
        print("CSV logging disabled (--csv-output false).")

    # Optional JSON bridge file (latest combined count only)
    if args.json_output:
        print(f"JSON bridge enabled -> {args.json_path}")
    else:
        print("JSON bridge disabled (--json-output false).")

    # Optionally open video writers if user provided --out (single combined output)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (max(w1, w2), max(h1, h2))
        writer = cv2.VideoWriter(args.out, fourcc, 20.0, out_size)
        print(f"Writing combined output video to {args.out} at size {out_size}")

    try:
        if args.mode == "recorded":
            run_recorded_mode(cam1, cam2, model, device, args, writer, csv_writer, csv_file, h1, w1, h2, w2)
        else:
            run_live_mode(cam1, cam2, model, device, args, writer, csv_writer, csv_file, OUTPUT_INTERVAL, h1, w1, h2, w2)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        watchdog_stop.set()
        cam1.stop_flag.set()
        cam2.stop_flag.set()
        time.sleep(0.3)
        if cam1.cap:
            cam1.cap.release()
        if cam2.cap:
            cam2.cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
            print(f"CSV results written to {args.csv_path}")
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
