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

# DEFAULT_MASK_FILES can be absolute paths, relative, or None.
# Prefer leaving it as None and pass --mask1/--mask2, or put absolute paths using raw strings.
DEFAULT_MASK_FILES = {
    "cam1": None,
    "cam2": None,
}

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
class CameraState:
    name: str
    source: str
    cap: Optional[cv2.VideoCapture] = None
    frame_q: Queue = field(default_factory=lambda: Queue(maxsize=1))
    mask_bin: Optional[np.ndarray] = None  # binary mask resized to camera frame
    last_frame_counts: int = 0
    last_detection_time: float = 0.0
    last_output_counts: Optional[int] = None
    last_inference_ms: float = 0.0
    conf: float = 0.25
    imgsz: int = 1920
    stop_flag: threading.Event = field(default_factory=threading.Event)


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
    p.add_argument("--mask1", help="Mask PNG for cam1 (white area = in-zone).")
    p.add_argument("--mask2", help="Mask PNG for cam2 (white area = in-zone).")
    p.add_argument("--out", help="Output video file path (optional).")
    p.add_argument("--display", action="store_true", help="Show live windows.")
    p.add_argument("--imgsz", type=int, default=1920, help="Inference image size (default: 1920)")
    p.add_argument("--conf1", type=float, default=0.06, help="Confidence threshold for cam1 (default 0.25)")
    p.add_argument("--conf2", type=float, default=0.03, help="Confidence threshold for cam2 (default 0.10)")
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


def write_person_count_file(filepath, ts, out_cam1, out_cam2, combined):
    """
    Atomically write the latest combined person count to a small JSON
    bridge file so other scripts (e.g. code.py) can pick it up.

    Uses a write-to-temp-then-rename pattern so a concurrent reader never
    sees a partially written file: os.replace() is atomic on the same
    filesystem, so any reader always sees either the previous complete file
    or the new complete file, never something half-written.
    """
    try:
        payload = {
            "timestamp": ts,
            "cam1": out_cam1,
            "cam2": out_cam2,
            "combined": combined,
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


def camera_capture_loop(cam: CameraState):
    """
    Live-mode capture thread: reads frames from cam.cap and places the latest
    frame in cam.frame_q (size 1), dropping the previous one if not yet
    consumed. On a failed read (transient RTSP hiccup) it waits briefly and
    retries indefinitely -- this is only used for --mode live.
    """
    cap = cam.cap
    if cap is None:
        return
    while not cam.stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            # If read fails, wait a bit and retry; this keeps thread alive for transient RTSP hiccups.
            time.sleep(0.5)
            continue
        # Put the newest frame in the queue, dropping older if necessary
        try:
            if cam.frame_q.full():
                _ = cam.frame_q.get_nowait()
        except Exception:
            pass
        cam.frame_q.put(frame)
    # release handled externally


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


def resolve_mask_path_for_camera(cam_name: str, cli_mask: Optional[str], script_dir: Path) -> Optional[str]:
    """
    Resolve a mask path for cam_name using these strategies (in order):
      1) explicit CLI mask path (expanded and resolved)
      2) DEFAULT_MASK_FILES entry (if not None)
      3) ./masks/ directory next to this script: find any file with cam_name in filename (case-insensitive)
      4) Return None if nothing found (caller will handle error)
    Prints debug info for each candidate and returns the first existing absolute path as a string.
    """
    candidates = []

    # 1) CLI provided
    if cli_mask:
        candidates.append(Path(cli_mask))

    # 2) DEFAULT_MASK_FILES mapping
    default = DEFAULT_MASK_FILES.get(cam_name)
    if default:
        candidates.append(Path(default))

    # 3) masks folder next to script
    masks_dir = script_dir / "masks"
    candidates.append(masks_dir / f"{cam_name}-mask-fixed.png")
    candidates.append(masks_dir / f"{cam_name}_mask-fixed.png")
    # scan masks_dir for any filenames mentioning the cam name
    if masks_dir.exists() and masks_dir.is_dir():
        for p in masks_dir.iterdir():
            if cam_name.lower() in p.name.lower():
                candidates.append(p)

    # Normalize, expanduser, and check existence
    print(f"Resolving mask for {cam_name}. Candidate list (in order):")
    for c in candidates:
        try:
            c_expanded = Path(str(c)).expanduser()
            # If not absolute, make relative to script_dir
            if not c_expanded.is_absolute():
                c_try = (script_dir / c_expanded).resolve()
            else:
                c_try = c_expanded.resolve()
        except Exception:
            c_try = c_expanded  # fallback; will check exists next
        print("  -", str(c_try))
        if c_try.exists():
            resolved = str(c_try)
            print(f"Resolved mask for {cam_name}: {resolved}")
            return resolved

    # If nothing matched, print helpful diagnostics
    print(f"Could not resolve mask for {cam_name}. Checked {len(candidates)} candidates.")
    if masks_dir.exists() and masks_dir.is_dir():
        files = sorted([p.name for p in masks_dir.iterdir()])
        print(f"Files in {masks_dir}: {files}")
    else:
        print(f"No masks/ directory at {masks_dir} (checked script directory).")
    return None


def _run_detection(model, frame, cam, device, args):
    """
    Shared per-frame inference + visualization helper used by both modes.
    Runs YOLO detection on `frame`, updates cam.last_frame_counts /
    cam.last_inference_ms, and returns (count, annotated_frame).
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

    count = 0
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
                count = center_in_mask_count(cam.mask_bin, boxes, overlap_thresh=0.0)
                confs_np = confs.cpu().numpy() if confs is not None else np.ones(len(boxes))
                for (x1, y1, x2, y2), conf in zip(boxes, confs_np):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    if 0 <= cy < cam.mask_bin.shape[0] and 0 <= cx < cam.mask_bin.shape[1] and cam.mask_bin[cy, cx] > 0:
                        color = (0, 255, 0)  # green for in-mask
                    else:
                        color = (0, 0, 255)  # red for out-of-mask
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{conf:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cam.last_frame_counts = count
    cam.last_detection_time = time.perf_counter()
    return count, vis


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

            count, vis = _run_detection(model, frame, cam, device, args)
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
            out_cam1 = cam1.last_frame_counts if (cam1.last_output_counts is None or cam1.last_detection_time > last_output_time) else cam1.last_output_counts
            out_cam2 = cam2.last_frame_counts if (cam2.last_output_counts is None or cam2.last_detection_time > last_output_time) else cam2.last_output_counts
            out_cam1 = int(out_cam1 or 0)
            out_cam2 = int(out_cam2 or 0)
            combined = out_cam1 + out_cam2
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            print(f"[{ts}] SUMMARY (every {int(output_interval)}s): cam1={out_cam1}  cam2={out_cam2}  "
                  f"COMBINED={combined}  FPS≈{avg_fps:.1f}  "
                  f"inf(cam1)≈{cam1.last_inference_ms:.1f}ms  inf(cam2)≈{cam2.last_inference_ms:.1f}ms")

            cam1.last_output_counts = out_cam1
            cam2.last_output_counts = out_cam2
            last_output_time = now

            if csv_writer:
                csv_writer.writerow([
                    ts, out_cam1, out_cam2, combined,
                    f"{avg_fps:.2f}", f"{cam1.last_inference_ms:.2f}", f"{cam2.last_inference_ms:.2f}",
                ])
                csv_file.flush()

            if args.json_output:
                write_person_count_file(args.json_path, ts, out_cam1, out_cam2, combined)

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
        row_counts = {}
        vis_by_cam = {}

        for cam in (cam1, cam2):
            if done[cam.name]:
                row_counts[cam.name] = int(cam.last_frame_counts or 0)
                continue

            frame, frame_counters[cam.name], eof = next_sampled_frame(
                cam.cap, frame_interval, frame_counters[cam.name]
            )
            if eof or frame is None:
                done[cam.name] = True
                row_counts[cam.name] = int(cam.last_frame_counts or 0)
                continue

            count, vis = _run_detection(model, frame, cam, device, args)
            row_counts[cam.name] = count
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

        out_cam1 = row_counts.get(cam1.name, 0)
        out_cam2 = row_counts.get(cam2.name, 0)
        combined = out_cam1 + out_cam2
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sample_index += 1

        print(f"[{ts}] SAMPLE #{sample_index} (1 of every {frame_interval} frames -- "
              f"cam1 frame {frame_counters[cam1.name]}, cam2 frame {frame_counters[cam2.name]}): "
              f"cam1={out_cam1}  cam2={out_cam2}  COMBINED={combined}  FPS≈{avg_fps:.1f}  "
              f"inf(cam1)≈{cam1.last_inference_ms:.1f}ms  inf(cam2)≈{cam2.last_inference_ms:.1f}ms")

        if csv_writer:
            csv_writer.writerow([
                ts, out_cam1, out_cam2, combined,
                f"{avg_fps:.2f}", f"{cam1.last_inference_ms:.2f}", f"{cam2.last_inference_ms:.2f}",
            ])
            csv_file.flush()

        if args.json_output:
            write_person_count_file(args.json_path, ts, out_cam1, out_cam2, combined)

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

    # Resolve mask paths (CLI arg preferred, then defaults, then scan ./masks/)
    mask1_path = resolve_mask_path_for_camera("cam1", args.mask1, script_dir)
    mask2_path = resolve_mask_path_for_camera("cam2", args.mask2, script_dir)

    if mask1_path is None:
        print("ERROR: mask for cam1 not found. Provide --mask1 or place a file containing 'cam1' in the masks/ directory.")
        cam1.cap.release()
        cam2.cap.release()
        return
    if mask2_path is None:
        print("ERROR: mask for cam2 not found. Provide --mask2 or place a file containing 'cam2' in the masks/ directory.")
        cam1.cap.release()
        cam2.cap.release()
        return

    # Load mask images (white masks)
    try:
        cam1.mask_bin = load_mask_binary(mask1_path, (h1, w1))
        print(f"Loaded mask for cam1: {mask1_path}")
    except Exception as e:
        print(f"ERROR loading cam1 mask ({mask1_path}): {e}")
        cam1.cap.release()
        cam2.cap.release()
        return

    try:
        cam2.mask_bin = load_mask_binary(mask2_path, (h2, w2))
        print(f"Loaded mask for cam2: {mask2_path}")
    except Exception as e:
        print(f"ERROR loading cam2 mask ({mask2_path}): {e}")
        cam1.cap.release()
        cam2.cap.release()
        return

    # Live mode needs background capture threads so the newest frame is
    # always ready; recorded mode reads frames directly and deterministically
    # in run_recorded_mode(), so no threads are started for it.
    if args.mode == "live":
        t1 = threading.Thread(target=camera_capture_loop, args=(cam1,), daemon=True)
        t2 = threading.Thread(target=camera_capture_loop, args=(cam2,), daemon=True)
        t1.start()
        t2.start()
        print("Capture threads started for cam1 and cam2.")

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
            "timestamp", "cam1_count", "cam2_count", "combined_count",
            "fps", "cam1_inference_ms", "cam2_inference_ms",
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
        cam1.stop_flag.set()
        cam2.stop_flag.set()
        time.sleep(0.1)
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