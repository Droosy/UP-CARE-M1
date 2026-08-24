#!/usr/bin/env python3
"""
Dual-camera detection using YOLOv5n and white mask images per camera.

Notes:
 - Uses white-on-black mask PNGs for each camera: white pixels are counted as "in-zone".
 - Two cameras run concurrently (capture threads); a single inference worker processes frames.
 - Confidence thresholds: cam1=0.25, cam2=0.10 by default.
 - imgsz default: 1920 (applies to both unless overridden).
 - Prints per-camera counts and combined total every OUTPUT_INTERVAL seconds.
 - Detection-only by default; original tracking calls are left commented for easy re-enable.
 - Robust mask path resolution and automatic CPU fallback if no CUDA devices are available.
"""

import argparse
import sys
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
#    "cam1": r"C:/Users/julia/yolo_project/videos/Feb27_front_cam.mp4",
#    "cam2": r"C:/Users/julia/yolo_project/videos/Feb27_rear_cam.mp4",
#}

# DEFAULT_MASK_FILES can be absolute paths, relative, or None.
# Prefer leaving it as None and pass --mask1/--mask2, or put absolute paths using raw strings.
DEFAULT_MASK_FILES = {
    "cam1": None,
    "cam2": None,
}


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
    conf: float = 0.25
    imgsz: int = 1920
    stop_flag: threading.Event = field(default_factory=threading.Event)


def parse_args():
    p = argparse.ArgumentParser(description="Dual-camera detection using white masks.")
    p.add_argument("--model", default="yolov5nu.pt",
                   help="YOLO model path (default: yolov5n.pt)")
    p.add_argument("--mask1", help="Mask PNG for cam1 (white area = in-zone).")
    p.add_argument("--mask2", help="Mask PNG for cam2 (white area = in-zone).")
    p.add_argument("--out", help="Output video file path (optional).")
    p.add_argument("--display", action="store_true", help="Show live windows.")
    p.add_argument("--imgsz", type=int, default=1920, help="Inference image size (default: 1920)")
    p.add_argument("--conf1", type=float, default=0.10, help="Confidence threshold for cam1 (default 0.25)")
    p.add_argument("--conf2", type=float, default=0.10, help="Confidence threshold for cam2 (default 0.10)")
    p.add_argument("--device", default="0", help="Device for inference (0 for GPU0, cpu for CPU).")
    p.add_argument("--output-interval", type=float, default=10.0, help="Seconds between summary outputs (default 10s)")
    p.add_argument("--max-persons", type=int, default=100, help="Max detections per frame")
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


def camera_capture_loop(cam: CameraState):
    """
    Capture thread: reads frames from cam.cap and places the latest frame in cam.frame_q (size 1).
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

    # Start capture threads
    t1 = threading.Thread(target=camera_capture_loop, args=(cam1,), daemon=True)
    t2 = threading.Thread(target=camera_capture_loop, args=(cam2,), daemon=True)
    t1.start()
    t2.start()
    print("Capture threads started for cam1 and cam2.")

    # Load single YOLO model instance (detection-only). Tracking call left commented for re-enable.
    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)

    # rolling FPS
    fps_window = deque(maxlen=30)
    last_time = time.perf_counter()

    # Output timing
    OUTPUT_INTERVAL = max(1.0, float(args.output_interval))
    last_output_time = time.perf_counter()

    # Optionally open video writers if user provided --out (single combined output)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (max(w1, w2), max(h1, h2))
        writer = cv2.VideoWriter(args.out, fourcc, 20.0, out_size)
        print(f"Writing combined output video to {args.out} at size {out_size}")

    try:
        while True:
            now = time.perf_counter()

            # Inference: process whichever camera has a frame available (non-blocking)
            processed_any = False
            for cam in (cam1, cam2):
                try:
                    frame = cam.frame_q.get(timeout=0.01)
                except Empty:
                    continue  # no frame for this cam now

                # Run detection for this camera (detection-only)
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

                # Extract boxes and count using center-in-mask
                count = 0
                if len(results) > 0:
                    r = results[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = getattr(r.boxes, "xyxy", None)
                        if xyxy is not None:
                            boxes = xyxy.cpu().numpy().astype(int)
                            count = center_in_mask_count(cam.mask_bin, boxes, overlap_thresh=0.0)
                cam.last_frame_counts = count
                cam.last_detection_time = time.perf_counter()
                processed_any = True

                # Visualization & optional writer/display
                vis = frame.copy()
                if len(results) > 0:
                    r = results[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = getattr(r.boxes, "xyxy", None)
                        confs = getattr(r.boxes, "conf", None)
                        if xyxy is not None:
                            boxes = xyxy.cpu().numpy().astype(int)
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

                if args.display:
                    cv2.imshow(cam.name, vis)

                if writer:
                    H_out, W_out = max(h1, h2), max(w1, w2)
                    canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)
                    if cam is cam1:
                        canvas[0:vis.shape[0], 0:vis.shape[1]] = vis
                    else:
                        x_off = W_out - vis.shape[1]
                        canvas[0:vis.shape[0], x_off:x_off + vis.shape[1]] = vis
                    writer.write(canvas)

            # Update FPS
            now2 = time.perf_counter()
            fps_window.append(1.0 / max(1e-6, now2 - last_time))
            last_time = now2
            avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

            # Periodic output every OUTPUT_INTERVAL seconds with fallback
            if now - last_output_time >= OUTPUT_INTERVAL:
                out_cam1 = cam1.last_frame_counts if (cam1.last_output_counts is None or cam1.last_detection_time > last_output_time) else cam1.last_output_counts
                out_cam2 = cam2.last_frame_counts if (cam2.last_output_counts is None or cam2.last_detection_time > last_output_time) else cam2.last_output_counts

                out_cam1 = int(out_cam1 or 0)
                out_cam2 = int(out_cam2 or 0)
                combined = out_cam1 + out_cam2
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{ts}] SUMMARY (every {int(OUTPUT_INTERVAL)}s): cam1={out_cam1}  cam2={out_cam2}  COMBINED={combined}  FPS≈{avg_fps:.1f}")

                cam1.last_output_counts = out_cam1
                cam2.last_output_counts = out_cam2
                last_output_time = now

            if args.display:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("'q' pressed, exiting.")
                    break

            if not processed_any:
                time.sleep(0.01)

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
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()