import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Iterable, List, Optional

import requests

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "Please install ultralytics: pip install ultralytics. Error: " + str(e)
    ) from e


AIR1_API_URL = os.getenv("AIR1_API_URL", "http://10.158.66.30:80")
AIR1_API_KEY = os.getenv("AIR1_API_KEY", "3a21fe5a-78cb-4252-99ea-c8a87be7982e")

MQTT_BROKER = os.getenv("SEN55_MQTT_BROKER", "10.158.71.19")
MQTT_PORT = int(os.getenv("SEN55_MQTT_PORT", "1883"))
MQTT_TOPICS = ["sen55_01/data", "sen55_02/data",]
MQTT_USERNAME = os.getenv("SEN55_MQTT_USERNAME", "guest")
MQTT_PASSWORD = os.getenv("SEN55_MQTT_PASSWORD", "smartilab123")

RTSP_MASK_DIR = Path(__file__).resolve().parent / "masks"
RTSP_CAM_SOURCES = {
    "cam1": os.getenv(
        "RTSP_CAM1_SOURCE",
        "rtsp://admin:++smartilab2023@10.158.71.241:554/Streaming/channels/101",
    ),
    "cam2": os.getenv(
        "RTSP_CAM2_SOURCE",
        "rtsp://admin:++smartilab2023@10.158.71.240:554/Streaming/channels/101",
    ),
}
RTSP_PEOPLE_FIELDS = [
    "rtsp_people_cam1_count",
    "rtsp_people_cam2_count",
    "rtsp_people_total_count",
    "rtsp_people_source_timestamp",
    "rtsp_people_age_seconds",
    "rtsp_people_is_fresh",
]

LOCAL_OFFSET = timedelta(hours=8)

SENSOR_ORDER = [
    "88e4c8",
    "88e590",
    "89e8d8",
    "889720",
    "87f510",
    "2da640",
    "89ea14",
    "889b88",
    "889938",
    "88e85c",
    "89e548",
    "88970c",
    "2deb24",
    "89e5f0",
    "cc8f24",
]

AIR1_VALUE_FIELDS = [
    ("temp", "temperature"),
    ("rh", "humidity"),
    ("co2", "co2"),
    ("pm25", "pm_2_5"),
]

SEN55_SENSORS = [
    "sen55_01",
    "sen55_02",
]

SEN55_METADATA_FIELDS = [
    "sensor_id",
    "location",
    "room",
]

SEN55_VALUE_FIELDS = [
    "pm1_0",
    "pm2_5",
    "pm4_0",
    "pm10_0",
    "temperature",
    "humidity",
    "voc",
    "nox",
]


@dataclass
class Reading:
    timestamp: datetime
    values: Dict[str, object]
    raw_json: str


class LatestCursor:
    def __init__(self, readings: Iterable[Reading]):
        self.readings = sorted(readings, key=lambda item: item.timestamp)
        self.index = -1

    def at_or_before(self, timestamp: datetime) -> Optional[Reading]:
        while self.index + 1 < len(self.readings):
            next_reading = self.readings[self.index + 1]
            if next_reading.timestamp > timestamp:
                break
            self.index += 1

        if self.index < 0:
            return None
        return self.readings[self.index]


class Air1Client:
    def __init__(self, api_url: str, api_key: str, timeout: int):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Accept": "*/*",
            "X-API-KEY": api_key,
        }

    def get_latest(self, device_id: str) -> Optional[Dict[str, object]]:
        url = f"{self.api_url}/air-1/{device_id}"
        return self._get_json(url, f"AIR-1 latest {device_id}")

    def get_historical(
        self, device_id: str, time_start_utc: datetime, time_end_utc: datetime
    ) -> List[Dict[str, object]]:
        start_encoded = urllib.parse.quote(format_utc_for_api(time_start_utc))
        end_encoded = urllib.parse.quote(format_utc_for_api(time_end_utc))
        url = (
            f"{self.api_url}/air-1/{device_id}"
            f"?time_start={start_encoded}&time_end={end_encoded}"
        )
        payload = self._get_json(url, f"AIR-1 history {device_id}")
        if payload is None:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        print(f"Skipping AIR-1 {device_id}: unexpected payload type {type(payload).__name__}")
        return []

    def _get_json(self, url: str, label: str) -> Optional[object]:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                print(f"{label} failed with status {response.status_code}")
                return None
            if not response.text or not response.text.strip():
                print(f"{label} returned an empty response")
                return None
            return response.json()
        except requests.exceptions.RequestException as exc:
            print(f"{label} connection error: {exc}")
        except json.JSONDecodeError as exc:
            print(f"{label} returned invalid JSON: {exc}")
        return None


def resolve_device(value: str):
    """
    Resolve the --device argument into a value accepted by Ultralytics:
      - 'cpu' -> 'cpu'
      - numeric GPU index -> that GPU if CUDA is available, otherwise CPU
      - other strings are returned as-is
    """
    s = value.strip().lower()

    if s == "cpu":
        return "cpu"

    if s.isdigit():
        gpu_index = int(s)
        if torch is not None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return gpu_index
            warnings.warn(
                f"No CUDA devices detected (torch.cuda.is_available()={torch.cuda.is_available()}). "
                "Falling back to CPU. To force CPU explicitly, pass --device cpu."
            )
            return "cpu"

        warnings.warn(
            "PyTorch not importable; falling back to CPU. Install torch for GPU support."
        )
        return "cpu"

    return value


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


def load_mask_binary(path: str, target_hw):
    """
    Load a mask image (white on black). Resize to target_hw (h, w)
    and return a binary uint8 mask with 255 for white.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found or cannot be decoded: {path}")

    H, W = target_hw
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

    _, binm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binm


def center_in_mask_count(mask_bin, boxes_xyxy, overlap_thresh: float = 0.0) -> int:
    """
    Count detections whose center point falls into the binary mask.
    """
    H, W = mask_bin.shape[:2]
    count = 0

    for (x1, y1, x2, y2) in boxes_xyxy:
        bx1 = max(0, int(x1))
        by1 = max(0, int(y1))
        bx2 = min(W - 1, int(x2))
        by2 = min(H - 1, int(y2))

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
    Capture thread: reads frames from cam.cap and places the latest
    frame in cam.frame_q (size 1).
    """
    cap = cam.cap
    if cap is None:
        return

    while not cam.stop_flag.is_set():
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.5)
            continue

        try:
            if cam.frame_q.full():
                _ = cam.frame_q.get_nowait()
        except Exception:
            pass

        cam.frame_q.put(frame)


class RtspPeopleCounter:
    def __init__(
        self,
        cam1_source: str,
        cam2_source: str,
        mask1_path: str,
        mask2_path: str,
        model_path: str,
        device: str,
        imgsz: int,
        conf1: float,
        conf2: float,
        max_persons: int,
    ):
        self.cam1_source = cam1_source
        self.cam2_source = cam2_source
        self.mask1_path = mask1_path
        self.mask2_path = mask2_path
        self.model_path = model_path
        self.device_value = device
        self.imgsz = imgsz
        self.conf1 = conf1
        self.conf2 = conf2
        self.max_persons = max_persons

        self.model = None
        self.device = None
        self.cams = []
        self.capture_threads = []
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_counts = {"cam1": None, "cam2": None}
        self.latest_timestamp: Optional[datetime] = None

    def start(self) -> None:
        try:
            self.device = resolve_device(self.device_value)
            cam1 = CameraState(
                name="cam1",
                source=self.cam1_source,
                conf=self.conf1,
                imgsz=self.imgsz,
            )
            cam2 = CameraState(
                name="cam2",
                source=self.cam2_source,
                conf=self.conf2,
                imgsz=self.imgsz,
            )
            self.cams = [cam1, cam2]

            for cam in self.cams:
                cam.cap = cv2.VideoCapture(cam.source)
                if not cam.cap.isOpened():
                    raise RuntimeError(f"Cannot open {cam.name} source: {cam.source}")

            first_frames = {}
            for cam in self.cams:
                frame = self._read_first_frame(cam.cap)
                if frame is None:
                    raise RuntimeError(f"Could not read initial frame from {cam.name}")
                first_frames[cam.name] = frame

            cam1.mask_bin = load_mask_binary(
                self.mask1_path,
                first_frames["cam1"].shape[:2],
            )
            cam2.mask_bin = load_mask_binary(
                self.mask2_path,
                first_frames["cam2"].shape[:2],
            )

            for cam in self.cams:
                thread = threading.Thread(
                    target=camera_capture_loop,
                    args=(cam,),
                    daemon=True,
                )
                thread.start()
                self.capture_threads.append(thread)

            print(f"Loading RTSP people-count model {self.model_path} ...")
            self.model = YOLO(self.model_path)
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("RTSP people counter started.")
        except Exception:
            self.stop()
            raise

    def _read_first_frame(self, cap):
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                return frame
            time.sleep(0.2)
        return None

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            processed_any = False
            for cam in self.cams:
                try:
                    frame = cam.frame_q.get(timeout=0.01)
                except Empty:
                    continue

                try:
                    results = self.model.predict(
                        source=frame,
                        conf=cam.conf,
                        iou=0.3,
                        classes=[0],
                        device=self.device,
                        imgsz=cam.imgsz,
                        max_det=self.max_persons,
                        verbose=False,
                    )
                    count = 0
                    if len(results) > 0:
                        result = results[0]
                        if hasattr(result, "boxes") and result.boxes is not None:
                            xyxy = getattr(result.boxes, "xyxy", None)
                            if xyxy is not None:
                                boxes = xyxy.cpu().numpy().astype(int)
                                count = center_in_mask_count(
                                    cam.mask_bin,
                                    boxes,
                                    overlap_thresh=0.0,
                                )
                    self._set_count(cam.name, count)
                    processed_any = True
                except Exception as exc:
                    print(f"RTSP people counter error for {cam.name}: {exc}")
                    time.sleep(1)

            if not processed_any:
                time.sleep(0.01)

    def _set_count(self, cam_name: str, count: int) -> None:
        with self.lock:
            self.latest_counts[cam_name] = int(count)
            self.latest_timestamp = local_now()

    def snapshot(self) -> Optional[Dict[str, object]]:
        with self.lock:
            if self.latest_timestamp is None:
                return None
            return {
                "cam1": self.latest_counts.get("cam1"),
                "cam2": self.latest_counts.get("cam2"),
                "timestamp": self.latest_timestamp,
            }

    def stop(self) -> None:
        self.stop_event.set()
        for cam in self.cams:
            cam.stop_flag.set()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        for thread in self.capture_threads:
            if thread.is_alive():
                thread.join(timeout=1)

        for cam in self.cams:
            if cam.cap:
                cam.cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one combined SEN55 + AIR-1 CSV for ML training."
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=10,
        help="Output grid interval in seconds. Default: 10.",
    )
    parser.add_argument(
        "--air1-stale-seconds",
        type=int,
        default=120,
        help="Maximum age for filling AIR-1 numeric values. Default: 120.",
    )
    parser.add_argument(
        "--sen55-stale-seconds",
        type=int,
        default=30,
        help="Maximum age for filling SEN55 sensor values. Default: 30.",
    )
    parser.add_argument(
        "--air1-api-url",
        default=AIR1_API_URL,
        help="AIR-1 API URL. Defaults to AIR1_API_URL env or local lab URL.",
    )
    parser.add_argument(
        "--air1-api-key",
        default=AIR1_API_KEY,
        help="AIR-1 API key. Defaults to AIR1_API_KEY env or the local fallback.",
    )
    parser.add_argument(
        "--air1-timeout",
        type=int,
        default=30,
        help="AIR-1 HTTP timeout in seconds. Default: 30.",
    )
    parser.add_argument(
        "--sensor-order",
        default=",".join(SENSOR_ORDER),
        help="Comma-separated AIR-1 device IDs mapped to s1..sN.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    historical = subparsers.add_parser(
        "historical",
        help="Fetch AIR-1 history and merge it with an existing SEN55 CSV.",
    )
    historical.add_argument(
        "--time-start",
        required=True,
        help="UTC start time, for example: 2026-03-28 03:00:00.",
    )
    historical.add_argument(
        "--time-end",
        required=True,
        help="UTC end time, for example: 2026-03-28 05:00:00.",
    )
    historical.add_argument(
        "--sen55-csv",
        default="sen55_data.csv",
        help="Existing SEN55 CSV from sen55_to_csv.py. Default: sen55_data.csv.",
    )
    historical.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: combined_training_<range>.csv.",
    )
    live = subparsers.add_parser(
        "live",
        help="Subscribe to SEN55 MQTT, poll AIR-1 latest readings, and append rows.",
    )
    live.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: combined_live_<timestamp>.csv.",
    )
    live.add_argument(
        "--duration-min",
        type=float,
        default=None,
        help="Optional run duration in minutes. Omit to run until Ctrl+C.",
    )
    live.add_argument(
        "--mqtt-broker",
        default=MQTT_BROKER,
        help="SEN55 MQTT broker. Defaults to SEN55_MQTT_BROKER env or lab broker.",
    )
    live.add_argument(
        "--mqtt-port",
        type=int,
        default=MQTT_PORT,
        help="SEN55 MQTT port. Default: 1883.",
    )
    live.add_argument(
        "--mqtt-username",
        default=MQTT_USERNAME,
        help="SEN55 MQTT username.",
    )
    live.add_argument(
        "--mqtt-password",
        default=MQTT_PASSWORD,
        help="SEN55 MQTT password.",
    )
    live.add_argument(
        "--allow-missing-sen55",
        action="store_true",
        help=(
            "Continue live collection if the SEN55 MQTT broker cannot be reached. "
            "SEN55 columns will stay blank until a reading is available."
        ),
    )
    live.add_argument(
        "--disable-rtsp-counter",
        action="store_true",
        help="Disable RTSP people-count columns in live mode.",
    )
    live.add_argument(
        "--rtsp-cam1",
        default=RTSP_CAM_SOURCES["cam1"],
        help="RTSP/video source for people counter cam1.",
    )
    live.add_argument(
        "--rtsp-cam2",
        default=RTSP_CAM_SOURCES["cam2"],
        help="RTSP/video source for people counter cam2.",
    )
    live.add_argument(
        "--rtsp-mask1",
        default=str(RTSP_MASK_DIR / "cam1-mask-fixed.png"),
        help="Mask PNG for people counter cam1.",
    )
    live.add_argument(
        "--rtsp-mask2",
        default=str(RTSP_MASK_DIR / "cam2-mask-fixed.png"),
        help="Mask PNG for people counter cam2.",
    )
    live.add_argument(
        "--rtsp-model",
        default="yolov5nu.pt",
        help="YOLO model path for RTSP people counting.",
    )
    live.add_argument(
        "--rtsp-device",
        default="0",
        help="Device for RTSP inference. Use 0 for GPU0 or cpu for CPU.",
    )
    live.add_argument(
        "--rtsp-imgsz",
        type=int,
        default=1920,
        help="RTSP inference image size. Default: 1920.",
    )
    live.add_argument(
        "--rtsp-conf1",
        type=float,
        default=0.10,
        help="RTSP confidence threshold for cam1. Default: 0.10.",
    )
    live.add_argument(
        "--rtsp-conf2",
        type=float,
        default=0.10,
        help="RTSP confidence threshold for cam2. Default: 0.10.",
    )
    live.add_argument(
        "--rtsp-max-persons",
        type=int,
        default=100,
        help="Maximum RTSP people detections per frame. Default: 100.",
    )
    live.add_argument(
        "--rtsp-stale-seconds",
        type=int,
        default=30,
        help="Maximum age for filling RTSP people counts. Default: 30.",
    )

    return parser.parse_args()


def format_utc_for_api(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_cli_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    text = text.replace("T", " ")
    if "." in text:
        text = text.split(".", 1)[0]
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def parse_source_timestamp(value: object, assume_utc: bool) -> Optional[datetime]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1]
        assume_utc = True

    text = text.replace("T", " ")
    if "." in text:
        text = text.split(".", 1)[0]

    try:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    if assume_utc:
        parsed = parsed + LOCAL_OFFSET
    return parsed.replace(microsecond=0)


def floor_to_interval(value: datetime, interval_seconds: int) -> datetime:
    value = value.replace(microsecond=0)
    seconds_since_midnight = value.hour * 3600 + value.minute * 60 + value.second
    remainder = seconds_since_midnight % interval_seconds
    return value - timedelta(seconds=remainder)


def ceil_to_interval(value: datetime, interval_seconds: int) -> datetime:
    floored = floor_to_interval(value, interval_seconds)
    if floored == value.replace(microsecond=0):
        return floored
    return floored + timedelta(seconds=interval_seconds)


def local_now() -> datetime:
    return (datetime.utcnow() + LOCAL_OFFSET).replace(microsecond=0)


def iter_grid(start_local: datetime, end_local: datetime, interval_seconds: int):
    current = ceil_to_interval(start_local, interval_seconds)
    final = floor_to_interval(end_local, interval_seconds)
    while current <= final:
        yield current
        current += timedelta(seconds=interval_seconds)


def raw_json(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def air1_reading_from_payload(payload: Dict[str, object]) -> Optional[Reading]:
    timestamp = parse_source_timestamp(payload.get("timestamp"), assume_utc=True)
    if timestamp is None:
        return None
    return Reading(timestamp=timestamp, values=payload, raw_json=raw_json(payload))


def sen55_reading_from_row(row: Dict[str, object]) -> Optional[Reading]:
    timestamp = parse_source_timestamp(row.get("timestamp"), assume_utc=False)
    if timestamp is None:
        return None
    return Reading(timestamp=timestamp, values=dict(row), raw_json=raw_json(dict(row)))


def load_sen55_csv(path: Path) -> List[Reading]:
    readings = []
    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reading = sen55_reading_from_row(row)
            if reading is not None:
                readings.append(reading)
    return sorted(readings, key=lambda item: item.timestamp)


def fetch_air1_history(
    client: Air1Client,
    sensor_order: List[str],
    time_start_utc: datetime,
    time_end_utc: datetime,
) -> Dict[int, List[Reading]]:
    readings_by_position: Dict[int, List[Reading]] = {}
    for position, device_id in enumerate(sensor_order, start=1):
        payloads = client.get_historical(device_id, time_start_utc, time_end_utc)
        readings = []
        for payload in payloads:
            reading = air1_reading_from_payload(payload)
            if reading is not None:
                readings.append(reading)
        readings_by_position[position] = sorted(readings, key=lambda item: item.timestamp)
        print(f"AIR-1 s{position} ({device_id}): {len(readings)} readings")
    return readings_by_position


def build_headers(sensor_count: int) -> List[str]:
    headers = ["timestamp"]

    for prefix, _ in AIR1_VALUE_FIELDS:
        for position in range(1, sensor_count + 1):
            headers.append(f"{prefix}_s{position}")

    for position in range(1, sensor_count + 1):
        headers.append(f"air1_device_id_s{position}")
    for position in range(1, sensor_count + 1):
        headers.append(f"air1_source_timestamp_s{position}")
    for position in range(1, sensor_count + 1):
        headers.append(f"air1_age_seconds_s{position}")
    for position in range(1, sensor_count + 1):
        headers.append(f"air1_is_fresh_s{position}")
    for position in range(1, sensor_count + 1):
        headers.append(f"air1_raw_json_s{position}")

    for sensor_name in SEN55_SENSORS:
        headers.append(f"{sensor_name}_source_timestamp")
        headers.append(f"{sensor_name}_age_seconds")
        headers.append(f"{sensor_name}_is_fresh")

        for field in SEN55_METADATA_FIELDS:
            headers.append(f"{sensor_name}_{field}")

        for field in SEN55_VALUE_FIELDS:
            headers.append(f"{sensor_name}_{field}")

        headers.append(f"{sensor_name}_raw_json")

    return headers


def build_row(
    timestamp: datetime,
    sensor_order: List[str],
    air1_latest_by_position: Dict[int, Optional[Reading]],
    sen55_latest_by_sensor: Dict[str, Optional[Reading]],
    air1_stale_seconds: int,
    sen55_stale_seconds: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    sensor_count = len(sensor_order)

    air1_fresh_by_position = {}
    for position in range(1, sensor_count + 1):
        reading = air1_latest_by_position.get(position)
        fresh = is_fresh(reading, timestamp, air1_stale_seconds)
        air1_fresh_by_position[position] = fresh

    for prefix, source_key in AIR1_VALUE_FIELDS:
        for position in range(1, sensor_count + 1):
            reading = air1_latest_by_position.get(position)
            col_name = f"{prefix}_s{position}"
            row[col_name] = reading.values.get(source_key, "") if air1_fresh_by_position[position] and reading else ""

    for position, device_id in enumerate(sensor_order, start=1):
        row[f"air1_device_id_s{position}"] = device_id

    for position in range(1, sensor_count + 1):
        reading = air1_latest_by_position.get(position)
        row[f"air1_source_timestamp_s{position}"] = (
            reading.timestamp.strftime("%Y-%m-%d %H:%M:%S") if reading else ""
        )

    for position in range(1, sensor_count + 1):
        reading = air1_latest_by_position.get(position)
        row[f"air1_age_seconds_s{position}"] = (
            int((timestamp - reading.timestamp).total_seconds()) if reading else ""
        )

    for position in range(1, sensor_count + 1):
        row[f"air1_is_fresh_s{position}"] = "1" if air1_fresh_by_position[position] else "0"

    for position in range(1, sensor_count + 1):
        reading = air1_latest_by_position.get(position)
        row[f"air1_raw_json_s{position}"] = reading.raw_json if reading else ""

    for sensor_name in SEN55_SENSORS:
        sen55_latest = sen55_latest_by_sensor.get(sensor_name)
        sen55_fresh = is_fresh(sen55_latest, timestamp, sen55_stale_seconds)

        row[f"{sensor_name}_source_timestamp"] = (
        sen55_latest.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if sen55_latest else ""
        )

        row[f"{sensor_name}_age_seconds"] = (
        int((timestamp - sen55_latest.timestamp).total_seconds())
        if sen55_latest else ""
        )

        row[f"{sensor_name}_is_fresh"] = "1" if sen55_fresh else "0"

        for field in SEN55_METADATA_FIELDS:
            row[f"{sensor_name}_{field}"] = (
                sen55_latest.values.get(field, "")
                if sen55_latest else ""
            )

        for field in SEN55_VALUE_FIELDS:
            row[f"{sensor_name}_{field}"] = (
                sen55_latest.values.get(field, "")
                if sen55_fresh and sen55_latest else ""
            )

        row[f"{sensor_name}_raw_json"] = (
            sen55_latest.raw_json if sen55_latest else ""
        )
    return row


def is_fresh(reading: Optional[Reading], timestamp: datetime, stale_seconds: int) -> bool:
    if reading is None:
        return False
    age = (timestamp - reading.timestamp).total_seconds()
    return 0 <= age <= stale_seconds


def write_rows(path: Path, headers: List[str], rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            row_count += 1
    return row_count


def append_row(path: Path, headers: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction="ignore")
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:
            print(f"Cannot write to {path}. Close the CSV file if it is open. Retrying in 2 seconds...")
            time.sleep(2)


def add_rtsp_people_counts(
    row: Dict[str, object],
    timestamp: datetime,
    snapshot: Optional[Dict[str, object]],
    stale_seconds: int,
) -> None:
    for field in RTSP_PEOPLE_FIELDS:
        row[field] = ""
    row["rtsp_people_is_fresh"] = "0"

    if not snapshot:
        return

    source_timestamp = snapshot.get("timestamp")
    if not isinstance(source_timestamp, datetime):
        return

    age_seconds = max(0, int((timestamp - source_timestamp).total_seconds()))
    is_fresh_count = age_seconds <= stale_seconds
    row["rtsp_people_source_timestamp"] = source_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    row["rtsp_people_age_seconds"] = age_seconds
    row["rtsp_people_is_fresh"] = "1" if is_fresh_count else "0"

    if not is_fresh_count:
        return

    cam1_count = snapshot.get("cam1")
    cam2_count = snapshot.get("cam2")
    row["rtsp_people_cam1_count"] = cam1_count if cam1_count is not None else ""
    row["rtsp_people_cam2_count"] = cam2_count if cam2_count is not None else ""
    row["rtsp_people_total_count"] = (
        int(cam1_count) + int(cam2_count)
        if cam1_count is not None and cam2_count is not None
        else ""
    )


def validate_existing_csv_header(path: Path, headers: List[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        actual_header = next(csv.reader(csvfile), None)

    if actual_header != headers:
        raise ValueError(
            f"{path} already exists with a different CSV header. "
            "Use a new --output path, or use --disable-rtsp-counter for the old live schema."
        )


def build_historical_rows(
    grid_start_local: datetime,
    grid_end_local: datetime,
    interval_seconds: int,
    sensor_order: List[str],
    air1_readings_by_position: Dict[int, List[Reading]],
    sen55_readings: List[Reading],
    air1_stale_seconds: int,
    sen55_stale_seconds: int,
):
    air1_cursors = {
        position: LatestCursor(readings)
        for position, readings in air1_readings_by_position.items()
    }
    sen55_cursor = LatestCursor(sen55_readings)

    for grid_timestamp in iter_grid(grid_start_local, grid_end_local, interval_seconds):
        air1_latest = {
            position: air1_cursors[position].at_or_before(grid_timestamp)
            for position in range(1, len(sensor_order) + 1)
        }
        sen55_latest = sen55_cursor.at_or_before(grid_timestamp)
        yield build_row(
            timestamp=grid_timestamp,
            sensor_order=sensor_order,
            air1_latest_by_position=air1_latest,
            sen55_latest=sen55_latest,
            air1_stale_seconds=air1_stale_seconds,
            sen55_stale_seconds=sen55_stale_seconds,
        )


def validate_csv_width(path: Path, headers: List[str]) -> None:
    expected = len(headers)
    bad_rows = []
    seen_timestamps = set()
    duplicate_count = 0

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        actual_header = next(reader, None)
        if actual_header != headers:
            raise ValueError("CSV header does not match the expected combined schema.")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected:
                bad_rows.append((row_number, len(row)))
            if row and row[0] in seen_timestamps:
                duplicate_count += 1
            elif row:
                seen_timestamps.add(row[0])

    if bad_rows:
        preview = ", ".join(f"row {num}: {width}" for num, width in bad_rows[:5])
        raise ValueError(f"CSV width check failed; expected {expected} columns ({preview}).")
    if duplicate_count:
        raise ValueError(f"CSV has {duplicate_count} duplicate timestamps.")


def print_summary(path: Path, row_count: int, headers: List[str]) -> None:
    print("=" * 80)
    print("COMBINED EXPORT COMPLETE")
    print(f"Rows: {row_count}")
    print(f"Columns: {len(headers)}")
    print(f"File: {path}")
    print(f"Full path: {path.resolve()}")
    print("=" * 80)


def run_historical(args: argparse.Namespace) -> None:
    sensor_order = parse_sensor_order(args.sensor_order)
    time_start_utc = parse_cli_datetime(args.time_start)
    time_end_utc = parse_cli_datetime(args.time_end)
    if time_end_utc < time_start_utc:
        raise ValueError("--time-end must be after --time-start")

    sen55_path = Path(args.sen55_csv)
    if not sen55_path.exists():
        raise FileNotFoundError(
            f"SEN55 CSV not found: {sen55_path}. Run sen55_to_csv.py first or pass --sen55-csv."
        )

    output_path = Path(args.output) if args.output else default_historical_output(time_start_utc, time_end_utc)
    client = Air1Client(args.air1_api_url, args.air1_api_key, args.air1_timeout)

    print("Loading SEN55 CSV...")
    sen55_readings = load_sen55_csv(sen55_path)
    print(f"SEN55 readings: {len(sen55_readings)}")

    print("Fetching AIR-1 historical data...")
    air1_readings = fetch_air1_history(client, sensor_order, time_start_utc, time_end_utc)

    headers = build_headers(len(sensor_order))
    start_local = time_start_utc + LOCAL_OFFSET
    end_local = time_end_utc + LOCAL_OFFSET

    rows = build_historical_rows(
        grid_start_local=start_local,
        grid_end_local=end_local,
        interval_seconds=args.interval_seconds,
        sensor_order=sensor_order,
        air1_readings_by_position=air1_readings,
        sen55_readings=sen55_readings,
        air1_stale_seconds=args.air1_stale_seconds,
        sen55_stale_seconds=args.sen55_stale_seconds,
    )
    row_count = write_rows(output_path, headers, rows)
    validate_csv_width(output_path, headers)
    print_summary(output_path, row_count, headers)


def run_live(args: argparse.Namespace) -> None:
    try:
        from paho.mqtt import client as mqtt_client
    except ImportError as exc:
        raise SystemExit(
            "Live mode requires paho-mqtt. Install it with: pip install paho-mqtt"
        ) from exc

    sensor_order = parse_sensor_order(args.sensor_order)
    headers = build_headers(len(sensor_order))
    include_rtsp_counter = not args.disable_rtsp_counter
    if include_rtsp_counter:
        headers = headers + RTSP_PEOPLE_FIELDS
    output_path = Path(args.output) if args.output else default_live_output()
    validate_existing_csv_header(output_path, headers)
    client = Air1Client(args.air1_api_url, args.air1_api_key, args.air1_timeout)
    latest_sen55: Dict[str, Optional[Reading]] = {"sen55_01": None,"sen55_02": None,}

    def on_connect(mqtt, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT")
            
            for topic in MQTT_TOPICS:
                print(f"Subscribing to {topic}")
                mqtt.subscribe(topic)
        else:
            print(f"MQTT connection failed with rc={rc}")

    def on_message(mqtt, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if isinstance(payload, dict):

                reading = sen55_reading_from_row(payload)
                
                if reading is not None:

                    sensor_id = payload.get("sensor_id")
                    
                    if sensor_id in latest_sen55:
                        latest_sen55[sensor_id] = reading

        except Exception as exc:
            print(f"Skipping SEN55 MQTT message: {exc}")

    mqtt = mqtt_client.Client(client_id="combined_training_export")
    mqtt.username_pw_set(args.mqtt_username, args.mqtt_password)
    mqtt.on_connect = on_connect
    mqtt.on_message = on_message

    mqtt_started = False
    try:
        mqtt.connect(args.mqtt_broker, args.mqtt_port)
        mqtt.loop_start()
        mqtt_started = True
    except OSError as exc:
        message = (
            f"Could not connect to SEN55 MQTT broker at "
            f"{args.mqtt_broker}:{args.mqtt_port}: {exc}. "
            "Check that the broker is running, the IP/port are correct, and the PC is on the lab network. "
            "You can pass --mqtt-broker/--mqtt-port, or use --allow-missing-sen55 to write AIR-1 rows "
            "with SEN55 columns blank."
        )
        if not args.allow_missing_sen55:
            raise SystemExit(message) from exc
        print(message)
        print("Continuing without SEN55 MQTT.")

    end_time = time.monotonic() + (args.duration_min * 60) if args.duration_min else None
    row_count = 0
    rtsp_counter = None

    if include_rtsp_counter:
        try:
            rtsp_counter = RtspPeopleCounter(
                cam1_source=args.rtsp_cam1,
                cam2_source=args.rtsp_cam2,
                mask1_path=args.rtsp_mask1,
                mask2_path=args.rtsp_mask2,
                model_path=args.rtsp_model,
                device=args.rtsp_device,
                imgsz=args.rtsp_imgsz,
                conf1=args.rtsp_conf1,
                conf2=args.rtsp_conf2,
                max_persons=args.rtsp_max_persons,
            )
            rtsp_counter.start()
        except Exception as exc:
            print(f"RTSP people counter disabled for this run: {exc}")

    print(f"Writing live combined rows to {output_path}")
    try:
        while True:
            if end_time is not None and time.monotonic() >= end_time:
                break

            sleep_until_next_tick(args.interval_seconds)
            grid_timestamp = floor_to_interval(local_now(), args.interval_seconds)
            air1_latest = fetch_air1_latest(client, sensor_order)
            row = build_row(
                timestamp=grid_timestamp,
                sensor_order=sensor_order,
                air1_latest_by_position=air1_latest,
                sen55_latest_by_sensor=latest_sen55,
                air1_stale_seconds=args.air1_stale_seconds,
                sen55_stale_seconds=args.sen55_stale_seconds,
            )
            if include_rtsp_counter:
                add_rtsp_people_counts(
                    row,
                    grid_timestamp,
                    rtsp_counter.snapshot() if rtsp_counter else None,
                    args.rtsp_stale_seconds,
                )
            append_row(output_path, headers, row)
            row_count += 1
            if include_rtsp_counter:
                print(
                    f"Wrote row {row_count}: {row['timestamp']} "
                    f"rtsp_total={row['rtsp_people_total_count']}"
                )
            else:
                print(f"Wrote row {row_count}: {row['timestamp']}")
    except KeyboardInterrupt:
        print("Stopping live collector...")
    finally:
        if rtsp_counter:
            rtsp_counter.stop()
        if mqtt_started:
            mqtt.loop_stop()
            mqtt.disconnect()

    validate_csv_width(output_path, headers)
    print_summary(output_path, row_count, headers)


def fetch_air1_latest(
    client: Air1Client, sensor_order: List[str]
) -> Dict[int, Optional[Reading]]:
    readings: Dict[int, Optional[Reading]] = {}
    for position, device_id in enumerate(sensor_order, start=1):
        payload = client.get_latest(device_id)
        readings[position] = air1_reading_from_payload(payload) if isinstance(payload, dict) else None
    return readings


def sleep_until_next_tick(interval_seconds: int) -> None:
    now = local_now()
    next_tick = floor_to_interval(now, interval_seconds) + timedelta(seconds=interval_seconds)
    wait_seconds = max(0.0, (next_tick - now).total_seconds())
    time.sleep(wait_seconds)


def parse_sensor_order(value: str) -> List[str]:
    sensor_order = [item.strip() for item in value.split(",") if item.strip()]
    if not sensor_order:
        raise ValueError("--sensor-order must contain at least one AIR-1 device ID")
    return sensor_order


def default_historical_output(time_start_utc: datetime, time_end_utc: datetime) -> Path:
    start = time_start_utc.strftime("%Y%m%d_%H%M%S")
    end = time_end_utc.strftime("%Y%m%d_%H%M%S")
    return Path(f"combined_training_{start}_to_{end}.csv")


def default_live_output() -> Path:
    stamp = local_now().strftime("%Y%m%d_%H%M%S")
    return Path(f"combined_live_{stamp}.csv")


def main() -> None:
    args = parse_args()
    if args.mode == "historical":
        run_historical(args)
    elif args.mode == "live":
        run_live(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
