import argparse
import csv
import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


AIR1_API_URL = os.getenv("AIR1_API_URL", "http://10.158.66.30:80")
AIR1_API_KEY = os.getenv("AIR1_API_KEY", "3a21fe5a-78cb-4252-99ea-c8a87be7982e")

MQTT_BROKER = os.getenv("SEN55_MQTT_BROKER", "10.158.71.19")
MQTT_PORT = int(os.getenv("SEN55_MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("SEN55_MQTT_TOPIC", "sen55/#")
MQTT_USERNAME = os.getenv("SEN55_MQTT_USERNAME", "guest")
MQTT_PASSWORD = os.getenv("SEN55_MQTT_PASSWORD", "smartilab123")

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

SEN55_METADATA_FIELDS = [
    "sensor_id",
    "x",
    "y",
    "z",
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
        "--mqtt-topic",
        default=MQTT_TOPIC,
        help="SEN55 MQTT topic. Default: sen55_01/data.",
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

    headers.append("sen55_source_timestamp")
    headers.append("sen55_age_seconds")
    headers.append("sen55_is_fresh")
    for field in SEN55_METADATA_FIELDS:
        headers.append(f"sen55_{field}")
    for field in SEN55_VALUE_FIELDS:
        headers.append(f"sen55_{field}")
    headers.append("sen55_raw_json")

    return headers


def build_row(
    timestamp: datetime,
    sensor_order: List[str],
    air1_latest_by_position: Dict[int, Optional[Reading]],
    sen55_latest: Optional[Reading],
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

    sen55_fresh = is_fresh(sen55_latest, timestamp, sen55_stale_seconds)
    row["sen55_source_timestamp"] = (
        sen55_latest.timestamp.strftime("%Y-%m-%d %H:%M:%S") if sen55_latest else ""
    )
    row["sen55_age_seconds"] = (
        int((timestamp - sen55_latest.timestamp).total_seconds()) if sen55_latest else ""
    )
    row["sen55_is_fresh"] = "1" if sen55_fresh else "0"

    for field in SEN55_METADATA_FIELDS:
        row[f"sen55_{field}"] = sen55_latest.values.get(field, "") if sen55_latest else ""
    for field in SEN55_VALUE_FIELDS:
        row[f"sen55_{field}"] = sen55_latest.values.get(field, "") if sen55_fresh and sen55_latest else ""

    row["sen55_raw_json"] = sen55_latest.raw_json if sen55_latest else ""
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
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


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
    output_path = Path(args.output) if args.output else default_live_output()
    client = Air1Client(args.air1_api_url, args.air1_api_key, args.air1_timeout)
    latest_sen55: Dict[str, Optional[Reading]] = {}

    def on_connect(mqtt, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT and subscribing to {args.mqtt_topic}")
            mqtt.subscribe(args.mqtt_topic)
        else:
            print(f"MQTT connection failed with rc={rc}")

    def on_connect(mqtt, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT and subscribing to {args.mqtt_topic}")
            mqtt.subscribe(args.mqtt_topic)
        else:
            print(f"MQTT connection failed with rc={rc}")

    def on_message(mqtt, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())

            if isinstance(payload, dict):

                reading = sen55_reading_from_row(payload)

                if reading is not None:

                    sensor_id = payload.get(
                        "sensor_id",
                        "unknown"
                    )

                    latest_sen55[sensor_id] = reading

                    print(f"Received data from {sensor_id}")

        except Exception as exc:
            print(f"Skipping SEN55 MQTT message: {exc}")

    mqtt = mqtt_client.Client(client_id="combined_training_export")

    mqtt.username_pw_set(
        args.mqtt_username,
        args.mqtt_password
    )

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

    print(f"Writing live combined rows to {output_path}")
    try:
        while True:
            if end_time is not None and time.monotonic() >= end_time:
                break

            sleep_until_next_tick(args.interval_seconds)
            grid_timestamp = floor_to_interval(local_now(), args.interval_seconds)
            air1_latest = fetch_air1_latest(client, sensor_order)
            for sensor_id, sensor_reading in latest_sen55.items():
                row = build_row(
                    timestamp=grid_timestamp,
                    sensor_order=sensor_order,
                    air1_latest_by_position=air1_latest,
                    sen55_latest=sensor_reading,
                    air1_stale_seconds=args.air1_stale_seconds,
                    sen55_stale_seconds=args.sen55_stale_seconds,
                )
                append_row(output_path, headers, row)
                row_count += 1
                print(f"Wrote row {row_count}: {row['timestamp']}")
    except KeyboardInterrupt:
        print("Stopping live collector...")
    finally:
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
