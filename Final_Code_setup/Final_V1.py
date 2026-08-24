import requests
import json
from datetime import datetime, timedelta
import csv
import os
import paho.mqtt.client as mqtt

# API Credentials
API_URL = "http://10.158.66.30:80"
API_KEY = "3a21fe5a-78cb-4252-99ea-c8a87be7982e"

# MQTT Credentials
MQTT_BROKER = "10.158.71.19"
MQTT_PORT = 1883
MQTT_TOPIC = "sen55_01/data"
MQTT_USERNAME = "guest"
MQTT_PASSWORD = "smartilab123"

# Sensor order
SENSOR_ORDER = [
    "88e4c8", "88e590", "89e8d8", "889720", "87f510",
    "2da640", "89ea14", "889b88", "889938", "88e85c",
    "89e548", "88970c", "2deb24", "89e5f0", "cc8f24"
]

# Map device id to position
DEVICE_TO_POSITION = {device_id: idx + 1 for idx, device_id in enumerate(SENSOR_ORDER)}

# SEN55 expected fields
SEN55_VALUE_FIELDS = [
    "pm1_0", "pm2_5", "pm4_0", "pm10_0",
    "temperature", "humidity", "voc", "nox"
]

SEN55_METADATA_FIELDS = [
    "sensor_id", "location", "room"
]

# ==================== CALIBRATION ====================
# Path to the offsets file produced by Air-Gradient_Reference.py. Must point
# to the same output_dir that script's main() uses, so the two stay in sync.
CALIBRATION_FILE = r"D:\CoE 199\Final_Code_setup\calibration_offsets.json"

# Parameters that get calibrated. SEN55 is intentionally excluded per your notes.
CALIBRATED_PARAMS = ["temperature", "humidity", "co2", "pm25"]

# ==================== PERSON COUNT BRIDGE ====================
# Path to the JSON file written by Rtsp_zone_tracker_updated2.py every time it
# prints a SUMMARY line. Must match PERSON_COUNT_FILE in that script exactly.
PERSON_COUNT_FILE = r"D:\CoE 199\Final_Code_setup\person_count_latest.json"

# If the person-count file hasn't been updated more recently than this many
# seconds, we treat it as stale (tracker not running / crashed) and leave the
# column blank rather than reporting a frozen old number.
PERSON_COUNT_MAX_AGE_SECONDS = 30

# ==================== CONTINUOUS COLLECTION SETTINGS ====================
# How often (in seconds) to poll all sensors and append a new row. Change
# this to whatever cadence makes sense for you.
POLL_INTERVAL_SECONDS = 10

# Fixed filename for continuous runs - no timestamp in the name since the
# file now represents an entire session (started/stopped whenever), not a
# single snapshot. Each row inside still carries its own reading timestamp.
CONTINUOUS_CSV_FILENAME = "continuous_sensor_data.csv"


def load_calibration_offsets(filepath):
    """Load calibration offsets saved by Air-Gradient_Reference.py.

    Expected file structure:
    {
        "generated_at": "...",
        "note": "...",
        "offsets": {
            "88970c": {"temperature": -0.60, "humidity": ..., "co2": ..., "pm25": ...},
            ...
        }
    }

    Returns an empty dict (no calibration applied) if the file is missing,
    unreadable, or malformed - the main script keeps working with raw values.
    """
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ Calibration file not found at {filepath} - readings will NOT be corrected.")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        offsets = payload.get("offsets", {})
        generated_at = payload.get("generated_at", "unknown time")
        print(f"✓ Loaded calibration offsets from {filepath} (generated {generated_at})")
        print(f"   Sensors with calibration data: {sum(1 for s, p in offsets.items() if p)}/{len(SENSOR_ORDER)}")
        return offsets
    except Exception as e:
        print(f"⚠️ Error reading calibration file {filepath}: {e} - readings will NOT be corrected.")
        return {}


def apply_calibration(device_id, reading, offsets):
    """Return a corrected copy of `reading` (a dict with temperature/humidity/co2/pm25 keys).

    corrected_value = raw_value - offset, where offset = sensor_avg - reference_avg
    (matches the sign convention printed by Air-Gradient_Reference.py).
    If no offset exists for a given sensor/parameter, that value is left as-is.
    """
    if not offsets or device_id not in offsets:
        return reading

    sensor_offsets = offsets[device_id]
    corrected = dict(reading)

    for param in CALIBRATED_PARAMS:
        raw_value = corrected.get(param)
        offset = sensor_offsets.get(param)
        if raw_value is not None and offset is not None:
            try:
                corrected[param] = raw_value - offset
            except TypeError:
                # raw_value wasn't numeric for some reason - leave untouched
                pass

    return corrected


def read_latest_person_count(filepath, max_age_seconds=PERSON_COUNT_MAX_AGE_SECONDS):
    """Read the combined person count written by Rtsp_zone_tracker_updated2.py.

    Returns 0 if the file is missing, unreadable, or too old (tracker likely
    not running) - so the CSV always shows a number (0, 1, 2, ...) rather than
    a blank cell.
    """
    if not filepath or not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        ts = datetime.strptime(payload['timestamp'], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - ts).total_seconds()
        if age > max_age_seconds:
            print(f"⚠️ Person count file is stale ({age:.0f}s old) - reporting 0.")
            return 0
        return int(payload.get('combined', 0) or 0)
    except Exception as e:
        print(f"⚠️ Error reading person count file: {e}")
        return 0


class Air1Device:

    def __init__(self, api_url, api_key, calibration_offsets=None):
        self.api_url = api_url
        self.headers = {
            "Accept": "*/*",
            "X-API-KEY": api_key
        }
        # Offsets dict as returned by load_calibration_offsets(). Empty dict = no correction.
        self.calibration_offsets = calibration_offsets or {}

    def get_all_devices(self):
        try:
            response = requests.get(f"{self.api_url}/air-1", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status code {response.status_code}")
                return []
        except Exception as error:
            print(f"Error getting devices: {error}")
            return []

    def get_device_data(self, device_id):
        """Get latest data from a single AIR-1 device"""
        try:
            response = requests.get(f"{self.api_url}/air-1/{device_id}", headers=self.headers)

            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"Device {device_id} has invalid json")
                        return None
                else:
                    print(f"Device {device_id} has an empty response (no data)")
                    return None
            else:
                print(f"Device {device_id} has a status code error {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Device {device_id} has connection error {e}")
            return None

    def convert_timestamp_to_datetime(self, timestamp_str):
        try:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '')

            if '.' in timestamp_str:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")

            # Adjust time by 8 hours
            dt_local = dt + timedelta(hours=8)
            return dt_local
        except Exception as e:
            print(f"Error converting timestamp {timestamp_str}: {e}")
            return None

    def get_all_latest_data(self):
        """Get latest data from all AIR-1 sensors in the order list, with
        calibration offsets applied (if available) to temperature, humidity,
        co2, and pm25."""
        latest_readings = {}

        print("\n" + "=" * 80)
        print("COLLECTING LATEST DATA FROM ALL AIR-1 SENSORS")
        print("=" * 80)

        for device_id in SENSOR_ORDER:
            position = DEVICE_TO_POSITION[device_id]
            print(f"\nFetching latest data from device {device_id} (Position {position})")

            data = self.get_device_data(device_id)

            if data and 'timestamp' in data:
                # Convert timestamp to local time
                dt_local = self.convert_timestamp_to_datetime(data.get('timestamp'))

                if dt_local:
                    raw_reading = {
                        'timestamp': dt_local,
                        'temperature': data.get('temperature'),
                        'humidity': data.get('humidity'),
                        'co2': data.get('co2'),
                        'pm25': data.get('pm_2_5'),
                        'device_id': device_id,
                        'raw_timestamp': data.get('timestamp')
                    }

                    # Apply calibration correction (corrected = raw - offset)
                    calibrated_reading = apply_calibration(device_id, raw_reading, self.calibration_offsets)
                    latest_readings[device_id] = calibrated_reading  # Store by device_id instead of position

                    print(f"  ✅ Latest reading at: {dt_local.strftime('%Y-%m-%d %H:%M:%S')}")
                    if device_id in self.calibration_offsets and self.calibration_offsets[device_id]:
                        print(f"     Temp: {raw_reading.get('temperature', 'N/A')}°C (raw) -> "
                              f"{calibrated_reading.get('temperature', 'N/A')}°C (calibrated), "
                              f"RH: {raw_reading.get('humidity', 'N/A')}% (raw) -> "
                              f"{calibrated_reading.get('humidity', 'N/A')}% (calibrated)")
                    else:
                        print(f"     Temp: {calibrated_reading.get('temperature', 'N/A')}°C, "
                              f"RH: {calibrated_reading.get('humidity', 'N/A')}% (no calibration data for this sensor)")
                else:
                    print(f"  ⚠️ Could not parse timestamp")
                    latest_readings[device_id] = None
            else:
                print(f"  ❌ No data available")
                latest_readings[device_id] = None

        return latest_readings


class Sen55MQTTCollector:
    """Collector for SEN55 MQTT data - gets one reading then disconnects"""

    def __init__(self, broker, port, topic, username, password):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.latest_reading = None
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✓ Connected to MQTT broker at {self.broker}:{self.port}")
            self.connected = True
            client.subscribe(self.topic)
            print(f"✓ Subscribed to topic: {self.topic}")
        else:
            print(f"✗ MQTT connection failed with code {rc}")
            self.connected = False

    def on_message(self, client, userdata, msg):
        """Called when a message is received - store the first one then disconnect"""
        if self.latest_reading is None:  # Only take the first message
            try:
                payload = json.loads(msg.payload.decode())
                print(f"✓ Received SEN55 MQTT message")
                self.latest_reading = payload
                client.disconnect()  # Disconnect after getting one reading
            except Exception as e:
                print(f"✗ Error parsing MQTT message: {e}")

    def get_latest_reading(self, timeout_seconds=10):
        """Connect to MQTT, wait for one message, then return it.

        Resets latest_reading first - without this, calling this method a
        second time (e.g. in a continuous polling loop) would just return the
        very first cached reading forever instead of fetching a fresh one.
        """
        self.latest_reading = None
        try:
            client = mqtt.Client()
            client.username_pw_set(self.username, self.password)
            client.on_connect = self.on_connect
            client.on_message = self.on_message

            print(f"\nConnecting to MQTT broker...")
            client.connect(self.broker, self.port, timeout_seconds)

            # Start loop and wait for message
            client.loop_start()

            # Wait for message or timeout
            wait_time = 0
            while self.latest_reading is None and wait_time < timeout_seconds:
                import time
                time.sleep(0.5)
                wait_time += 0.5

            client.loop_stop()

            if self.latest_reading:
                return self.latest_reading
            else:
                print(f"✗ No SEN55 message received within {timeout_seconds} seconds")
                return None

        except Exception as e:
            print(f"✗ Error connecting to MQTT broker: {e}")
            return None

    def parse_reading(self, reading_dict):
        """Parse the MQTT reading into a structured format"""
        if not reading_dict:
            return None

        # Extract timestamp (assuming it's in the message)
        timestamp = None
        if 'timestamp' in reading_dict:
            try:
                # Try to parse timestamp (adjust format as needed)
                timestamp_str = reading_dict['timestamp']
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str.replace('Z', '')
                if '.' in timestamp_str:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                else:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                # Add 8 hours for local time
                timestamp = dt + timedelta(hours=8)
            except:
                timestamp = datetime.now() + timedelta(hours=8)
        else:
            timestamp = datetime.now() + timedelta(hours=8)

        parsed = {
            'timestamp': timestamp,
            'raw_json': json.dumps(reading_dict)
        }

        # Extract value fields
        for field in SEN55_VALUE_FIELDS:
            parsed[field] = reading_dict.get(field, '')

        # Extract metadata fields
        for field in SEN55_METADATA_FIELDS:
            parsed[field] = reading_dict.get(field, '')

        return parsed


def build_csv_headers():
    """Build the fixed column order used for every row: timestamp, then
    AIR-1 columns per sensor ID, then SEN55 columns, then person count."""
    headers = ['timestamp']

    for device_id in SENSOR_ORDER:
        headers.append(f'temp_{device_id}')
    for device_id in SENSOR_ORDER:
        headers.append(f'rh_{device_id}')
    for device_id in SENSOR_ORDER:
        headers.append(f'co2_{device_id}')
    for device_id in SENSOR_ORDER:
        headers.append(f'pm25_{device_id}')

    headers.append('sen55_timestamp')
    for field in SEN55_VALUE_FIELDS:
        headers.append(f'sen55_{field}')
    headers.append('sen55_raw_json')

    headers.append('person_count')  # from Rtsp_zone_tracker_updated2.py, stays at the very right

    return headers


def collect_one_row(air1, sen55):
    """Poll AIR-1 + SEN55 once and return a single row_data dict (calibration
    already applied to the AIR-1 values inside air1.get_all_latest_data()).
    Also returns air1_readings so the caller can print a status summary."""

    print("\n" + "=" * 80)
    print(f"POLLING SENSORS AT {(datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    air1_readings = air1.get_all_latest_data()

    sen55_raw = sen55.get_latest_reading(timeout_seconds=10)
    sen55_reading = sen55.parse_reading(sen55_raw) if sen55_raw else None

    if sen55_reading:
        print(f"\n✓ SEN55 reading at: {sen55_reading['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   PM2.5: {sen55_reading.get('pm2_5', 'N/A')} µg/m³")
    else:
        print("\n⚠️ No SEN55 data available this cycle")

    # Determine the overall timestamp for this row (use AIR-1 timestamps, or
    # SEN55's, or just "now" if neither responded)
    overall_timestamp = datetime.now() + timedelta(hours=8)
    for device_id in SENSOR_ORDER:
        if air1_readings.get(device_id):
            overall_timestamp = air1_readings[device_id]['timestamp']
            break
    if sen55_reading and not any(air1_readings.get(device_id) for device_id in SENSOR_ORDER):
        overall_timestamp = sen55_reading['timestamp']

    row_data = {
        'timestamp': overall_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }

    for device_id in SENSOR_ORDER:
        reading = air1_readings.get(device_id)

        col_name = f'temp_{device_id}'
        row_data[col_name] = reading['temperature'] if reading and reading.get('temperature') is not None else ''

        col_name = f'rh_{device_id}'
        row_data[col_name] = reading['humidity'] if reading and reading.get('humidity') is not None else ''

        col_name = f'co2_{device_id}'
        row_data[col_name] = reading['co2'] if reading and reading.get('co2') is not None else ''

        col_name = f'pm25_{device_id}'
        row_data[col_name] = reading['pm25'] if reading and reading.get('pm25') is not None else ''

    if sen55_reading:
        row_data['sen55_timestamp'] = sen55_reading['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        for field in SEN55_VALUE_FIELDS:
            row_data[f'sen55_{field}'] = sen55_reading.get(field, '')
        row_data['sen55_raw_json'] = sen55_reading.get('raw_json', '')
    else:
        row_data['sen55_timestamp'] = ''
        for field in SEN55_VALUE_FIELDS:
            row_data[f'sen55_{field}'] = ''
        row_data['sen55_raw_json'] = ''

    # Person count from the RTSP zone tracker bridge file
    row_data['person_count'] = read_latest_person_count(PERSON_COUNT_FILE)

    return row_data, air1_readings, sen55_reading


def append_row_to_csv(filepath, headers, row_data):
    """Append one row to the CSV, writing the header first only if the file
    doesn't exist yet. This is what lets the file grow across the whole
    continuous run instead of being overwritten each cycle."""
    file_is_new = not os.path.exists(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row_data)


def run_continuous_collection(output_dir=r"D:\CoE 199\data_199",
                               poll_interval_seconds=POLL_INTERVAL_SECONDS):
    """Continuously poll AIR-1 + SEN55 every `poll_interval_seconds` and
    append each reading as a new row to one fixed-name CSV file, until
    interrupted with Ctrl+C."""

    import time

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created directory: {output_dir}")

    filepath = os.path.join(output_dir, CONTINUOUS_CSV_FILENAME)
    headers = build_csv_headers()

    # Load calibration offsets once at startup. If you want the script to
    # pick up a freshly re-calibrated offsets file without restarting, move
    # this line inside the while loop below.
    calibration_offsets = load_calibration_offsets(CALIBRATION_FILE)

    air1 = Air1Device(API_URL, API_KEY, calibration_offsets=calibration_offsets)
    sen55 = Sen55MQTTCollector(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_USERNAME, MQTT_PASSWORD)

    print("=" * 80)
    print("CONTINUOUS DATA COLLECTOR FOR AIR-1 + SEN55 (WITH CALIBRATION)")
    print("=" * 80)
    print(f"Output file: {os.path.abspath(filepath)}")
    print(f"Polling every {poll_interval_seconds} seconds")
    print(f"Person count bridge file: {PERSON_COUNT_FILE}")
    print("Press Ctrl+C to stop.")
    print("=" * 80)

    row_count = 0
    start_time = datetime.now() + timedelta(hours=8)

    try:
        while True:
            try:
                row_data, air1_readings, sen55_reading = collect_one_row(air1, sen55)
                append_row_to_csv(filepath, headers, row_data)
                row_count += 1

                active_sensors = sum(1 for d in SENSOR_ORDER if air1_readings.get(d))
                print(f"\n✅ Row {row_count} written to {CONTINUOUS_CSV_FILENAME} "
                      f"({active_sensors}/15 AIR-1 sensors, SEN55: {'Yes' if sen55_reading else 'No'}, "
                      f"person_count: {row_data.get('person_count', '')})")

            except Exception as e:
                # A single failed cycle (e.g. one bad API call) shouldn't kill
                # the whole continuous run - log it and keep going.
                print(f"❌ Error during this polling cycle: {e}")
                import traceback
                traceback.print_exc()

            print(f"\nSleeping {poll_interval_seconds}s until next poll... (Ctrl+C to stop)")
            time.sleep(poll_interval_seconds)

    except KeyboardInterrupt:
        elapsed = (datetime.now() + timedelta(hours=8)) - start_time
        print("\n\n" + "=" * 80)
        print("STOPPED BY USER (Ctrl+C)")
        print("=" * 80)
        print(f"Total rows written: {row_count}")
        print(f"Session duration: {elapsed}")
        print(f"File saved at: {os.path.abspath(filepath)}")
        print("=" * 80)


def main():
    run_continuous_collection(output_dir=r"D:\CoE 199\data_199")


if __name__ == "__main__":
    # Note: You may need to install paho-mqtt if not already installed
    # Run: python -m pip install paho-mqtt
    main()