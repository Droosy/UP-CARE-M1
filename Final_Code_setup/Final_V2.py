import requests
import json
import pickle
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

# Sensor order (this is the order sensors are POLLED in - independent of the
# order columns appear in the CSV, which is controlled by HEADER_ORDER below)
SENSOR_ORDER = [
    "88e4c8", "88e590", "89e8d8", "889720", "87f510",
    "2da640", "89ea14", "889b88", "889938", "88e85c",
    "89e548", "88970c", "2deb24", "89e5f0", "cc8f24"
]

# Map device id to position
DEVICE_TO_POSITION = {device_id: idx + 1 for idx, device_id in enumerate(SENSOR_ORDER)}

# ==================== SENSOR LABELS (Right / Middle / Left) ====================
# Human-readable label for each sensor id, used to build CSV column names like
# "temp_R_Sensor1(88e4c8)". Also defines the display order for CSV columns:
# all Right sensors first, then Middle, then Left.
SENSOR_LABELS = {
    "88e4c8": "R_Sensor1",
    "89e8d8": "R_Sensor2",
    "88e590": "R_Sensor3",
    "889720": "M_Sensor1",
    "889b88": "M_Sensor2",
    "87f510": "M_Sensor3",
    "889938": "M_Sensor4",
    "2da640": "M_Sensor5",
    "88e85c": "M_Sensor6",
    "89ea14": "M_Sensor7",
    "89e548": "M_Sensor8",
    "88970c": "L_Sensor1",
    "2deb24": "L_Sensor2",
    "89e5f0": "L_Sensor3",
    "cc8f24": "L_Sensor4",
}

# Column order: Right sensors (1-3), then Middle (1-8), then Left (1-4).
HEADER_ORDER = [
    "88e4c8", "89e8d8", "88e590",   # Right 1, 2, 3
    "889720", "889b88", "87f510", "889938", "2da640", "88e85c", "89ea14", "89e548",  # Middle 1-8
    "88970c", "2deb24", "89e5f0", "cc8f24",  # Left 1-4
]

assert set(HEADER_ORDER) == set(SENSOR_ORDER), "HEADER_ORDER must contain exactly the same sensor ids as SENSOR_ORDER"


def sensor_column_name(param, device_id):
    """Build a CSV column name like 'temp_R_Sensor1(88e4c8)' for a given
    parameter prefix (temp/rh/co2/pm25) and device id."""
    label = SENSOR_LABELS[device_id]
    return f"{param}_{label}({device_id})"


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

# ==================== OCCUPANCY PREDICTION MODEL ====================
# Path to the trained RandomForest model (.pkl) that predicts occupancy from
# the temp / RH / CO2 / PM2.5 readings.
OCCUPANCY_MODEL_FILE = r"D:\CoE 199\random_forest_models\my_occupancy_model.pkl"

# ==================== CONTINUOUS COLLECTION SETTINGS ====================
# How often (in seconds) to poll all sensors and append a new row. Change
# this to whatever cadence makes sense for you.
POLL_INTERVAL_SECONDS = 60

# Fixed filename for continuous runs - no timestamp in the name since the
# file now represents an entire session (started/stopped whenever), not a
# single snapshot. Each row inside still carries its own reading timestamp.
CONTINUOUS_CSV_FILENAME = "Final_V2_iLabData.csv"


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


# ==================== OCCUPANCY MODEL HELPERS ====================

def build_feature_column_order():
    """The list of CSV column names, IN THE ORDER the occupancy model expects
    them as input features: all temp columns, then all RH columns, then all
    CO2 columns, then all PM2.5 columns - each block following HEADER_ORDER
    (Right sensors, then Middle, then Left). This is meant to mirror columns
    B:BI in your training spreadsheet (15 temp + 15 RH + 15 CO2 + 15 PM25 = 60
    features).

    IMPORTANT: if your training CSV had the sensor columns in a different
    order within each block, update HEADER_ORDER (or this function) to match,
    otherwise the model will receive mismatched inputs silently.
    """
    cols = []
    for device_id in HEADER_ORDER:
        cols.append(sensor_column_name('temp', device_id))
    for device_id in HEADER_ORDER:
        cols.append(sensor_column_name('rh', device_id))
    for device_id in HEADER_ORDER:
        cols.append(sensor_column_name('co2', device_id))
    for device_id in HEADER_ORDER:
        cols.append(sensor_column_name('pm25', device_id))
    return cols


# Computed once at import time - the fixed 60-column feature order fed to the model.
OCCUPANCY_FEATURE_COLUMNS = build_feature_column_order()


def load_occupancy_model(filepath):
    """Load the pickled RandomForest occupancy model. Returns None (predictions
    will be left blank) if the file is missing or fails to load, so a bad/missing
    model never crashes data collection."""
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ Occupancy model not found at {filepath} - predicted_occupancy will be blank.")
        return None
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded occupancy prediction model from {filepath}")

        # If the model was trained on a pandas DataFrame, scikit-learn stores
        # the training column names/order here - print them so you can
        # visually confirm they match OCCUPANCY_FEATURE_COLUMNS below.
        if hasattr(model, 'feature_names_in_'):
            trained_cols = list(model.feature_names_in_)
            print(f"   Model was trained with {len(trained_cols)} named features, in this order:")
            print(f"   {trained_cols}")
            if trained_cols != OCCUPANCY_FEATURE_COLUMNS:
                print("   ⚠️ WARNING: this does NOT match OCCUPANCY_FEATURE_COLUMNS below.")
                print("      Predictions may be inaccurate until the order/names are fixed.")
        elif hasattr(model, 'n_features_in_'):
            print(f"   Model expects {model.n_features_in_} input features "
                  f"(no column names stored - can't verify order automatically).")

        return model
    except Exception as e:
        print(f"⚠️ Error loading occupancy model: {e} - predicted_occupancy will be blank.")
        return None


def predict_occupancy(model, row_data):
    """Run one prediction using the temp/RH/CO2/PM2.5 values already placed in
    row_data. Missing sensor readings (blank string) are treated as 0.0 so a
    single dropped sensor doesn't stop prediction entirely - keep an eye on
    the 'active sensors' count printed each cycle to judge reliability.

    Returns an int occupancy count, or '' if no model is loaded or prediction
    fails for any reason.
    """
    if model is None:
        return ''
    try:
        features = []
        for col in OCCUPANCY_FEATURE_COLUMNS:
            val = row_data.get(col, '')
            if val == '' or val is None:
                features.append(0.0)
            else:
                features.append(float(val))

        prediction = model.predict([features])[0]
        return max(0, round(float(prediction)))
    except Exception as e:
        print(f"⚠️ Error predicting occupancy: {e}")
        return ''


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
    AIR-1 columns (Right sensors first, then Middle, then Left, per
    HEADER_ORDER) for each parameter block, then SEN55 columns, then
    person count, then the model's predicted occupancy (new, last column)."""
    headers = ['timestamp']

    for device_id in HEADER_ORDER:
        headers.append(sensor_column_name('temp', device_id))
    for device_id in HEADER_ORDER:
        headers.append(sensor_column_name('rh', device_id))
    for device_id in HEADER_ORDER:
        headers.append(sensor_column_name('co2', device_id))
    for device_id in HEADER_ORDER:
        headers.append(sensor_column_name('pm25', device_id))

    headers.append('sen55_timestamp')
    for field in SEN55_VALUE_FIELDS:
        headers.append(f'sen55_{field}')
    headers.append('sen55_raw_json')

    headers.append('person_count')  # from Rtsp_zone_tracker_updated2.py

    headers.append('predicted_occupancy')  # NEW: from my_occupancy_model.pkl, stays at the very right

    return headers


def collect_one_row(air1, sen55, occupancy_model=None):
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

    # Determine the overall timestamp for this row: use the LATEST (max)
    # timestamp among all AIR-1 sensors that responded and the SEN55 reading
    # (if present). Falls back to "now" if nothing responded at all.
    all_timestamps = [
        air1_readings[device_id]['timestamp']
        for device_id in SENSOR_ORDER
        if air1_readings.get(device_id) and air1_readings[device_id].get('timestamp')
    ]
    if sen55_reading and sen55_reading.get('timestamp'):
        all_timestamps.append(sen55_reading['timestamp'])

    overall_timestamp = max(all_timestamps) if all_timestamps else datetime.now() + timedelta(hours=8)

    row_data = {
        'timestamp': overall_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }

    for device_id in HEADER_ORDER:
        reading = air1_readings.get(device_id)

        col_name = sensor_column_name('temp', device_id)
        row_data[col_name] = reading['temperature'] if reading and reading.get('temperature') is not None else ''

        col_name = sensor_column_name('rh', device_id)
        row_data[col_name] = reading['humidity'] if reading and reading.get('humidity') is not None else ''

        col_name = sensor_column_name('co2', device_id)
        row_data[col_name] = reading['co2'] if reading and reading.get('co2') is not None else ''

        col_name = sensor_column_name('pm25', device_id)
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

    # NEW: live occupancy prediction from the trained RandomForest model,
    # computed from the temp/RH/CO2/PM2.5 values already filled in above.
    row_data['predicted_occupancy'] = predict_occupancy(occupancy_model, row_data)

    return row_data, air1_readings, sen55_reading


def append_row_to_csv(filepath, headers, row_data):
    """Append one row to the CSV, writing the header first only if the file
    doesn't exist yet. This is what lets the file grow across the whole
    continuous run instead of being overwritten each cycle."""
    file_is_new = not os.path.exists(filepath)

    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if file_is_new:
                writer.writeheader()
            writer.writerow(row_data)
    except PermissionError:
        print(f"❌ Can't write to {os.path.basename(filepath)} - it looks like the file is open "
              f"in Excel. Please CLOSE the CSV in Excel so data collection can continue.")
        print(f"   Tip: if you want to view the data while the script is running, open a COPY "
              f"of the CSV instead of the original file.")
        raise


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

    # Load the occupancy prediction model once at startup too.
    occupancy_model = load_occupancy_model(OCCUPANCY_MODEL_FILE)

    air1 = Air1Device(API_URL, API_KEY, calibration_offsets=calibration_offsets)
    sen55 = Sen55MQTTCollector(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_USERNAME, MQTT_PASSWORD)

    print("=" * 80)
    print("CONTINUOUS DATA COLLECTOR FOR AIR-1 + SEN55 (WITH CALIBRATION + OCCUPANCY PREDICTION)")
    print("=" * 80)
    print(f"Output file: {os.path.abspath(filepath)}")
    print(f"Polling every {poll_interval_seconds} seconds")
    print(f"Person count bridge file: {PERSON_COUNT_FILE}")
    print(f"Occupancy model file: {OCCUPANCY_MODEL_FILE}")
    print("Press Ctrl+C to stop.")
    print("=" * 80)

    row_count = 0
    start_time = datetime.now() + timedelta(hours=8)

    try:
        while True:
            try:
                row_data, air1_readings, sen55_reading = collect_one_row(air1, sen55, occupancy_model)
                append_row_to_csv(filepath, headers, row_data)
                row_count += 1

                active_sensors = sum(1 for d in SENSOR_ORDER if air1_readings.get(d))
                print(f"\n✅ Row {row_count} written to {CONTINUOUS_CSV_FILENAME} "
                      f"({active_sensors}/15 AIR-1 sensors, SEN55: {'Yes' if sen55_reading else 'No'}, "
                      f"person_count: {row_data.get('person_count', '')}, "
                      f"predicted_occupancy: {row_data.get('predicted_occupancy', '')})")

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
    # Note: You may need to install paho-mqtt and scikit-learn if not already installed
    # Run: python -m pip install paho-mqtt scikit-learn
    main()