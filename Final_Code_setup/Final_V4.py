import requests
import json
import pickle
import sys
import time
import subprocess
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
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
CALIBRATION_FILE = r"C:\PlatformIO\Projects\Final_Code_setup\calibration_offsets.json"

# Parameters that get calibrated. SEN55 is intentionally excluded per your notes.
CALIBRATED_PARAMS = ["temperature", "humidity", "co2", "pm25"]

# ==================== PERSON COUNT BRIDGE ====================
# JSON file written by Rtsp_zone_tracker_updated2.py. Expected contents:
#   {"timestamp": "2026-09-29 22:14:51",
#    "zones": {"zone1": 0, "zone3": 0, "zone2": 0, "zone4": 0},
#    "cam1": 0, "cam2": 0, "combined": 0, "total": 0,
#    "cam1_stale": false, "cam2_stale": false, "stale": false,
#    "epoch": 1790691291.78}
PERSON_COUNT_FILE = r"D:\CoE 199\Final_Code_setup\person_count_latest.json"

# If the person-count file is older than this many seconds, it is treated as
# stale (tracker not running / crashed) and counts are reported as 0.
PERSON_COUNT_MAX_AGE_SECONDS = 30

# ==================== OCCUPANCY PREDICTION MODEL ====================
OCCUPANCY_MODEL_FILE = r"C:\PlatformIO\Projects\Final_Code_setup\ML_Models\occupancy_RF_model_mk1.pkl"

# ==================== PER-ZONE (CAM 1 / CAM 2) PREDICTION MODEL ====================
# MULTI-OUTPUT RandomForest trained with:
#     y = df[['cam_1_person_count', 'cam_2_person_count']]
ZONE_OCCUPANCY_MODEL_FILE = r"C:\PlatformIO\Projects\Final_Code_setup\ML_Models\to_change.pkl"

# Only needed if the model was trained on a numpy array (no stored column names).
# Leave as None to use the model's stored names, or all 60 sensor columns.
ZONE_MODEL_FEATURE_COLUMNS = None

# ==================== 4-ZONE PREDICTION MODEL ====================
# MULTI-OUTPUT RandomForest (.pkl) that predicts the person count in each of the
# 4 zones. Must be trained with a 4-column target, in THIS order:
#     y = df[['zone_1_person_count', 'zone_2_person_count',
#             'zone_3_person_count', 'zone_4_person_count']]
# Until this file exists, the zone_X_predicted_count columns are left blank.
# NOTE: this must be a DIFFERENT file from the 2-camera model above.
FOUR_ZONES_OCCUPANCY_MODEL_FILE = r"C:\PlatformIO\Projects\Final_Code_setup\ML_Models\to_change.pkl"

# Same idea as ZONE_MODEL_FEATURE_COLUMNS: only needed for numpy-trained models.
FOUR_ZONES_MODEL_FEATURE_COLUMNS = None

# ==================== CONTINUOUS COLLECTION SETTINGS ====================
POLL_INTERVAL_SECONDS = 10
AIR1_REQUEST_TIMEOUT_SECONDS = 5

# ==================== AIR-1 STALE DATA CHECK ====================
AIR1_MAX_AGE_SECONDS = 180

# ==================== SEN55 STALE DATA CHECK ====================
SEN55_MAX_AGE_SECONDS = 30

# NOTE: the header now has 8 extra columns (zone_1..zone_4). If an older file
# with this name already exists, rename/move it or change this filename,
# otherwise new rows will be misaligned with the old header.
CONTINUOUS_CSV_FILENAME = "Final_V4_iLabData.csv"

# ==================== DAILY MIDNIGHT RESTART ====================
RESTART_DAILY_AT_MIDNIGHT = True   # set False to disable
RESTART_EXIT_CODE = 42             # worker -> supervisor signal meaning "restart me"
WORKER_FLAG = "--worker"           # marks the child process that does the real work


def load_calibration_offsets(filepath):
    """Load calibration offsets saved by Air-Gradient_Reference.py.
    Returns an empty dict (no calibration applied) if the file is missing,
    unreadable, or malformed."""
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
    """Return a corrected copy of `reading`.
    corrected_value = raw_value - offset, where offset = sensor_avg - reference_avg.
    If no offset exists for a given sensor/parameter, that value is left as-is."""
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
                pass

    return corrected


def read_latest_person_counts(filepath, max_age_seconds=PERSON_COUNT_MAX_AGE_SECONDS):
    """Read the person counts written by Rtsp_zone_tracker_updated2.py.

    Returns a dict: {'cam1', 'cam2', 'combined', 'zone1', 'zone2', 'zone3', 'zone4'}

    All values are 0 if the file is missing, unreadable, or too old
    (tracker likely not running). The file is read once per cycle so every
    number comes from the same snapshot.
    """
    zeros = {'cam1': 0, 'cam2': 0, 'combined': 0,
             'zone1': 0, 'zone2': 0, 'zone3': 0, 'zone4': 0}

    if not filepath or not os.path.exists(filepath):
        return zeros
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        ts = datetime.strptime(payload['timestamp'], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - ts).total_seconds()
        if age > max_age_seconds:
            print(f"⚠️ Person count file is stale ({age:.0f}s old) - reporting 0.")
            return zeros
        zones = payload.get('zones', {}) or {}
        return {
            'cam1': int(payload.get('cam1', 0) or 0),
            'cam2': int(payload.get('cam2', 0) or 0),
            'combined': int(payload.get('combined', 0) or 0),
            'zone1': int(zones.get('zone1', 0) or 0),
            'zone2': int(zones.get('zone2', 0) or 0),
            'zone3': int(zones.get('zone3', 0) or 0),
            'zone4': int(zones.get('zone4', 0) or 0),
        }
    except Exception as e:
        print(f"⚠️ Error reading person count file: {e}")
        return zeros


# ==================== OCCUPANCY MODEL HELPERS ====================

def build_feature_column_order():
    """CSV column names in the order the occupancy model expects: all temp,
    then RH, then CO2, then PM2.5 - each block following HEADER_ORDER
    (15 x 4 = 60 features). If your training CSV used a different order,
    update HEADER_ORDER (or this function) to match."""
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


OCCUPANCY_FEATURE_COLUMNS = build_feature_column_order()


def load_occupancy_model(filepath):
    """Load the pickled RandomForest occupancy model. Returns None (predictions
    left blank) if the file is missing or fails to load."""
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ Occupancy model not found at {filepath} - predicted_occupancy will be blank.")
        return None
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded occupancy prediction model from {filepath}")

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
    """Run one prediction using the values already in row_data. Blank sensor
    readings are treated as 0.0. Returns an int, or '' on failure/no model."""
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


# ==================== PER-ZONE (CAM 1 / CAM 2) MODEL HELPERS ====================

def load_zone_model(filepath):
    """Load the 2-output per-camera RandomForest. Returns (model, feature_columns),
    or (None, []) if missing/unusable.

    Feature columns are chosen in this priority order:
      1. The names stored in the model (feature_names_in_)
      2. ZONE_MODEL_FEATURE_COLUMNS
      3. All 60 sensor columns
    """
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ Zone model not found at {filepath} - cam_1/cam_2 predicted_count will be blank.")
        return None, []

    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded zone occupancy model from {filepath}")

        n_outputs = getattr(model, 'n_outputs_', None)
        if n_outputs is not None and n_outputs != 2:
            print(f"   ⚠️ Zone model has {n_outputs} output(s) but 2 are required "
                  f"([cam_1, cam_2]) - zone predictions will be blank.")
            return None, []

        if hasattr(model, 'feature_names_in_'):
            feature_columns = [str(c) for c in model.feature_names_in_]
            source = "the model's stored feature names"
        elif ZONE_MODEL_FEATURE_COLUMNS:
            feature_columns = list(ZONE_MODEL_FEATURE_COLUMNS)
            source = "ZONE_MODEL_FEATURE_COLUMNS"
        else:
            feature_columns = list(OCCUPANCY_FEATURE_COLUMNS)
            source = "the default all-60-sensor-columns order"

        valid_columns = set(OCCUPANCY_FEATURE_COLUMNS)
        unknown = [c for c in feature_columns if c not in valid_columns]
        if unknown:
            print(f"   ⚠️ Zone model expects column(s) that don't exist in this script's CSV layout: {unknown}")
            print("      Zone predictions will be blank. Train using the exact CSV column names "
                  "(e.g. 'temp_R_Sensor1(88e4c8)').")
            return None, []

        n_expected = getattr(model, 'n_features_in_', None)
        if n_expected is not None and n_expected != len(feature_columns):
            print(f"   ⚠️ Zone model expects {n_expected} features but {len(feature_columns)} were "
                  f"resolved from {source}. Zone predictions will be blank.")
            return None, []

        print(f"   Zone model uses {len(feature_columns)} input columns (from {source}):")
        print(f"   {feature_columns}")
        return model, feature_columns

    except Exception as e:
        print(f"⚠️ Error loading zone model: {e} - cam_1/cam_2 predicted_count will be blank.")
        return None, []


def predict_zone_counts(model, feature_columns, row_data):
    """Returns (cam_1_count, cam_2_count) as non-negative ints, or ('', '')
    if no model is loaded or prediction fails."""
    if model is None or not feature_columns:
        return '', ''
    try:
        features = []
        for col in feature_columns:
            val = row_data.get(col, '')
            if val == '' or val is None:
                features.append(0.0)
            else:
                features.append(float(val))

        prediction = model.predict([features])[0]   # -> [cam_1, cam_2]
        cam1 = max(0, round(float(prediction[0])))
        cam2 = max(0, round(float(prediction[1])))
        return cam1, cam2
    except Exception as e:
        print(f"⚠️ Error predicting zone counts: {e}")
        return '', ''


# ==================== 4-ZONE MODEL HELPERS ====================

def load_four_zone_model(filepath):
    """Load the 4-output per-zone RandomForest. Returns (model, feature_columns),
    or (None, []) if missing/unusable (predicted columns are then left blank)."""
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ 4-zone model not found at {filepath} - zone_X_predicted_count will be blank.")
        return None, []

    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded 4-zone occupancy model from {filepath}")

        n_outputs = getattr(model, 'n_outputs_', None)
        if n_outputs is not None and n_outputs != 4:
            print(f"   ⚠️ 4-zone model has {n_outputs} output(s) but 4 are required "
                  f"([zone1..zone4]) - zone predictions will be blank.")
            return None, []

        if hasattr(model, 'feature_names_in_'):
            feature_columns = [str(c) for c in model.feature_names_in_]
            source = "the model's stored feature names"
        elif FOUR_ZONES_MODEL_FEATURE_COLUMNS:
            feature_columns = list(FOUR_ZONES_MODEL_FEATURE_COLUMNS)
            source = "FOUR_ZONES_MODEL_FEATURE_COLUMNS"
        else:
            feature_columns = list(OCCUPANCY_FEATURE_COLUMNS)
            source = "the default all-60-sensor-columns order"

        valid_columns = set(OCCUPANCY_FEATURE_COLUMNS)
        unknown = [c for c in feature_columns if c not in valid_columns]
        if unknown:
            print(f"   ⚠️ 4-zone model expects column(s) not in this CSV layout: {unknown}")
            print("      Zone predictions will be blank.")
            return None, []

        n_expected = getattr(model, 'n_features_in_', None)
        if n_expected is not None and n_expected != len(feature_columns):
            print(f"   ⚠️ 4-zone model expects {n_expected} features but {len(feature_columns)} were "
                  f"resolved from {source}. Zone predictions will be blank.")
            return None, []

        print(f"   4-zone model uses {len(feature_columns)} input columns (from {source}):")
        print(f"   {feature_columns}")
        return model, feature_columns

    except Exception as e:
        print(f"⚠️ Error loading 4-zone model: {e} - zone_X_predicted_count will be blank.")
        return None, []


def predict_four_zone_counts(model, feature_columns, row_data):
    """Returns (z1, z2, z3, z4) as non-negative ints, or ('', '', '', '') if no
    model is loaded or prediction fails. Blank sensor readings count as 0.0."""
    blank = ('', '', '', '')
    if model is None or not feature_columns:
        return blank
    try:
        features = []
        for col in feature_columns:
            val = row_data.get(col, '')
            features.append(0.0 if val == '' or val is None else float(val))

        prediction = model.predict([features])[0]   # -> [zone1, zone2, zone3, zone4]
        return tuple(max(0, round(float(p))) for p in prediction[:4])
    except Exception as e:
        print(f"⚠️ Error predicting 4-zone counts: {e}")
        return blank


class Air1Device:

    def __init__(self, api_url, api_key, calibration_offsets=None):
        self.api_url = api_url
        self.headers = {
            "Accept": "*/*",
            "X-API-KEY": api_key
        }
        self.calibration_offsets = calibration_offsets or {}

    def get_all_devices(self):
        try:
            response = requests.get(f"{self.api_url}/air-1", headers=self.headers,
                                    timeout=AIR1_REQUEST_TIMEOUT_SECONDS)
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
            response = requests.get(f"{self.api_url}/air-1/{device_id}", headers=self.headers,
                                    timeout=AIR1_REQUEST_TIMEOUT_SECONDS)

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

    def _fetch_one(self, device_id):
        """Fetch + stale-check + calibrate ONE AIR-1 device. Returns the
        calibrated reading dict, or None if no data / bad timestamp / stale.

        Runs inside a worker thread. All log lines are collected and printed
        in one call so they don't interleave with other devices' output."""
        position = DEVICE_TO_POSITION[device_id]
        lines = [f"\nFetching latest data from device {device_id} (Position {position})"]
        result = None

        try:
            data = self.get_device_data(device_id)

            if data and 'timestamp' in data:
                dt_local = self.convert_timestamp_to_datetime(data.get('timestamp'))

                if dt_local:
                    # API timestamp is UTC, so compare against UTC "now"
                    dt_utc = dt_local - timedelta(hours=8)
                    age_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - dt_utc).total_seconds()

                    if age_seconds > AIR1_MAX_AGE_SECONDS:
                        lines.append(f"  ⚠️ Stale data (last reading {dt_local.strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"{age_seconds:.0f}s old > {AIR1_MAX_AGE_SECONDS}s limit) - leaving blank")
                    else:
                        raw_reading = {
                            'timestamp': dt_local,
                            'temperature': data.get('temperature'),
                            'humidity': data.get('humidity'),
                            'co2': data.get('co2'),
                            'pm25': data.get('pm_2_5'),
                            'device_id': device_id,
                            'raw_timestamp': data.get('timestamp')
                        }

                        calibrated_reading = apply_calibration(device_id, raw_reading, self.calibration_offsets)
                        result = calibrated_reading

                        lines.append(f"  ✅ Latest reading at: {dt_local.strftime('%Y-%m-%d %H:%M:%S')} ({age_seconds:.0f}s old)")
                        if device_id in self.calibration_offsets and self.calibration_offsets[device_id]:
                            lines.append(f"     Temp: {raw_reading.get('temperature', 'N/A')}°C (raw) -> "
                                         f"{calibrated_reading.get('temperature', 'N/A')}°C (calibrated), "
                                         f"RH: {raw_reading.get('humidity', 'N/A')}% (raw) -> "
                                         f"{calibrated_reading.get('humidity', 'N/A')}% (calibrated)")
                        else:
                            lines.append(f"     Temp: {calibrated_reading.get('temperature', 'N/A')}°C, "
                                         f"RH: {calibrated_reading.get('humidity', 'N/A')}% (no calibration data for this sensor)")
                else:
                    lines.append("  ⚠️ Could not parse timestamp")
            else:
                lines.append("  ❌ No data available")

        except Exception as e:
            lines.append(f"  ❌ Unexpected error: {e}")
            result = None

        print("\n".join(lines))
        return result

    def get_all_latest_data(self):
        """Get latest data from all AIR-1 sensors IN PARALLEL, with calibration
        applied. Stale sensors are stored as None so their cells are blank."""
        print("\nCollecting latest data from all AIR-1 sensors...")

        with ThreadPoolExecutor(max_workers=len(SENSOR_ORDER)) as pool:
            results = list(pool.map(self._fetch_one, SENSOR_ORDER))   # keeps SENSOR_ORDER

        return dict(zip(SENSOR_ORDER, results))


class Sen55MQTTCollector:
    """Persistent MQTT subscriber for the SEN55.

    Connects ONCE at construction, stays subscribed in a background thread
    (auto-reconnects and re-subscribes), and always holds the newest message.
    """

    def __init__(self, broker, port, topic, username, password):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.latest_reading = None
        self.latest_received_at = 0.0

        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

        print(f"\nConnecting to MQTT broker at {broker}:{port}...")
        self.client.connect_async(broker, port)
        self.client.loop_start()   # background thread handles connect/reconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✓ Connected to MQTT broker at {self.broker}:{self.port}")
            client.subscribe(self.topic)   # re-subscribes after every reconnect
            print(f"✓ Subscribed to topic: {self.topic}")
        else:
            print(f"✗ MQTT connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        """Called for every message - just keep the newest one."""
        try:
            self.latest_reading = json.loads(msg.payload.decode())
            self.latest_received_at = time.time()
        except Exception as e:
            print(f"✗ Error parsing MQTT message: {e}")

    def get_latest_reading(self, max_age_seconds=SEN55_MAX_AGE_SECONDS):
        """Instant - returns the cached newest message, or None if there isn't
        one yet or it's older than max_age_seconds."""
        if self.latest_reading and (time.time() - self.latest_received_at) <= max_age_seconds:
            return self.latest_reading
        return None

    def stop(self):
        """Cleanly shut down the background MQTT thread."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def parse_reading(self, reading_dict):
        """Parse the MQTT reading into a structured format"""
        if not reading_dict:
            return None

        timestamp = None
        if 'timestamp' in reading_dict:
            try:
                timestamp_str = reading_dict['timestamp']
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str.replace('Z', '')
                if '.' in timestamp_str:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                else:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                timestamp = dt + timedelta(hours=8)
            except:
                timestamp = datetime.now() + timedelta(hours=8)
        else:
            timestamp = datetime.now() + timedelta(hours=8)

        parsed = {
            'timestamp': timestamp,
            'raw_json': json.dumps(reading_dict)
        }

        for field in SEN55_VALUE_FIELDS:
            parsed[field] = reading_dict.get(field, '')

        for field in SEN55_METADATA_FIELDS:
            parsed[field] = reading_dict.get(field, '')

        return parsed


def build_csv_headers():
    """Fixed column order for every row: timestamp, AIR-1 blocks (temp, rh,
    co2, pm25; Right then Middle then Left), SEN55 columns, person count,
    predicted occupancy, per-camera counts/predictions (BV:BY), and finally
    the 4-zone person counts and predictions at the very end."""
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

    headers.append('person_count')  # BT - combined count from Rtsp_zone_tracker_updated2.py

    headers.append('predicted_occupancy')  # BU - from the occupancy model

    headers.append('cam_1_person_count')     # BV - ground truth, camera 1
    headers.append('cam_1_predicted_count')  # BW - from the 2-output model (output 1)
    headers.append('cam_2_person_count')     # BX - ground truth, camera 2
    headers.append('cam_2_predicted_count')  # BY - from the 2-output model (output 2)

    # Appended at the very end so existing column positions don't shift
    for z in range(1, 5):
        headers.append(f'zone_{z}_person_count')
        headers.append(f'zone_{z}_predicted_count')

    return headers


def collect_one_row(air1, sen55, occupancy_model=None, zone_model=None, zone_feature_columns=None,
                    four_zone_model=None, four_zone_feature_columns=None,
                    row_timestamp=None):
    """Poll AIR-1 + SEN55 once and return a single row_data dict (calibration
    already applied). Also returns air1_readings and sen55_reading so the
    caller can print a status summary.

    row_timestamp: the scheduled tick time (UTC+8, naive datetime) to stamp on
    this row. If omitted, the current time (UTC+8) is used.
    """

    print(f"\n--- Polling sensors at {(datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')} ---")

    air1_readings = air1.get_all_latest_data()

    sen55_raw = sen55.get_latest_reading()   # instant - cached newest message
    sen55_reading = sen55.parse_reading(sen55_raw) if sen55_raw else None

    if sen55_reading:
        print(f"\n✓ SEN55 reading at: {sen55_reading['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   PM2.5: {sen55_reading.get('pm2_5', 'N/A')} µg/m³")
    else:
        print("\n⚠️ No fresh SEN55 data available this cycle")

    # Row timestamp is the SCHEDULED tick time, so rows are exactly
    # POLL_INTERVAL_SECONDS apart regardless of sensor-reported timestamps.
    overall_timestamp = row_timestamp or (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8))

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

    # Person counts (combined + per camera + per zone) from the RTSP zone
    # tracker bridge file - read once so all numbers come from the same snapshot.
    person_counts = read_latest_person_counts(PERSON_COUNT_FILE)
    row_data['person_count'] = person_counts['combined']
    row_data['cam_1_person_count'] = person_counts['cam1']
    row_data['cam_2_person_count'] = person_counts['cam2']

    # Live occupancy prediction from the trained RandomForest model
    row_data['predicted_occupancy'] = predict_occupancy(occupancy_model, row_data)

    # Per-camera predictions from the 2-output model
    cam1_pred, cam2_pred = predict_zone_counts(zone_model, zone_feature_columns, row_data)
    row_data['cam_1_predicted_count'] = cam1_pred
    row_data['cam_2_predicted_count'] = cam2_pred

    # 4-zone ground truth (from the tracker JSON) and predictions
    for z in range(1, 5):
        row_data[f'zone_{z}_person_count'] = person_counts[f'zone{z}']

    zone_preds = predict_four_zone_counts(four_zone_model, four_zone_feature_columns, row_data)
    for z, pred in enumerate(zone_preds, start=1):
        row_data[f'zone_{z}_predicted_count'] = pred

    return row_data, air1_readings, sen55_reading


def append_row_to_csv(filepath, headers, row_data):
    """Append one row to the CSV, writing the header first only if the file
    doesn't exist yet."""
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


# ==================== MIDNIGHT RESTART HELPERS ====================

def next_midnight(now=None):
    """The next 12:00 AM (PC local time) after `now`."""
    now = now or datetime.now()
    return datetime.combine(now.date() + timedelta(days=1), datetime.min.time())


def run_supervisor():
    """Launch the collector as a child process and relaunch it whenever it
    exits with RESTART_EXIT_CODE (i.e. at midnight). Any other exit (Ctrl+C,
    clean stop, crash) ends the supervisor too."""
    script_path = os.path.abspath(__file__)

    while True:
        print("[supervisor] Starting collector process...")
        proc = subprocess.Popen([sys.executable, "-u", script_path, WORKER_FLAG])

        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C also reaches the child, so give it time to print its summary
            try:
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                proc.kill()
            print("[supervisor] Stopped by user.")
            return

        if exit_code == RESTART_EXIT_CODE:
            print("[supervisor] Midnight reached - restarting collector...")
            time.sleep(1)
            continue

        print(f"[supervisor] Collector exited with code {exit_code} - not restarting.")
        return


def run_continuous_collection(output_dir=r"C:\PlatformIO\Projects\Final_Code_setup\data_199",
                               poll_interval_seconds=POLL_INTERVAL_SECONDS,
                               restart_at_midnight=RESTART_DAILY_AT_MIDNIGHT):
    """Continuously poll AIR-1 + SEN55 on a FIXED clock schedule and append
    each reading as a new row to one fixed-name CSV file, until interrupted
    with Ctrl+C. Returns True if it stopped because midnight was reached
    (caller should restart), False otherwise."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created directory: {output_dir}")

    filepath = os.path.join(output_dir, CONTINUOUS_CSV_FILENAME)
    headers = build_csv_headers()

    # Load calibration offsets once at startup (reloaded on every midnight restart).
    calibration_offsets = load_calibration_offsets(CALIBRATION_FILE)

    # Load the occupancy prediction model once at startup too.
    occupancy_model = load_occupancy_model(OCCUPANCY_MODEL_FILE)

    # Per-camera (cam 1 / cam 2) 2-output model.
    zone_model, zone_feature_columns = load_zone_model(ZONE_OCCUPANCY_MODEL_FILE)

    # 4-zone 4-output model. Returns (None, []) until the .pkl exists, in
    # which case the zone_X_predicted_count columns stay blank.
    four_zone_model, four_zone_feature_columns = load_four_zone_model(FOUR_ZONES_OCCUPANCY_MODEL_FILE)

    air1 = Air1Device(API_URL, API_KEY, calibration_offsets=calibration_offsets)
    sen55 = Sen55MQTTCollector(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_USERNAME, MQTT_PASSWORD)

    restart_deadline = next_midnight() if restart_at_midnight else None

    print(f"\nAIR-1 + SEN55 continuous collector (calibration + occupancy + camera/zone prediction)")
    print(f"Output file: {os.path.abspath(filepath)}")
    print(f"Fixed schedule: one row every {poll_interval_seconds}s (clock-aligned) | Ctrl+C to stop")
    print(f"AIR-1 readings older than {AIR1_MAX_AGE_SECONDS}s are written as blank (stale)")
    if restart_deadline:
        print(f"Auto-restart at: {restart_deadline.strftime('%Y-%m-%d %H:%M:%S')}")

    row_count = 0
    start_time = datetime.now() + timedelta(hours=8)
    restart_requested = False

    interval = poll_interval_seconds
    next_tick = (time.time() // interval + 1) * interval   # next :00/:10/:20... boundary

    try:
        while True:
            # Checked BEFORE polling so we don't write a stray row after midnight
            if restart_deadline and datetime.now() >= restart_deadline:
                restart_requested = True
                break

            # Wait for the tick, but never sleep past midnight
            wait = next_tick - time.time()
            if restart_deadline:
                wait = min(wait, (restart_deadline - datetime.now()).total_seconds())
            if wait > 0:
                time.sleep(wait)
            if time.time() < next_tick:
                continue   # woke early (e.g. because of the midnight cap) - re-check

            # Row timestamp = the scheduled tick (UTC+8, independent of PC timezone)
            tick_dt = datetime.fromtimestamp(next_tick, timezone.utc).replace(tzinfo=None) + timedelta(hours=8)

            try:
                row_data, air1_readings, sen55_reading = collect_one_row(
                    air1, sen55, occupancy_model, zone_model, zone_feature_columns,
                    four_zone_model, four_zone_feature_columns,
                    row_timestamp=tick_dt
                )
                append_row_to_csv(filepath, headers, row_data)
                row_count += 1

                active_sensors = sum(1 for d in SENSOR_ORDER if air1_readings.get(d))
                print(f"\n✅ Row {row_count} written "
                      f"({active_sensors}/15 AIR-1 fresh, SEN55: {'Yes' if sen55_reading else 'No'}, "
                      f"person_count: {row_data.get('person_count', '')}, "
                      f"predicted_occupancy: {row_data.get('predicted_occupancy', '')})")
                print(f"   Cam 1: {row_data.get('cam_1_person_count', '')} actual / "
                      f"{row_data.get('cam_1_predicted_count', '') if row_data.get('cam_1_predicted_count', '') != '' else '-'} predicted | "
                      f"Cam 2: {row_data.get('cam_2_person_count', '')} actual / "
                      f"{row_data.get('cam_2_predicted_count', '') if row_data.get('cam_2_predicted_count', '') != '' else '-'} predicted")
                print("   " + " | ".join(
                    f"Z{z}: {row_data[f'zone_{z}_person_count']} actual / "
                    f"{row_data[f'zone_{z}_predicted_count'] if row_data[f'zone_{z}_predicted_count'] != '' else '-'} pred"
                    for z in range(1, 5)))

            except Exception as e:
                # A single failed cycle shouldn't kill the whole continuous run
                print(f"❌ Error during this polling cycle: {e}")
                import traceback
                traceback.print_exc()

            # Schedule the next tick from the clock, not from "now", so it can't drift.
            next_tick += interval
            if time.time() >= next_tick:   # cycle overran: skip missed ticks instead of bursting
                missed = int((time.time() - next_tick) // interval) + 1
                print(f"⚠️ Cycle took too long - skipped {missed} tick(s)")
                next_tick += missed * interval

            print(f"Next row at {datetime.fromtimestamp(next_tick).strftime('%H:%M:%S')} "
                  f"({max(0, next_tick - time.time()):.1f}s from now)")

        # Only reached via the midnight break above
        print(f"\n🔄 Midnight reached - restarting. Rows this session: {row_count}")

    except KeyboardInterrupt:
        elapsed = (datetime.now() + timedelta(hours=8)) - start_time
        print(f"\n\n🛑 Stopped by user. Rows written: {row_count} | "
              f"Duration: {elapsed} | File: {os.path.abspath(filepath)}")

    finally:
        sen55.stop()   # shut down the background MQTT thread cleanly

    return restart_requested


def main():
    # Launched normally -> become the supervisor, which spawns the real worker
    if RESTART_DAILY_AT_MIDNIGHT and WORKER_FLAG not in sys.argv:
        run_supervisor()
        return

    # Worker process (or restart feature disabled): do the actual collection
    restart_needed = run_continuous_collection(output_dir=r"C:\PlatformIO\Projects\Final_Code_setup\data_199")
    if restart_needed:
        sys.exit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    # Note: You may need to install paho-mqtt and scikit-learn if not already installed
    # Run: python -m pip install paho-mqtt scikit-learn
    main()
