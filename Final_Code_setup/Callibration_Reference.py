import requests
import json
from datetime import datetime, timedelta
import urllib.parse
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import numpy as np

# Set font for better rendering
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

#API Credentials
API_URL = "http://10.158.66.30:80"
API_KEY = "3a21fe5a-78cb-4252-99ea-c8a87be7982e"

# Name of the calibration offsets file, and the folder it gets saved into.
# This is intentionally SEPARATE from the CSV/plots output_dir below, so the
# JSON can live in a different folder than the historical CSV export.
# CSV_NotLive_MixwithDroos.py's CALIBRATION_FILE constant must match this path.
CALIBRATION_FILENAME = "calibration_offsets.json"
CALIBRATION_DIR = r"D:\CoE 199\Final_Code_setup"


class AirDeviceManager:

    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.headers = {
            "Accept": "*/*",
            "X-API-KEY": api_key
        }

        # Define sensor mapping based on YOUR CORRECT MAPPING
        # Format: 'sensor_code': 'Display Name'
        self.sensor_mapping = {
            '88970c': 'Left Sensor 1',
            '2deb24': 'Left Sensor 2',
            '89e5f0': 'Left Sensor 3',
            'cc8f24': 'Left Sensor 4',
            '889720': 'Middle Sensor 1',
            '889b88': 'Middle Sensor 2',
            '87f510': 'Middle Sensor 3',
            '889938': 'Middle Sensor 4',
            '2da640': 'Middle Sensor 5',
            '88e85c': 'Middle Sensor 6',
            '89ea14': 'Middle Sensor 7',
            '89e548': 'Middle Sensor 8',
            '88e4c8': 'Right Sensor 1',
            '89e8d8': 'Right Sensor 2',
            '88e590': 'Right Sensor 3'
        }

        # Define the correct order of Air-1 sensors for CSV columns
        self.sensor_order = [
            '88970c', '2deb24', '89e5f0', 'cc8f24',  # Left Sensors 1-4
            '889720', '889b88', '87f510', '889938', '2da640', '88e85c', '89ea14', '89e548',  # Middle Sensors 1-8
            '88e4c8', '89e8d8', '88e590'  # Right Sensors 1-3
        ]

        # Track all devices
        self.air1_devices = []
        self.ag_one_devices = []

    # ==================== AIR-1 DEVICE METHODS ====================

    def get_all_air1_devices(self):
        """Find all active air-1 devices"""
        try:
            response = requests.get(f"{self.api_url}/air-1", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Air-1 request failed with status code {response.status_code}")
                return []
        except Exception as error:
            print(f"Error getting Air-1 devices: {error}")
            return []

    def get_air1_device_data(self, device_id):
        """Get latest air-1 data"""
        try:
            response = requests.get(f"{self.api_url}/air-1/{device_id}", headers=self.headers)

            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"Air-1 Device {device_id} has invalid json")
                        return None
                else:
                    print(f"Air-1 Device {device_id} has an empty response(no data)")
                    return None
            else:
                print(f"Air-1 Device {device_id} has a status code error {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Air-1 Device {device_id} has connection error {e}")
            return None

    def get_air1_historical_data(self, device_id, time_start, time_end):
        """Get historical data for a specific air-1 device within a time range"""
        try:
            start_str = time_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            start_encoded = urllib.parse.quote(start_str)
            end_encoded = urllib.parse.quote(end_str)

            url = f"{self.api_url}/air-1/{device_id}?time_start={start_encoded}&time_end={end_encoded}"

            print(f"\nRequesting Air-1 historical data from: {url}")

            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"Air-1 Device {device_id} has invalid json in historical data")
                        return None
                else:
                    print(f"Air-1 Device {device_id} has empty historical response")
                    return None
            else:
                print(f"Air-1 historical data request for device {device_id} failed with status code {response.status_code}")
                return None

        except Exception as e:
            print(f"Error getting Air-1 historical data for device {device_id}: {e}")
            return None

    # ==================== AIRGRADIENT ONE DEVICE METHODS ====================

    def get_all_ag_one_devices(self):
        """Find all active AirGradient One units"""
        try:
            response = requests.get(f"{self.api_url}/ag-one", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"AirGradient One request failed with status code {response.status_code}")
                return []
        except Exception as error:
            print(f"Error getting AirGradient One devices: {error}")
            return []

    def get_ag_one_device_data(self, device_id):
        """Get latest AirGradient One data"""
        try:
            response = requests.get(f"{self.api_url}/ag-one/{device_id}", headers=self.headers)

            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"AirGradient One Device {device_id} has invalid json")
                        return None
                else:
                    print(f"AirGradient One Device {device_id} has an empty response(no data)")
                    return None
            else:
                print(f"AirGradient One Device {device_id} has a status code error {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"AirGradient One Device {device_id} has connection error {e}")
            return None

    def get_ag_one_historical_data(self, device_id, time_start, time_end):
        """Get historical data for a specific AirGradient One device within a time range"""
        try:
            start_str = time_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            start_encoded = urllib.parse.quote(start_str)
            end_encoded = urllib.parse.quote(end_str)

            url = f"{self.api_url}/ag-one/{device_id}?time_start={start_encoded}&time_end={end_encoded}"

            print(f"\nRequesting AirGradient One historical data from: {url}")

            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"AirGradient One Device {device_id} has invalid json in historical data")
                        return None
                else:
                    print(f"AirGradient One Device {device_id} has empty historical response")
                    return None
            else:
                print(f"AirGradient One historical data request for device {device_id} failed with status code {response.status_code}")
                return None

        except Exception as e:
            print(f"Error getting AirGradient One historical data for device {device_id}: {e}")
            return None

    # ==================== COMMON METHODS ====================

    def convert_timestamp(self, timestamp_str, include_date=True):
        """Convert timestamp from API to local time (+8 hours)"""
        try:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '')

            if '.' in timestamp_str:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")

            dt_local = dt + timedelta(hours=8)

            if include_date:
                return dt_local.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return dt_local.strftime("%H:%M:%S")
        except Exception as e:
            print(f"Error converting timestamp {timestamp_str}: {e}")
            return timestamp_str

    def print_device_summary(self, device_id, device_type="air-1"):
        """Print summary for a device"""
        if device_type == "air-1":
            data = self.get_air1_device_data(device_id)
        else:
            data = self.get_ag_one_device_data(device_id)

        if data:
            print("\n________________________________")
            print(f"{device_type.upper()} Device: {device_id}")
            print("________________________________")
            print(data)

            original_timestamp = data.get('timestamp')
            if original_timestamp:
                local_time = self.convert_timestamp(original_timestamp, include_date=True)
                print(f"Original Timestamp: {original_timestamp}")
                print(f"Local Time (+8 hrs): {local_time}")
            else:
                print(f"Timestamp: {data.get('timestamp')}")

            print(f"Temperature: {data.get('temperature', 0):.2f}°C")
            print(f"Humidity: {data.get('humidity', 0):.1f}%")
            print(f"CO2: {data.get('co2')} ppm")
            print(f"PM2.5: {data.get('pm_2_5')} µg/m³")
            print("________________________________")
            return True
        else:
            print(f"No data available for {device_type} device {device_id}")
            return False

    def get_sensor_display_name(self, sensor_code):
        """Return display name for sensor based on mapping"""
        return self.sensor_mapping.get(sensor_code, sensor_code)

    def collect_device_data(self, time_start, time_end):
        """Collect historical data from all working devices"""
        all_data = {}

        print("\n" + "=" * 80)
        print("COLLECTING HISTORICAL DATA FROM ALL DEVICES")
        print("=" * 80)

        # Collect from Air-1 devices
        for device in self.air1_devices:
            if device not in self.sensor_order:
                print(f"⚠️ Warning: Air-1 Device {device} not in specified sensor order list, skipping")
                continue

            print(f"\nFetching Air-1 data from device: {device} ({self.get_sensor_display_name(device)})")
            historical_data = self.get_air1_historical_data(device, time_start, time_end)

            if historical_data:
                readings_list = historical_data if isinstance(historical_data, list) else [historical_data]

                for reading in readings_list:
                    timestamp = reading.get('timestamp')
                    if timestamp:
                        local_time = self.convert_timestamp(timestamp, include_date=True)

                        if local_time not in all_data:
                            all_data[local_time] = {}

                        all_data[local_time][device] = {
                            'temperature': reading.get('temperature', 'N/A'),
                            'humidity': reading.get('humidity', 'N/A'),
                            'co2': reading.get('co2', 'N/A'),
                            'pm25': reading.get('pm_2_5', 'N/A'),
                            'device_type': 'Air-1'
                        }

                print(f"  ✅ Found {len(readings_list)} readings for {self.get_sensor_display_name(device)} ({device})")
            else:
                print(f"  ❌ No data available for {self.get_sensor_display_name(device)} ({device})")

        # Collect from AirGradient One devices
        for device in self.ag_one_devices:
            print(f"\nFetching AirGradient One data from device: {device}")
            historical_data = self.get_ag_one_historical_data(device, time_start, time_end)

            if historical_data:
                readings_list = historical_data if isinstance(historical_data, list) else [historical_data]

                for reading in readings_list:
                    timestamp = reading.get('timestamp')
                    if timestamp:
                        local_time = self.convert_timestamp(timestamp, include_date=True)

                        if local_time not in all_data:
                            all_data[local_time] = {}

                        all_data[local_time][device] = {
                            'temperature': reading.get('temperature', 'N/A'),
                            'humidity': reading.get('humidity', 'N/A'),
                            'co2': reading.get('co2', 'N/A'),
                            'pm25': reading.get('pm_2_5', 'N/A'),
                            'device_type': 'AirGradient One'
                        }

                print(f"  ✅ Found {len(readings_list)} readings for Reference Sensor ({device})")
            else:
                print(f"  ❌ No data available for Reference Sensor ({device})")

        return all_data

    def calculate_and_print_averages(self, all_data, all_devices_ordered, ag_one_device_ids,
                                      calibration_output_path=None):
        """Calculate averages for each parameter, compare Air-1 sensors to Reference Sensor,
        and (if calibration_output_path is given) write per-sensor offsets to a JSON file.

        offset[sensor][param] = sensor_avg - ref_avg   (same "diff" printed in the table)
        CSV_NotLive_MixwithDroos.py applies: corrected_value = raw_value - offset
        """

        print("\n" + "=" * 80)
        print("CALCULATING AVERAGES AND COMPARISONS")
        print("=" * 80)

        # Separate Air-1 sensors and Reference Sensor
        air1_sensors = [d for d in all_devices_ordered if d not in ag_one_device_ids]
        ref_sensors = [d for d in all_devices_ordered if d in ag_one_device_ids]

        # Parameters to analyze
        parameters = ['temperature', 'humidity', 'co2', 'pm25']
        param_names = {'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'co2': 'CO2 (ppm)', 'pm25': 'PM2.5 (µg/m³)'}

        # Collect all values for each sensor and parameter
        sensor_values = {param: {sensor: [] for sensor in all_devices_ordered} for param in parameters}

        for timestamp in all_data:
            for sensor in all_devices_ordered:
                if sensor in all_data[timestamp]:
                    for param in parameters:
                        value = all_data[timestamp][sensor].get(param, 'N/A')
                        if isinstance(value, (int, float)):
                            sensor_values[param][sensor].append(value)

        # This is what gets written out to calibration_offsets.json
        # Structure: { sensor_id: { param: offset, ... }, ... }
        calibration_offsets = {sensor: {} for sensor in air1_sensors}
        reference_averages = {}

        # Calculate and print averages for each parameter
        for param in parameters:
            print(f"\n{'=' * 60}")
            print(f"PARAMETER: {param_names[param]}")
            print(f"{'=' * 60}")

            # Calculate average for Reference Sensor
            ref_avg = None
            for ref_sensor in ref_sensors:
                values = sensor_values[param][ref_sensor]
                if values:
                    ref_avg = np.mean(values)
                    print(f"\n📊 REFERENCE SENSOR (AirGradient One):")
                    print(f"   Average {param_names[param]}: {ref_avg:.2f}")

            reference_averages[param] = ref_avg if ref_avg is not None else None

            # Calculate averages for Air-1 sensors and differences
            if ref_avg is not None:
                print(f"\n📊 AIR-1 SENSORS (compared to Reference):")
                print(f"{'Sensor Code':<15} {'Sensor Name':<20} {'Average':<12} {'Difference (Air-1 - Ref)':<25}")
                print(f"{'-' * 70}")

                for sensor in air1_sensors:
                    values = sensor_values[param][sensor]
                    if values:
                        sensor_avg = np.mean(values)
                        diff = sensor_avg - ref_avg
                        display_name = self.get_sensor_display_name(sensor)
                        print(f"{sensor:<15} {display_name:<20} {sensor_avg:.2f}        {diff:+.2f}")

                        # Store the offset for calibration. corrected = raw - diff
                        calibration_offsets[sensor][param] = float(diff)
                    else:
                        display_name = self.get_sensor_display_name(sensor)
                        print(f"{sensor:<15} {display_name:<20} {'NO DATA':<12} {'N/A':<25}")
            else:
                print(f"\n⚠️ No Reference Sensor data available for {param_names[param]}")

        # Summary statistics
        print(f"\n{'=' * 80}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 80}")

        for param in parameters:
            print(f"\n📈 {param_names[param]}:")

            # Calculate stats for Air-1 sensors
            air1_avgs = []
            for sensor in air1_sensors:
                values = sensor_values[param][sensor]
                if values:
                    air1_avgs.append(np.mean(values))

            ref_avg = reference_averages.get(param)
            if air1_avgs and ref_avg:
                print(f"   Air-1 Sensors - Mean: {np.mean(air1_avgs):.2f}")
                print(f"   Air-1 Sensors - Std Dev: {np.std(air1_avgs):.2f}")
                print(f"   Air-1 Sensors - Min: {min(air1_avgs):.2f}")
                print(f"   Air-1 Sensors - Max: {max(air1_avgs):.2f}")
                print(f"   Reference Sensor: {ref_avg:.2f}")
                print(f"   Average Difference (Air-1 - Ref): {np.mean(air1_avgs) - ref_avg:+.2f}")

        print(f"\n{'=' * 80}")

        # ==================== SAVE CALIBRATION OFFSETS ====================
        if calibration_output_path:
            self.save_calibration_offsets(calibration_offsets, calibration_output_path)

        return calibration_offsets

    def save_calibration_offsets(self, calibration_offsets, output_path):
        """Write calibration offsets to a JSON file that the main data collection
        script (CSV_NotLive_MixwithDroos.py) can load and apply automatically.

        Only sensors/parameters that actually had reference data and readings
        during this calibration run are written. Sensors with no offset entry
        will simply be left uncorrected by the main script.
        """
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            payload = {
                "generated_at": (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                "note": "offset = sensor_avg - reference_avg. Apply as corrected = raw - offset.",
                "offsets": calibration_offsets
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

            print(f"\n✅ Calibration offsets saved to: {os.path.abspath(output_path)}")
            num_calibrated = sum(1 for sensor, params in calibration_offsets.items() if params)
            print(f"   Sensors with at least one calibrated parameter: {num_calibrated}")

        except Exception as e:
            print(f"❌ Error saving calibration offsets: {e}")

    def export_all_historical_to_single_csv(self, time_start, time_end, output_dir="historical_data"):
        """Export historical data from ALL devices to a SINGLE CSV file with clean format"""
        all_data = self.collect_device_data(time_start, time_end)

        if not all_data:
            print("\n❌ No readings collected from any device")
            return None, None, None

        sorted_timestamps = sorted(list(all_data.keys()))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\nCreated directory: {output_dir}")

        start_str = time_start.strftime("%Y%m%d_%H%M%S")
        end_str = time_end.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/ReferenceSensor_{start_str}_to_{end_str}.csv"

        # Get all unique device IDs from the data
        all_device_ids = set()
        for timestamp_data in all_data.values():
            for device_id in timestamp_data.keys():
                all_device_ids.add(device_id)

        # Order devices: first the ones in sensor_order that exist, then AG-One devices
        air1_in_data = [d for d in self.sensor_order if d in all_device_ids]
        ag_one_in_data = [d for d in all_device_ids if d not in self.sensor_order]
        all_devices_ordered = air1_in_data + ag_one_in_data

        print(f"\n📊 Devices to export: {len(all_devices_ordered)}")
        print(f"   Air-1 devices: {len(air1_in_data)}")
        print(f"   AirGradient One devices: {len(ag_one_in_data)}")

        # Print detailed mapping
        print("\n📋 DETAILED SENSOR MAPPING FOR THIS EXPORT:")
        for device in air1_in_data:
            print(f"   {device} -> {self.get_sensor_display_name(device)}")
        for device in ag_one_in_data:
            print(f"   {device} -> Reference Sensor")

        # Check for missing expected sensors
        expected_sensors = set(self.sensor_order)
        found_sensors = set(air1_in_data)
        missing_sensors = expected_sensors - found_sensors
        if missing_sensors:
            print(f"\n⚠️ WARNING: The following expected sensors were NOT found in the data:")
            for sensor in missing_sensors:
                print(f"   {sensor} -> {self.get_sensor_display_name(sensor)}")

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Create a SINGLE header row with all columns (no duplicate timestamp columns)
                header_row = ['Timestamp']

                # Temperature columns (Air-1 sensors + Reference Sensor at the end)
                for device in all_devices_ordered:
                    header_row.append(f'Temp_{device}')

                # Humidity columns
                for device in all_devices_ordered:
                    header_row.append(f'RH_{device}')

                # CO2 columns
                for device in all_devices_ordered:
                    header_row.append(f'CO2_{device}')

                # PM2.5 columns
                for device in all_devices_ordered:
                    header_row.append(f'PM25_{device}')

                writer.writerow(header_row)

                # Write data rows
                for timestamp in sorted_timestamps:
                    row = [timestamp]

                    # Temperature data
                    for device in all_devices_ordered:
                        if device in all_data[timestamp]:
                            value = all_data[timestamp][device].get('temperature', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.2f}"
                            row.append(value)
                        else:
                            row.append('N/A')

                    # Humidity data
                    for device in all_devices_ordered:
                        if device in all_data[timestamp]:
                            value = all_data[timestamp][device].get('humidity', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.1f}"
                            row.append(value)
                        else:
                            row.append('N/A')

                    # CO2 data
                    for device in all_devices_ordered:
                        if device in all_data[timestamp]:
                            value = all_data[timestamp][device].get('co2', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{int(value)}"
                            row.append(value)
                        else:
                            row.append('N/A')

                    # PM2.5 data
                    for device in all_devices_ordered:
                        if device in all_data[timestamp]:
                            value = all_data[timestamp][device].get('pm25', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.1f}"
                            row.append(value)
                        else:
                            row.append('N/A')

                    writer.writerow(row)

            print("\n" + "=" * 80)
            print(f"✅ EXPORT COMPLETE!")
            print(f"📊 Total timestamps exported: {len(sorted_timestamps)}")
            print(f"📁 File saved as: {filename}")
            print(f"📍 Full path: {os.path.abspath(filename)}")
            print("\n📋 NEW CSV STRUCTURE:")
            print(f"   - Single timestamp column at the beginning")
            print(f"   - Temperature data: {len(all_devices_ordered)} columns")
            print(f"   - Humidity data: {len(all_devices_ordered)} columns")
            print(f"   - CO2 data: {len(all_devices_ordered)} columns")
            print(f"   - PM2.5 data: {len(all_devices_ordered)} columns")
            print(f"   Total columns: 1 + {4 * len(all_devices_ordered)}")
            print("=" * 80)

            # Calculate averages, print comparison table, AND save calibration_offsets.json
            # NOTE: this JSON goes to CALIBRATION_DIR, which is independent of
            # output_dir (the CSV/plots folder) - they can be different paths.
            calibration_path = os.path.join(CALIBRATION_DIR, CALIBRATION_FILENAME)
            self.calculate_and_print_averages(all_data, all_devices_ordered, ag_one_in_data,
                                               calibration_output_path=calibration_path)

            return filename, all_devices_ordered, ag_one_in_data

        except Exception as e:
            print(f"❌ Error writing CSV file: {e}")
            return None, None, None

    def generate_plots(self, csv_filename, all_devices_ordered, ag_one_device_ids, output_dir="historical_data"):
        """Generate line plots for Temperature, RH, CO2, and PM2.5 from the combined CSV"""
        print("\n" + "=" * 80)
        print("GENERATING PLOTS FROM CSV DATA")
        print("=" * 80)

        import re
        date_match = re.search(r'(\d{8}_\d{6})_to_(\d{8}_\d{6})', csv_filename)
        if date_match:
            start_date_str = date_match.group(1)
            end_date_str = date_match.group(2)
            date_range = f"{start_date_str}_to_{end_date_str}"
        else:
            date_range = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Number of sensors (15 Air-1 + Reference Sensor = 16 total)
        num_sensors = len(all_devices_ordered)

        temp_start_col = 1
        temp_end_col = num_sensors

        rh_start_col = num_sensors + 1
        rh_end_col = 2 * num_sensors

        co2_start_col = 2 * num_sensors + 1
        co2_end_col = 3 * num_sensors

        pm25_start_col = 3 * num_sensors + 1
        pm25_end_col = 4 * num_sensors

        print(f"\n📊 NEW CSV STRUCTURE DEBUG:")
        print(f"   Total number of sensors (including Reference): {num_sensors}")
        print(f"   Temperature columns: {temp_start_col} to {temp_end_col}")
        print(f"   Humidity columns: {rh_start_col} to {rh_end_col}")
        print(f"   CO2 columns: {co2_start_col} to {co2_end_col}")
        print(f"   PM2.5 columns: {pm25_start_col} to {pm25_end_col}")

        # Initialize data storage
        timestamps = []
        temp_data = {device: [] for device in all_devices_ordered}
        rh_data = {device: [] for device in all_devices_ordered}
        co2_data = {device: [] for device in all_devices_ordered}
        pm25_data = {device: [] for device in all_devices_ordered}

        try:
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)

                print(f"\n📋 CSV HEADERS (first {min(20, len(headers))} columns):")
                for i, h in enumerate(headers[:20]):
                    print(f"   Column {i}: {h}")

                row_count = 0
                for row in reader:
                    if not row:
                        continue

                    row_count += 1

                    # Get timestamp from column 0
                    timestamp_str = row[0] if len(row) > 0 else None
                    if timestamp_str:
                        timestamps.append(timestamp_str)

                    # Temperature data
                    for i, device in enumerate(all_devices_ordered):
                        col_idx = temp_start_col + i
                        if len(row) > col_idx:
                            try:
                                val = float(row[col_idx]) if row[col_idx] != 'N/A' else None
                                temp_data[device].append(val)
                            except:
                                temp_data[device].append(None)
                        else:
                            temp_data[device].append(None)

                    # Humidity data
                    for i, device in enumerate(all_devices_ordered):
                        col_idx = rh_start_col + i
                        if len(row) > col_idx:
                            try:
                                val = float(row[col_idx]) if row[col_idx] != 'N/A' else None
                                rh_data[device].append(val)
                            except:
                                rh_data[device].append(None)
                        else:
                            rh_data[device].append(None)

                    # CO2 data
                    for i, device in enumerate(all_devices_ordered):
                        col_idx = co2_start_col + i
                        if len(row) > col_idx:
                            try:
                                val = float(row[col_idx]) if row[col_idx] != 'N/A' else None
                                co2_data[device].append(val)
                            except:
                                co2_data[device].append(None)
                        else:
                            co2_data[device].append(None)

                    # PM2.5 data
                    for i, device in enumerate(all_devices_ordered):
                        col_idx = pm25_start_col + i
                        if len(row) > col_idx:
                            try:
                                val = float(row[col_idx]) if row[col_idx] != 'N/A' else None
                                pm25_data[device].append(val)
                            except:
                                pm25_data[device].append(None)
                        else:
                            pm25_data[device].append(None)

            print(f"\n📊 Total rows read: {row_count}")
            print(f"📊 Total timestamps: {len(timestamps)}")

            if not timestamps:
                print("❌ No data found in CSV file")
                return False

            # Convert timestamps to datetime objects
            datetime_timestamps = []
            for ts in timestamps:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    datetime_timestamps.append(dt)
                except:
                    datetime_timestamps.append(None)

            # Determine if multi-day
            if len(datetime_timestamps) > 1 and datetime_timestamps[0] and datetime_timestamps[-1]:
                first_date = datetime_timestamps[0]
                last_date = datetime_timestamps[-1]
                days_span = (last_date - first_date).days
                is_multi_day = days_span >= 1
            else:
                is_multi_day = False
                first_date = datetime.now()
                last_date = datetime.now()

            # Create plots directory
            plots_dir = os.path.join(output_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                print(f"\nCreated plots directory: {plots_dir}")

            # Debug: Check data availability for each sensor
            print(f"\n🔍 DATA AVAILABILITY CHECK:")
            for device in all_devices_ordered:
                display_name = self.get_sensor_display_name(device) if device in self.sensor_mapping else 'Reference Sensor'
                temp_count = sum(1 for v in temp_data[device] if v is not None)
                rh_count = sum(1 for v in rh_data[device] if v is not None)
                co2_count = sum(1 for v in co2_data[device] if v is not None)
                pm25_count = sum(1 for v in pm25_data[device] if v is not None)
                print(f"   {device} ({display_name}): Temp={temp_count}, RH={rh_count}, CO2={co2_count}, PM2.5={pm25_count}")

            # Get colors for all devices
            colors = plt.cm.tab20.colors[:num_sensors]

            # ==================== PLOT 1: TEMPERATURE ====================
            print("\n📊 Generating Temperature plot...")
            fig, ax = plt.subplots(figsize=(16, 10))

            for idx, device in enumerate(all_devices_ordered):
                if any(v is not None for v in temp_data[device]):
                    if device in ag_one_device_ids:
                        ax.plot(datetime_timestamps, temp_data[device],
                               label='Reference Sensor', color='red',
                               linewidth=3.5, marker='o', markersize=4,
                               linestyle='-', alpha=0.9)
                    else:
                        display_name = self.get_sensor_display_name(device)
                        ax.plot(datetime_timestamps, temp_data[device],
                               label=display_name, color=colors[idx % len(colors)],
                               linewidth=1.5, marker='.', markersize=3)

            ax.set_xlabel('Date & Time' if is_multi_day else 'Time', fontsize=12)
            ax.set_ylabel('Temperature (°C)', fontsize=12)

            if is_multi_day:
                ax.set_title(f'Temperature Over Time - All Sensors (Multi-Day: {first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
            else:
                ax.set_title(f'Temperature Over Time - All Sensors ({first_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.xticks(rotation=45)

            ax.legend(loc='upper right', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            temp_plot_path = os.path.join(plots_dir, f'temperature_plot_{date_range}.png')
            plt.savefig(temp_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved to: {temp_plot_path}")

            # ==================== PLOT 2: RELATIVE HUMIDITY ====================
            print("\n📊 Generating Relative Humidity plot...")
            fig, ax = plt.subplots(figsize=(16, 10))

            for idx, device in enumerate(all_devices_ordered):
                if any(v is not None for v in rh_data[device]):
                    if device in ag_one_device_ids:
                        ax.plot(datetime_timestamps, rh_data[device],
                               label='Reference Sensor', color='red',
                               linewidth=3.5, marker='o', markersize=4,
                               linestyle='-', alpha=0.9)
                    else:
                        display_name = self.get_sensor_display_name(device)
                        ax.plot(datetime_timestamps, rh_data[device],
                               label=display_name, color=colors[idx % len(colors)],
                               linewidth=1.5, marker='.', markersize=3)

            ax.set_xlabel('Date & Time' if is_multi_day else 'Time', fontsize=12)
            ax.set_ylabel('Relative Humidity (%)', fontsize=12)

            if is_multi_day:
                ax.set_title(f'Relative Humidity Over Time - All Sensors (Multi-Day: {first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
            else:
                ax.set_title(f'Relative Humidity Over Time - All Sensors ({first_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.xticks(rotation=45)

            ax.legend(loc='upper right', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            rh_plot_path = os.path.join(plots_dir, f'humidity_plot_{date_range}.png')
            plt.savefig(rh_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved to: {rh_plot_path}")

            # ==================== PLOT 3: CO2 ====================
            print("\n📊 Generating CO2 plot...")
            fig, ax = plt.subplots(figsize=(16, 10))

            for idx, device in enumerate(all_devices_ordered):
                if any(v is not None for v in co2_data[device]):
                    if device in ag_one_device_ids:
                        ax.plot(datetime_timestamps, co2_data[device],
                               label='Reference Sensor', color='red',
                               linewidth=3.5, marker='o', markersize=4,
                               linestyle='-', alpha=0.9)
                    else:
                        display_name = self.get_sensor_display_name(device)
                        ax.plot(datetime_timestamps, co2_data[device],
                               label=display_name, color=colors[idx % len(colors)],
                               linewidth=1.5, marker='.', markersize=3)

            ax.set_xlabel('Date & Time' if is_multi_day else 'Time', fontsize=12)
            ax.set_ylabel('CO2 (ppm)', fontsize=12)

            if is_multi_day:
                ax.set_title(f'CO2 Over Time - All Sensors (Multi-Day: {first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
            else:
                ax.set_title(f'CO2 Over Time - All Sensors ({first_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.xticks(rotation=45)

            ax.legend(loc='upper right', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            co2_plot_path = os.path.join(plots_dir, f'co2_plot_{date_range}.png')
            plt.savefig(co2_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved to: {co2_plot_path}")

            # ==================== PLOT 4: PM2.5 ====================
            print("\n📊 Generating PM2.5 plot...")
            fig, ax = plt.subplots(figsize=(16, 10))

            for idx, device in enumerate(all_devices_ordered):
                if any(v is not None for v in pm25_data[device]):
                    if device in ag_one_device_ids:
                        ax.plot(datetime_timestamps, pm25_data[device],
                               label='Reference Sensor', color='red',
                               linewidth=3.5, marker='o', markersize=4,
                               linestyle='-', alpha=0.9)
                    else:
                        display_name = self.get_sensor_display_name(device)
                        ax.plot(datetime_timestamps, pm25_data[device],
                               label=display_name, color=colors[idx % len(colors)],
                               linewidth=1.5, marker='.', markersize=3)

            ax.set_xlabel('Date & Time' if is_multi_day else 'Time', fontsize=12)
            ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)

            if is_multi_day:
                ax.set_title(f'PM2.5 Over Time - All Sensors (Multi-Day: {first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
            else:
                ax.set_title(f'PM2.5 Over Time - All Sensors ({first_date.strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.xticks(rotation=45)

            ax.legend(loc='upper right', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pm25_plot_path = os.path.join(plots_dir, f'pm25_plot_{date_range}.png')
            plt.savefig(pm25_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved to: {pm25_plot_path}")

            print("\n" + "=" * 80)
            print(f"✅ ALL PLOTS GENERATED SUCCESSFULLY!")
            print(f"📁 Plots saved in: {os.path.abspath(plots_dir)}")
            print(f"📅 Data span: {len(timestamps)} readings")
            print(f"\n📊 Generated plots with date stamp: {date_range}")
            print("   - temperature_plot_{date_range}.png")
            print("   - humidity_plot_{date_range}.png")
            print("   - co2_plot_{date_range}.png")
            print("   - pm25_plot_{date_range}.png")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    manager = AirDeviceManager(API_URL, API_KEY)

    # Get all Air-1 devices
    air1_devices_list = manager.get_all_air1_devices()
    print(f"Found {len(air1_devices_list)} Air-1 devices from API: {air1_devices_list}")

    # Get all AirGradient One devices
    ag_one_devices_list = manager.get_all_ag_one_devices()
    print(f"Found {len(ag_one_devices_list)} AirGradient One devices from API: {ag_one_devices_list}")

    if not air1_devices_list and not ag_one_devices_list:
        print("No devices found")
        return

    # Test Air-1 devices
    print("\n" + "=" * 80)
    print("TESTING AIR-1 DEVICES")
    print("=" * 80)
    working_air1 = []
    for device in air1_devices_list:
        data = manager.get_air1_device_data(device)
        if data and 'timestamp' in data:
            working_air1.append(device)
            print(f"✅ Air-1 Device {device} ({manager.get_sensor_display_name(device)}) is working")
        else:
            print(f"❌ Air-1 Device {device} ({manager.get_sensor_display_name(device)}) is inactive")

    # Test AirGradient One devices
    print("\n" + "=" * 80)
    print("TESTING AIRGRADIENT ONE DEVICES")
    print("=" * 80)
    working_ag_one = []
    for device in ag_one_devices_list:
        data = manager.get_ag_one_device_data(device)
        if data and 'timestamp' in data:
            working_ag_one.append(device)
            print(f"✅ AirGradient One Device {device} is working")
        else:
            print(f"❌ AirGradient One Device {device} is inactive")

    manager.air1_devices = working_air1
    manager.ag_one_devices = working_ag_one

    print(f"\n📊 SUMMARY:")
    print(f"   Air-1 active devices: {len(working_air1)}")
    for device in working_air1:
        print(f"      - {device} ({manager.get_sensor_display_name(device)})")
    print(f"   AirGradient One active devices: {len(working_ag_one)}")

    # Check for missing expected sensors
    expected_all = set(manager.sensor_order)
    found = set(working_air1)
    missing = expected_all - found
    if missing:
        print(f"\n⚠️ MISSING EXPECTED SENSORS (not found in API response):")
        for sensor in missing:
            print(f"   {sensor} ({manager.get_sensor_display_name(sensor)})")

    if working_air1 or working_ag_one:
        print("\n" + "=" * 80)
        print("LATEST DATA FROM ALL SENSORS")
        print("=" * 80)

        for device in working_air1:
            manager.print_device_summary(device, "air-1")
        for device in working_ag_one:
            manager.print_device_summary(device, "ag-one")

        print("\n" + "=" * 80)
        print("EXPORTING DATA IN MATRIX FORMAT")
        print("=" * 80)

        try:
            time_start = datetime(2026, 6, 15, 0, 0, 0)
            time_end = datetime(2026, 6, 23, 0, 0, 0)

            print(f"\nTime range for export (UTC): {time_start} to {time_end}")

            output_dir = r"D:\CoE 199\data_199"
            csv_file, all_devices, ag_one_ids = manager.export_all_historical_to_single_csv(time_start, time_end, output_dir)

            if csv_file:
                manager.generate_plots(csv_file, all_devices, ag_one_ids, output_dir)
            else:
                print("\n❌ No file was exported")

        except Exception as e:
            print(f"Error in historical data export: {e}")

    else:
        print("\nNo devices are working")


if __name__ == "__main__":
    main()