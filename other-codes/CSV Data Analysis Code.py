import requests
import json
from datetime import datetime, timedelta
import urllib.parse
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# Set font for better rendering
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

#API Credentials
API_URL = "http://10.158.66.30:80"
API_KEY = "3a21fe5a-78cb-4252-99ea-c8a87be7982e"

class Air1Device:
   
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.headers = {
            "Accept": "*/*",
            "X-API-KEY": api_key
        }
        
        # Define sensor mapping based on the image layout
        # Format: 'sensor_code': 'Display Name'
        self.sensor_mapping = {
            '88970c': 'Left Sensor 1',
            '88e4c8': 'Right Sensor 1',
            '88e590': 'Right Sensor 3',
            '89e8d8': 'Right Sensor 2',
            '889720': 'Middle Sensor 1',
            '87f510': 'Middle Sensor 3',
            '2da640': 'Middle Sensor 5',
            '89ea14': 'Middle Sensor 7',
            '889b88': 'Middle Sensor 2',
            '889938': 'Middle Sensor 4',
            '88e85c': 'Middle Sensor 6',
            '89e548': 'Middle Sensor 8',
            '2deb24': 'Left Sensor 2',
            '89e5f0': 'Left Sensor 3',
            'cc8f24': 'Left Sensor 4'
        }
        
        # Define the exact order of sensors (based on the image layout)
        self.sensor_order = [
            '88e4c8', '88e590', '89e8d8', '889720', '87f510', '2da640', 
            '89ea14', '889b88', '889938', '88e85c', '89e548', '88970c', 
            '2deb24', '89e5f0', 'cc8f24'
        ]
   
    def get_all_devices(self):
        #find all active air-1 devices
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
        #get latest air-1 data
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
                    print(f"Device {device_id} has an empty response(no data)")
                    return None
            else:
                print(f"Device {device_id} has a status code error {response.status_code}")
                return None
               
        except requests.exceptions.RequestException as e:
            print(f"Device {device_id} has connection error {e}")
            return None
   
    def get_historical_data(self, device_id, time_start, time_end):
        """
        Get historical data for a specific air-1 device within a time range
        time_start and time_end should be datetime objects
        Uses the same endpoint structure but with time parameters
        """
        try:
            # Format datetime objects to ISO 8601 format
            start_str = time_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
           
            # URL encode the timestamps (colon : becomes %3A)
            start_encoded = urllib.parse.quote(start_str)
            end_encoded = urllib.parse.quote(end_str)
           
            # Construct the URL with air-1 endpoint and time parameters
            url = f"{self.api_url}/air-1/{device_id}?time_start={start_encoded}&time_end={end_encoded}"
           
            print(f"\nRequesting historical data from: {url}")
           
            response = requests.get(url, headers=self.headers)
           
            if response.status_code == 200:
                if response.text and response.text.strip():
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"Device {device_id} has invalid json in historical data")
                        return None
                else:
                    print(f"Device {device_id} has empty historical response")
                    return None
            else:
                print(f"Historical data request for device {device_id} failed with status code {response.status_code}")
                print(f"Response text: {response.text}")
                return None
               
        except Exception as e:
            print(f"Error getting historical data for device {device_id}: {e}")
            return None
   
    def convert_timestamp(self, timestamp_str, include_date=True):
        """
        Convert timestamp from API to local time (+8 hours)
        Input: "2026-03-27T23:00:00.000Z"
        Output: "2026-03-28 07:00:00" (after adding 8 hours) if include_date=True
                "07:00:00" if include_date=False
        """
        try:
            # Parse the ISO timestamp string
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '')
            
            # Parse the datetime
            if '.' in timestamp_str:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            
            # Add 8 hours for timezone conversion
            dt_local = dt + timedelta(hours=8)
            
            # Return full datetime or just time
            if include_date:
                return dt_local.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return dt_local.strftime("%H:%M:%S")
        except Exception as e:
            print(f"Error converting timestamp {timestamp_str}: {e}")
            return timestamp_str
    
    def print_device_summary(self, device_id):
        data = self.get_device_data(device_id)
        if data:
            print("\n" + "____________________________________________")
            print(f"AIR-1 Device: {device_id}")
            print("____________________________________________")
            print(data)
            
            # Convert timestamp to local time
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
            print("____________________________________________")
            return True
        else:
            print(f"No data available for device {device_id}")
            return False
   
    def get_sensor_display_name(self, sensor_code):
        """Return display name for sensor based on mapping"""
        return self.sensor_mapping.get(sensor_code, sensor_code)
   
    def generate_plots(self, csv_filename, output_dir="historical_data"):
        """
        Generate line plots for Temperature, RH, CO2, and PM2.5
        Automatically handles both single-day and multi-day data
        X-axis shows time for single day, datetime for multiple days
        Plot filenames include date range for easy identification
        Legend uses friendly sensor names instead of codes
        """
        print("\n" + "="*80)
        print("GENERATING PLOTS FROM CSV DATA")
        print("="*80)
        
        # Extract date range from csv_filename for plot naming
        import re
        date_match = re.search(r'(\d{8}_\d{6})_to_(\d{8}_\d{6})', csv_filename)
        if date_match:
            start_date_str = date_match.group(1)
            end_date_str = date_match.group(2)
            date_range = f"{start_date_str}_to_{end_date_str}"
        else:
            # Fallback: use current date
            date_range = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Read the CSV file
        timestamps = []
        temp_data = {sensor: [] for sensor in self.sensor_order}
        rh_data = {sensor: [] for sensor in self.sensor_order}
        co2_data = {sensor: [] for sensor in self.sensor_order}
        pm25_data = {sensor: [] for sensor in self.sensor_order}
        
        try:
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Skip header row
                
                for row in reader:
                    if not row:
                        continue
                    
                    # Get timestamp from column A (now contains full datetime)
                    timestamp_str = row[0] if len(row) > 0 else None
                    if timestamp_str:
                        timestamps.append(timestamp_str)
                    
                    # Temperature data (columns 1-15)
                    for i, sensor in enumerate(self.sensor_order):
                        if len(row) > i + 1:
                            try:
                                val = float(row[i + 1]) if row[i + 1] != 'N/A' else None
                                temp_data[sensor].append(val)
                            except:
                                temp_data[sensor].append(None)
                    
                    # Humidity data (columns 17-31)
                    for i, sensor in enumerate(self.sensor_order):
                        if len(row) > i + 17:
                            try:
                                val = float(row[i + 17]) if row[i + 17] != 'N/A' else None
                                rh_data[sensor].append(val)
                            except:
                                rh_data[sensor].append(None)
                    
                    # CO2 data (columns 33-47)
                    for i, sensor in enumerate(self.sensor_order):
                        if len(row) > i + 33:
                            try:
                                val = float(row[i + 33]) if row[i + 33] != 'N/A' else None
                                co2_data[sensor].append(val)
                            except:
                                co2_data[sensor].append(None)
                    
                    # PM2.5 data (columns 49-63)
                    for i, sensor in enumerate(self.sensor_order):
                        if len(row) > i + 49:
                            try:
                                val = float(row[i + 49]) if row[i + 49] != 'N/A' else None
                                pm25_data[sensor].append(val)
                            except:
                                pm25_data[sensor].append(None)
            
            if not timestamps:
                print("❌ No data found in CSV file")
                return False
            
            # Convert timestamps to datetime objects for plotting
            datetime_timestamps = []
            for ts in timestamps:
                try:
                    # Try to parse full datetime string
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    datetime_timestamps.append(dt)
                except:
                    # Fallback: try parsing as just time
                    try:
                        # If only time, use a base date
                        base_date = datetime.now().replace(year=2024, month=1, day=1)
                        hours, minutes, seconds = map(int, ts.split(':'))
                        dt = base_date.replace(hour=hours, minute=minutes, second=seconds)
                        datetime_timestamps.append(dt)
                    except:
                        datetime_timestamps.append(None)
            
            # Determine if we have multiple days of data
            if len(datetime_timestamps) > 1 and datetime_timestamps[0] and datetime_timestamps[-1]:
                first_date = datetime_timestamps[0]
                last_date = datetime_timestamps[-1]
                days_span = (last_date - first_date).days
                is_multi_day = days_span >= 1
            else:
                is_multi_day = False
            
            # Create output directory for plots if it doesn't exist
            plots_dir = os.path.join(output_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                print(f"\nCreated plots directory: {plots_dir}")
            
            # Generate color map for 15 sensors
            colors = plt.cm.tab20.colors[:15]
            
            # Plot 1: Temperature
            print("\n📊 Generating Temperature plot...")
            fig, ax = plt.subplots(figsize=(14, 8))
            for idx, sensor in enumerate(self.sensor_order):
                if any(v is not None for v in temp_data[sensor]):
                    # Use display name for legend
                    display_name = self.get_sensor_display_name(sensor)
                    ax.plot(datetime_timestamps, temp_data[sensor], 
                           label=display_name, color=colors[idx], linewidth=1.5, marker='.', markersize=3)
            
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
            
            # Plot 2: Relative Humidity
            print("\n📊 Generating Relative Humidity plot...")
            fig, ax = plt.subplots(figsize=(14, 8))
            for idx, sensor in enumerate(self.sensor_order):
                if any(v is not None for v in rh_data[sensor]):
                    # Use display name for legend
                    display_name = self.get_sensor_display_name(sensor)
                    ax.plot(datetime_timestamps, rh_data[sensor], 
                           label=display_name, color=colors[idx], linewidth=1.5, marker='.', markersize=3)
            
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
            
            # Plot 3: CO2
            print("\n📊 Generating CO2 plot...")
            fig, ax = plt.subplots(figsize=(14, 8))
            for idx, sensor in enumerate(self.sensor_order):
                if any(v is not None for v in co2_data[sensor]):
                    # Use display name for legend
                    display_name = self.get_sensor_display_name(sensor)
                    ax.plot(datetime_timestamps, co2_data[sensor], 
                           label=display_name, color=colors[idx], linewidth=1.5, marker='.', markersize=3)
            
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
            
            # Plot 4: PM2.5
            print("\n📊 Generating PM2.5 plot...")
            fig, ax = plt.subplots(figsize=(14, 8))
            for idx, sensor in enumerate(self.sensor_order):
                if any(v is not None for v in pm25_data[sensor]):
                    # Use display name for legend
                    display_name = self.get_sensor_display_name(sensor)
                    ax.plot(datetime_timestamps, pm25_data[sensor], 
                           label=display_name, color=colors[idx], linewidth=1.5, marker='.', markersize=3)
            
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
            
            print("\n" + "="*80)
            print(f"✅ ALL PLOTS GENERATED SUCCESSFULLY!")
            print(f"📁 Plots saved in: {os.path.abspath(plots_dir)}")
            if is_multi_day:
                print(f"📅 Data span: {len(timestamps)} readings from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({days_span + 1} day(s))")
            else:
                print(f"📅 Data span: {len(timestamps)} readings on {first_date.strftime('%Y-%m-%d')}")
            print(f"\n📊 Generated plots with date stamp: {date_range}")
            print("   - temperature_plot_{date_range}.png")
            print("   - humidity_plot_{date_range}.png")
            print("   - co2_plot_{date_range}.png")
            print("   - pm25_plot_{date_range}.png")
            print("\n🔍 Legend Labels:")
            for sensor in self.sensor_order:
                print(f"   {sensor} → {self.get_sensor_display_name(sensor)}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            return False
   
    def export_all_historical_to_single_csv(self, devices, time_start, time_end, output_dir="historical_data"):
        """
        Export historical data with ALL parameters in the SAME rows:
        - Columns A: Timestamp (with date and time)
        - Columns B-P: Temperature data for 15 sensors
        - Columns Q: Timestamp (repeated for humidity section)
        - Columns R-AF: Humidity data for 15 sensors
        - Columns AG: Timestamp (repeated for CO2 section)
        - Columns AH-AV: CO2 data for 15 sensors
        - Columns AW: Timestamp (repeated for PM2.5 section)
        - Columns AX-BL: PM2.5 data for 15 sensors
        
        ALL data for the same timestamp appears in the SAME ROW
        """
        
        # Store data by timestamp
        all_data = {}
        
        print("\n" + "="*80)
        print("COLLECTING HISTORICAL DATA FROM ALL DEVICES")
        print("="*80)
        
        # Collect readings from all devices
        for device in devices:
            if device not in self.sensor_order:
                print(f"⚠️ Warning: Device {device} not in specified sensor order list, skipping")
                continue
                
            print(f"\nFetching data from device: {device}")
            historical_data = self.get_historical_data(device, time_start, time_end)
            
            if historical_data:
                readings_list = historical_data if isinstance(historical_data, list) else [historical_data]
                
                for reading in readings_list:
                    timestamp = reading.get('timestamp')
                    if timestamp:
                        # Convert timestamp to local time with date included
                        local_time = self.convert_timestamp(timestamp, include_date=True)
                        
                        # Initialize timestamp entry if not exists
                        if local_time not in all_data:
                            all_data[local_time] = {}
                        
                        # Store all parameters for this sensor at this timestamp
                        all_data[local_time][device] = {
                            'temperature': reading.get('temperature', 'N/A'),
                            'humidity': reading.get('humidity', 'N/A'),
                            'co2': reading.get('co2', 'N/A'),
                            'pm25': reading.get('pm_2_5', 'N/A')
                        }
                
                print(f"  ✅ Found {len(readings_list)} readings for device {device}")
            else:
                print(f"  ❌ No data available for device {device}")
        
        if not all_data:
            print("\n❌ No readings collected from any device")
            return None
        
        # Sort timestamps chronologically
        sorted_timestamps = sorted(list(all_data.keys()))
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\nCreated directory: {output_dir}")
        
        # Create filename with timestamp range
        start_str = time_start.strftime("%Y%m%d_%H%M%S")
        end_str = time_end.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/sensor_data_analysis_{start_str}_to_{end_str}.csv"
        
        # Write to CSV file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create the header row with all columns
                header_row = []
                
                # Temperature section header
                header_row.append('Timestamp_Temp')
                for sensor in self.sensor_order:
                    header_row.append(f'Temp_{sensor}')
                
                # Humidity section header
                header_row.append('Timestamp_RH')
                for sensor in self.sensor_order:
                    header_row.append(f'RH_{sensor}')
                
                # CO2 section header
                header_row.append('Timestamp_CO2')
                for sensor in self.sensor_order:
                    header_row.append(f'CO2_{sensor}')
                
                # PM2.5 section header
                header_row.append('Timestamp_PM25')
                for sensor in self.sensor_order:
                    header_row.append(f'PM25_{sensor}')
                
                writer.writerow(header_row)
                
                # Write data rows (one row per timestamp)
                for timestamp in sorted_timestamps:
                    row = []
                    
                    # Temperature section (Columns A-P)
                    row.append(timestamp)
                    for sensor in self.sensor_order:
                        if sensor in all_data[timestamp]:
                            value = all_data[timestamp][sensor].get('temperature', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.2f}"
                            row.append(value)
                        else:
                            row.append('N/A')
                    
                    # Humidity section (Columns Q-AF)
                    row.append(timestamp)
                    for sensor in self.sensor_order:
                        if sensor in all_data[timestamp]:
                            value = all_data[timestamp][sensor].get('humidity', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.1f}"
                            row.append(value)
                        else:
                            row.append('N/A')
                    
                    # CO2 section (Columns AG-AV)
                    row.append(timestamp)
                    for sensor in self.sensor_order:
                        if sensor in all_data[timestamp]:
                            value = all_data[timestamp][sensor].get('co2', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{int(value)}"
                            row.append(value)
                        else:
                            row.append('N/A')
                    
                    # PM2.5 section (Columns AW-BL)
                    row.append(timestamp)
                    for sensor in self.sensor_order:
                        if sensor in all_data[timestamp]:
                            value = all_data[timestamp][sensor].get('pm25', 'N/A')
                            if isinstance(value, (int, float)):
                                value = f"{value:.1f}"
                            row.append(value)
                        else:
                            row.append('N/A')
                    
                    writer.writerow(row)
            
            print("\n" + "="*80)
            print(f"✅ EXPORT COMPLETE!")
            print(f"📊 Total timestamps exported: {len(sorted_timestamps)}")
            print(f"📁 File saved as: {filename}")
            print(f"📍 Full path: {os.path.abspath(filename)}")
            print("\n📋 CSV STRUCTURE (ALL DATA IN SAME ROWS):")
            print(f"   - Column A: Timestamp (with date and time)")
            print(f"   - Columns B-P: Temperature data for 15 sensors (15 columns)")
            print(f"   - Column Q: Timestamp (for Humidity - same as column A)")
            print(f"   - Columns R-AF: Humidity data for 15 sensors (15 columns)")
            print(f"   - Column AG: Timestamp (for CO2 - same as column A)")
            print(f"   - Columns AH-AV: CO2 data for 15 sensors (15 columns)")
            print(f"   - Column AW: Timestamp (for PM2.5 - same as column A)")
            print(f"   - Columns AX-BL: PM2.5 data for 15 sensors (15 columns)")
            print(f"\n📊 Total columns: 1 + 15 + 1 + 15 + 1 + 15 + 1 + 15 = 64 columns")
            print("="*80)
            
            return filename
            
        except Exception as e:
            print(f"❌ Error writing CSV file: {e}")
            return None

def main():
    #initialize the device interface
    air1 = Air1Device(API_URL, API_KEY)
    
    #get all devices
    devices = air1.get_all_devices()
    print(f"Found {len(devices)} air-1 devices")
    
    if not devices:
        print("no devices found")
        return
    
    #split working and not working
    working_devices = []
    non_working_devices = []
    
    print("\nTesting each device if functioning")
    for device in devices:
        data = air1.get_device_data(device)
        if data and 'timestamp' in data:
            working_devices.append(device)
            print(f"✅ Device {device} is working")
        else:
            non_working_devices.append(device)
            print(f"❌ Device {device} is inactive")
    
    print(f"\nSummary: {len(working_devices)} active devices, {len(non_working_devices)} inactive devices")
    
    if working_devices:
        # Show current data for all devices
        print("\n" + "____________________________________________")
        print("Latest data from the sensors")
        print("____________________________________________")
        for device in working_devices:
            air1.print_device_summary(device)
        
        # Export ALL historical data to a SINGLE CSV file
        print("\n" + "____________________________________________")
        print("Exporting data in matrix format (all data in same rows)")
        print("____________________________________________")
        
        # Set time range for historical data
        try:
            # You can change these dates/times as needed
            # Example for single day:
            # time_start = datetime(2026, 4, 5, 16, 0, 0)
            # time_end = datetime(2026, 4, 5, 20, 0, 0)
            
            # Example for multiple days:
            time_start = datetime(2026, 9, 16, 0, 0, 0)
            time_end = datetime(2026, 9, 18, 0, 0, 0)
            
            print(f"\nTime range for export (UTC): {time_start} to {time_end}")
            print(f"This will be converted to local time (+8 hours) in the CSV file")
            print("____________________________________________")
            
            # Export all historical data to a single CSV file
            output_dir = r"D:\CoE 199\data_199"
            csv_file = air1.export_all_historical_to_single_csv(working_devices, time_start, time_end, output_dir=output_dir)
            
            if csv_file:
                print("\n💡 To open in Google Sheets:")
                print("   1. Go to https://sheets.google.com")
                print("   2. Click 'Open file picker' (folder icon)")
                print("   3. Go to the 'Upload' tab")
                print("   4. Upload the CSV file")
                print("   5. Double-click to open in Google Sheets")
                print("\n📊 Each row contains ALL parameters (Temp, RH, CO2, PM2.5) for the same timestamp")
                print("   The timestamp appears 4 times in each row (once for each parameter section)")
                print("\n⏰ Timestamps include both date and time for multi-day analysis")
                
                # Generate plots from the CSV file
                print("\n" + "____________________________________________")
                print("Generating plots from exported data")
                print("____________________________________________")
                air1.generate_plots(csv_file, output_dir)
            else:
                print("\n❌ No file was exported")
               
        except Exception as e:
            print(f"Error in historical data export: {e}")
            print("Please adjust the timestamp and year in the datetime objects")
           
    else:
        print("\nNo devices are working")

if __name__ == "__main__":
    main()