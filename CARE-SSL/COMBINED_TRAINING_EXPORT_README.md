# Combined SEN55 + AIR-1 Training Data Export

This document explains how to run `combined_training_export.py` to collect one
CSV file for ML training using:

- AIR-1 sensors from the lab API
- SEN55 air-quality monitor data from MQTT or an existing SEN55 CSV

The script writes one wide CSV where each row is aligned to a fixed time grid.
By default, the grid interval is 10 seconds.

## What This Script Produces

The output CSV contains:

- AIR-1 temperature columns: `temp_s1` to `temp_s15`
- AIR-1 relative humidity columns: `rh_s1` to `rh_s15`
- AIR-1 CO2 columns: `co2_s1` to `co2_s15`
- AIR-1 PM2.5 columns: `pm25_s1` to `pm25_s15`
- SEN55 PM, temperature, humidity, VOC, and NOx columns
- Metadata columns, including source timestamps, device IDs, data age, freshness flags, and raw JSON

The metadata is intentionally kept in the CSV so it can be inspected later. It
can be removed during the model-training step.

## Requirements

Run these commands in PowerShell from the project folder:

```powershell
cd C:\Users\pjtio\OneDrive\Desktop\CARE-SSL
```

Install the Python packages needed by the exporter:

```powershell
pip install requests paho-mqtt
```

`requests` is used for the AIR-1 API.

`paho-mqtt` is used only for live SEN55 MQTT collection.

## Network Requirements

The script expects these lab services by default:

```text
AIR-1 API:        http://10.158.66.30:80
SEN55 MQTT:       10.158.71.19:1883
SEN55 MQTT topic: sen55_01/data
SEN55 username:   guest
SEN55 password:   smartilab123
```

Before live collection, check that both services are reachable:

```powershell
Test-NetConnection 10.158.66.30 -Port 80
Test-NetConnection 10.158.71.19 -Port 1883
```

Both should show:

```text
TcpTestSucceeded : True
```

If the MQTT test fails but ping works, the VM is reachable but EMQX is probably
not listening on port `1883`, the port is blocked by firewall, or the EMQX
listener is not enabled.

## Main Use Case: Live Training Data Collection

Use this command to collect live data from both AIR-1 and SEN55 for 60 minutes:

```powershell
python .\combined_training_export.py live --duration-min 60 --output .\combined_training_live.csv
```

The output file will be:

```text
combined_training_live.csv
```

The script writes one row every 10 seconds.

To collect for a different duration, change `--duration-min`:

```powershell
python .\combined_training_export.py live --duration-min 180 --output .\combined_training_3hr.csv
```

To run until manually stopped with `Ctrl+C`, omit `--duration-min`:

```powershell
python .\combined_training_export.py live --output .\combined_training_live.csv
```

## If SEN55 MQTT Is Down

If AIR-1 is working but SEN55 MQTT is not reachable, the normal live command
will stop with an error. To keep collecting AIR-1 rows anyway, run:

```powershell
python .\combined_training_export.py live --duration-min 60 --output .\combined_training_live.csv --allow-missing-sen55
```

This still creates all SEN55 columns, but they remain blank until SEN55 data is
available.

## Historical Merge Mode

Historical mode fetches AIR-1 historical data from the API and merges it with an
existing SEN55 CSV file, usually `sen55_data.csv`.

Example:

```powershell
python .\combined_training_export.py historical --time-start "2026-03-28 03:00:00" --time-end "2026-03-28 05:00:00" --sen55-csv .\sen55_data.csv --output .\combined_training.csv
```

Important timestamp note:

- `--time-start` and `--time-end` are UTC times.
- Output timestamps are converted to local UTC+8 time.

Historical mode requires that `sen55_data.csv` already contains SEN55 readings.
If `sen55_data.csv` only contains the header, the output CSV will still be
created, but SEN55 values will be blank.

## Useful Options

Show all options:

```powershell
python .\combined_training_export.py --help
```

Show historical-mode options:

```powershell
python .\combined_training_export.py historical --help
```

Show live-mode options:

```powershell
python .\combined_training_export.py live --help
```

Change the output interval from 10 seconds to 60 seconds:

```powershell
python .\combined_training_export.py --interval-seconds 60 live --duration-min 60 --output .\combined_training_1min.csv
```

Use a different MQTT broker:

```powershell
python .\combined_training_export.py live --mqtt-broker 10.158.71.19 --mqtt-port 1883 --duration-min 60 --output .\combined_training_live.csv
```

Use a different AIR-1 API URL:

```powershell
python .\combined_training_export.py --air1-api-url http://10.158.66.30:80 live --duration-min 60 --output .\combined_training_live.csv
```

## Default Freshness Rules

The script uses the latest reading at or before each output timestamp.

Default limits:

```text
AIR-1 values are fresh for:  120 seconds
SEN55 values are fresh for:  30 seconds
```

If the latest reading is older than the freshness limit, the numeric sensor
values are left blank for that row.

Metadata and raw JSON are still kept so the data can be inspected later.

To change freshness limits:

```powershell
python .\combined_training_export.py --air1-stale-seconds 180 --sen55-stale-seconds 60 live --duration-min 60 --output .\combined_training_live.csv
```

## AIR-1 Sensor Order

AIR-1 device IDs are mapped to sensor positions in this order:

```text
s1  = 88e4c8
s2  = 88e590
s3  = 89e8d8
s4  = 889720
s5  = 87f510
s6  = 2da640
s7  = 89ea14
s8  = 889b88
s9  = 889938
s10 = 88e85c
s11 = 89e548
s12 = 88970c
s13 = 2deb24
s14 = 89e5f0
s15 = cc8f24
```

To override the order:

```powershell
python .\combined_training_export.py --sensor-order "88e4c8,88e590,89e8d8" live --duration-min 60 --output .\combined_training_live.csv
```

## Output Columns

For the default 15 AIR-1 sensors, the output CSV has 151 columns.

Main feature columns:

```text
timestamp
temp_s1 ... temp_s15
rh_s1 ... rh_s15
co2_s1 ... co2_s15
pm25_s1 ... pm25_s15
sen55_pm1_0
sen55_pm2_5
sen55_pm4_0
sen55_pm10_0
sen55_temperature
sen55_humidity
sen55_voc
sen55_nox
```

Metadata columns include:

```text
air1_device_id_s1 ... air1_device_id_s15
air1_source_timestamp_s1 ... air1_source_timestamp_s15
air1_age_seconds_s1 ... air1_age_seconds_s15
air1_is_fresh_s1 ... air1_is_fresh_s15
air1_raw_json_s1 ... air1_raw_json_s15
sen55_source_timestamp
sen55_age_seconds
sen55_is_fresh
sen55_sensor_id
sen55_location
sen55_room
sen55_raw_json
```

## Common Errors

### `No module named 'requests'`

Install `requests`:

```powershell
pip install requests
```

### `No module named 'paho'`

Install the MQTT package:

```powershell
pip install paho-mqtt
```

### `ConnectionRefusedError [WinError 10061]`

This usually means the VM is reachable, but EMQX is not accepting MQTT
connections on port `1883`.

Check from Windows:

```powershell
Test-NetConnection 10.158.71.19 -Port 1883
```

On the EMQX VM, check that EMQX is running and listening:

```bash
sudo systemctl status emqx
sudo ss -tulpn | grep 1883
```

If EMQX is running in Docker, check that port `1883` is published:

```bash
docker ps
```

You should see something like:

```text
0.0.0.0:1883->1883/tcp
```

Also check firewall rules on the VM and Proxmox.

### AIR-1 API timeout or connection error

Check the AIR-1 API port:

```powershell
Test-NetConnection 10.158.66.30 -Port 80
```

If the test fails, confirm that the PC is on the lab network and that the API
server is available.

## Quick Verification

These commands check the script without doing live collection:

```powershell
python .\combined_training_export.py --help
python .\combined_training_export.py historical --help
python .\combined_training_export.py live --help
python -m py_compile .\combined_training_export.py
```

Successful live collection requires the lab network, AIR-1 API, and EMQX MQTT
broker to be reachable.
