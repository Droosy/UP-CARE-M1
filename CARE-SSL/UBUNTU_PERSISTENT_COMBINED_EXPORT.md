# Running `combined_training_export.py` Persistently on Ubuntu Server

Use `systemd` on Ubuntu to keep the live exporter running in the background.
This makes the process start on boot, restart if it crashes, and write logs to
`journalctl`.

The examples below assume the project is located at:

```bash
/home/ubuntu/CARE-SSL
```

Adjust the path and Linux username if your server uses different values.

## 1. Prepare the Project

SSH into the Ubuntu server and go to the project folder:

```bash
cd /home/ubuntu/CARE-SSL
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages needed by the exporter:

```bash
pip install requests paho-mqtt
```

## 2. Test the Exporter Manually

Run this first before creating the persistent service:

```bash
python combined_training_export.py live \
  --output /home/ubuntu/CARE-SSL/combined_live.csv
```

If SEN55 MQTT may be unavailable but you still want AIR-1 rows to be written:

```bash
python combined_training_export.py live \
  --output /home/ubuntu/CARE-SSL/combined_live.csv \
  --allow-missing-sen55
```

Stop the manual test with `Ctrl+C`.

Important: do not include `--duration-min` if you want the exporter to run
continuously.

## 3. Create the `systemd` Service

Create a service file:

```bash
sudo nano /etc/systemd/system/combined-training-export.service
```

Paste this service definition:

```ini
[Unit]
Description=CARE-SSL Combined Training Export
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/CARE-SSL
ExecStart=/home/ubuntu/CARE-SSL/.venv/bin/python /home/ubuntu/CARE-SSL/combined_training_export.py live --output /home/ubuntu/CARE-SSL/combined_live.csv --allow-missing-sen55
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit.

If your Ubuntu username is not `ubuntu`, change this line:

```ini
User=ubuntu
```

## 4. Enable and Start the Service

Reload `systemd`:

```bash
sudo systemctl daemon-reload
```

Enable the service so it starts on boot:

```bash
sudo systemctl enable combined-training-export
```

Start it now:

```bash
sudo systemctl start combined-training-export
```

## 5. Check Status and Logs

Check whether the service is running:

```bash
sudo systemctl status combined-training-export
```

Follow live logs:

```bash
journalctl -u combined-training-export -f
```

View recent logs:

```bash
journalctl -u combined-training-export -n 100
```

## 6. Stop, Restart, or Disable

Stop the service:

```bash
sudo systemctl stop combined-training-export
```

Restart after changing the script or service file:

```bash
sudo systemctl restart combined-training-export
```

Disable startup on boot:

```bash
sudo systemctl disable combined-training-export
```

## Notes

- The exporter appends rows to `combined_live.csv`.
- The script writes one row every 10 seconds by default.
- Omitting `--duration-min` is what makes the script run indefinitely.
- `Restart=always` makes Ubuntu restart the process if it exits unexpectedly.
- Logs are available through `journalctl`, not printed to an open terminal.
