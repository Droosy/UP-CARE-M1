import json
import csv
import os
from paho.mqtt import client as mqtt_client

BROKER = "10.158.71.19"
PORT = 1883
TOPIC = "sen55_01/data"
USERNAME = "guest"
PASSWORD = "smartilab123"

CSV_FILE = "sen55_data.csv"
CLIENT_ID = "sen55_csv_subscriber"

HEADERS = [
    "timestamp", "sensor_id", "location", "room",
    "pm1_0", "pm2_5", "pm4_0", "pm10_0",
    "temperature", "humidity", "voc", "nox"
]


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)


def save_csv(data):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data.get("timestamp"),
            data.get("sensor_id"),
            data.get("location"),
            data.get("room"),
            data.get("pm1_0"),
            data.get("pm2_5"),
            data.get("pm4_0"),
            data.get("pm10_0"),
            data.get("temperature"),
            data.get("humidity"),
            data.get("voc"),
            data.get("nox"),
        ])


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected")
        client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    print(payload)
    save_csv(payload)
    print("Saved to CSV")


init_csv()

client = mqtt_client.Client(CLIENT_ID)
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)
client.loop_forever()