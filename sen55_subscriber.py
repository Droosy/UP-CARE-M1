import json
import time
from paho.mqtt import client as mqtt_client

BROKER_IP   = "10.158.71.19"
BROKER_PORT = 1883
USERNAME    = "guest"
PASSWORD    = "smartilab123"
TOPIC       = "sen55_01/data"
CLIENT_ID   = "sen55_subscriber"

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_DELAY = 60
MAX_RECONNECT_COUNT = 100

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"Failed to connect. Return code {rc}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from broker.")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        print("")
        print("=================================")
        print(f"Topic: {msg.topic}")
        print(json.dumps(data, indent=2))
        print("=================================")

    except Exception as e:
        print(f"Error: {e}")

def connect_mqtt():
    client = mqtt_client.Client(
        mqtt_client.CallbackAPIVersion.VERSION1,
        CLIENT_ID
    )

    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.connect(BROKER_IP, BROKER_PORT, 60)
    return client

def run():
    client = connect_mqtt()
    client.loop_forever()

if __name__ == "__main__":
    run()
