import json
import psycopg2
from paho.mqtt import client as mqtt_client

# MQTT SETTINGS
BROKER = "10.158.71.19"
PORT = 1883
TOPIC = "sen55_01/data"
USERNAME = "guest"
PASSWORD = "smartilab123"
CLIENT_ID = "sen55_pg_subscriber"

# POSTGRES SETTINGS
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "smart_ilab"
DB_USER = "postgres"
DB_PASS = "postgres"


def insert_data(data):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO sen55_data (
            timestamp, sensor_id, location, room,
            pm1_0, pm2_5, pm4_0, pm10_0,
            temperature, humidity, voc, nox
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
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
    ))

    conn.commit()
    cur.close()
    conn.close()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT")
        client.subscribe(TOPIC)
    else:
        print("MQTT failed:", rc)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print("Received:", payload)
        insert_data(payload)
        print("Inserted into PostgreSQL")
    except Exception as e:
        print("Error:", e)


client = mqtt_client.Client(CLIENT_ID)
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)
client.loop_forever()