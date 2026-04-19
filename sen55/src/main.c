#include <stdio.h>
#include <string.h>
#include <time.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "mqtt_client.h"
#include "esp_sntp.h"

#include "sen55.h"

static const char *TAG = "SEN55_MQTT";

/* =========================
   WIFI CONFIG
   ========================= */
#define WIFI_SSID      "Smart-iLab"
#define WIFI_PASS      "++adminsmartilab2023!!"

/* =========================
   MQTT CONFIG
   ========================= */
#define MQTT_URI       "mqtt://10.158.71.19:1883"
#define MQTT_USER      "guest"
#define MQTT_PASSWD    "smartilab123"
#define MQTT_TOPIC     "sen55_01/data"

/* ==========================
   SENSOR METADATA
   ========================= */
#define SENSOR_ID      "sen55_01"
#define LOCATION       "Zone 1 - Top Left Window Side"
#define ROOM_NAME      "UP EEEI Smart I-Lab"

static esp_mqtt_client_handle_t client;

/* =========================================================
   WIFI EVENT HANDLER
   ========================================================= */
static void wifi_event_handler(void *arg,
                               esp_event_base_t event_base,
                               int32_t event_id,
                               void *event_data)
{
    if (event_base == WIFI_EVENT &&
        event_id == WIFI_EVENT_STA_START) {

        esp_wifi_connect();
        ESP_LOGI(TAG, "Connecting to WiFi...");

    } else if (event_base == WIFI_EVENT &&
               event_id == WIFI_EVENT_STA_DISCONNECTED) {

        ESP_LOGW(TAG, "WiFi disconnected, retrying...");
        esp_wifi_connect();

    } else if (event_base == IP_EVENT &&
               event_id == IP_EVENT_STA_GOT_IP) {

        ESP_LOGI(TAG, "WiFi connected. Got IP address.");
    }
}

/* =========================================================
   WIFI INIT
   ========================================================= */
static void wifi_init(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(
        esp_event_handler_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            &wifi_event_handler,
            NULL
        )
    );

    ESP_ERROR_CHECK(
        esp_event_handler_register(
            IP_EVENT,
            IP_EVENT_STA_GOT_IP,
            &wifi_event_handler,
            NULL
        )
    );

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(
        esp_wifi_set_config(WIFI_IF_STA, &wifi_config)
    );
    ESP_ERROR_CHECK(esp_wifi_start());
}

/* =========================================================
   SNTP TIME SYNC
   ========================================================= */
static void obtain_time(void)
{
    ESP_LOGI(TAG, "Starting SNTP...");

    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "pool.ntp.org");
    esp_sntp_init();

    time_t now = 0;
    struct tm timeinfo = {0};
    int retry = 0;
    const int retry_count = 15;

    while (timeinfo.tm_year < (2020 - 1900) &&
           ++retry < retry_count) {

        ESP_LOGI(TAG, "Waiting for time sync...");
        vTaskDelay(pdMS_TO_TICKS(2000));

        time(&now);
        localtime_r(&now, &timeinfo);
    }

    setenv("TZ", "PST-8", 1);   // Change if needed
    tzset();

    ESP_LOGI(TAG, "Time synchronized.");
}

/* =========================================================
   MQTT EVENT HANDLER
   ========================================================= */
static void mqtt_event_handler(void *handler_args,
                               esp_event_base_t base,
                               int32_t event_id,
                               void *event_data)
{
    switch ((esp_mqtt_event_id_t)event_id) {

        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT Connected");
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT Disconnected");
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT Error");
            break;

        default:
            break;
    }
}

/* =========================================================
   MQTT START
   ========================================================= */
static void mqtt_start(void)
{
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_URI,
        .credentials.username = MQTT_USER,
        .credentials.authentication.password = MQTT_PASSWD,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);

    esp_mqtt_client_register_event(
        client,
        ESP_EVENT_ANY_ID,
        mqtt_event_handler,
        NULL
    );

    esp_mqtt_client_start(client);
}

/* =========================================================
   CREATE TIMESTAMP STRING
   ========================================================= */
static void get_timestamp(char *buffer, size_t len)
{
    time_t now;
    struct tm timeinfo;

    time(&now);
    localtime_r(&now, &timeinfo);

    strftime(buffer, len, "%Y-%m-%d %H:%M:%S%z", &timeinfo);
}

/* =========================================================
   MAIN
   ========================================================= */
void app_main(void)
{
    sen55_data_t data;
    char timestamp[64];
    char msg[512];

    ESP_LOGI(TAG, "Booting system...");

    wifi_init();

    /* Wait for WiFi */
    vTaskDelay(pdMS_TO_TICKS(5000));

    obtain_time();

    mqtt_start();

    sen55_init();

    if (sen55_start_measurement() == ESP_OK) {
        ESP_LOGI(TAG, "SEN55 measurement started");
    } else {
        ESP_LOGE(TAG, "Failed to start SEN55");
    }

    vTaskDelay(pdMS_TO_TICKS(3000));

    while (1) {

        if (sen55_read_values(&data) == ESP_OK) {

            get_timestamp(timestamp, sizeof(timestamp));

            snprintf(msg, sizeof(msg),
                "{"
                "\"timestamp\":\"%s\","
                "\"sensor_id\":\"%s\","
                "\"location\":\"%s\","
                "\"room\":\"%s\","
                "\"pm1_0\":%.1f,"
                "\"pm2_5\":%.1f,"
                "\"pm4_0\":%.1f,"
                "\"pm10_0\":%.1f,"
                "\"temperature\":%.2f,"
                "\"humidity\":%.2f"
                "}",
                timestamp,
                SENSOR_ID,
                LOCATION,
                ROOM_NAME,
                data.pm1_0,
                data.pm2_5,
                data.pm4_0,
                data.pm10,
                data.temperature,
                data.humidity
            );

            int msg_id = esp_mqtt_client_publish(
                client,
                MQTT_TOPIC,
                msg,
                0,
                1,
                0
            );

            if (msg_id >= 0) {
                ESP_LOGI(TAG, "Published msg_id=%d", msg_id);
                ESP_LOGI(TAG, "%s", msg);
            } else {
                ESP_LOGW(TAG, "Publish failed");
            }

        } else {
            ESP_LOGW(TAG, "SEN55 read failed");
        }

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}