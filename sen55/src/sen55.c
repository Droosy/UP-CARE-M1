#include "sen55.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define I2C_MASTER_SCL_IO 22
#define I2C_MASTER_SDA_IO 21
#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_MASTER_FREQ_HZ 100000

#define SEN55_ADDR 0x69

// ---------- CRC FUNCTION ----------
static uint8_t sen55_crc(uint8_t *data, int len)
{
    uint8_t crc = 0xFF;

    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 0x80)
                crc = (crc << 1) ^ 0x31;
            else
                crc <<= 1;
        }
    }
    return crc;
}

// ---------- I2C INIT ----------
static void i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ
    };

    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

void sen55_init(void)
{
    i2c_master_init();
}

esp_err_t sen55_start_measurement(void)
{
    uint8_t cmd[] = {0x00, 0x21};

    return i2c_master_write_to_device(
        I2C_MASTER_NUM,
        SEN55_ADDR,
        cmd,
        sizeof(cmd),
        pdMS_TO_TICKS(1000)
    );
}

// ---------- READ + DECODE ----------
esp_err_t sen55_read_values(sen55_data_t *out)
{
    uint8_t cmd[] = {0x03, 0xC4};
    uint8_t buffer[24];

    // Send read command
    esp_err_t ret = i2c_master_write_to_device(
        I2C_MASTER_NUM,
        SEN55_ADDR,
        cmd,
        2,
        pdMS_TO_TICKS(1000)
    );
    if (ret != ESP_OK) return ret;

    vTaskDelay(pdMS_TO_TICKS(20));

    // Read response
    ret = i2c_master_read_from_device(
        I2C_MASTER_NUM,
        SEN55_ADDR,
        buffer,
        sizeof(buffer),
        pdMS_TO_TICKS(1000)
    );
    if (ret != ESP_OK) return ret;

    // ---------- PARSE WITH CRC ----------
    uint16_t values[8];
    int idx = 0;

    for (int i = 0; i < 24; i += 3) {
        uint8_t data[2] = {buffer[i], buffer[i+1]};
        uint8_t crc = buffer[i+2];

        if (sen55_crc(data, 2) != crc) {
            return ESP_ERR_INVALID_CRC;
        }

        values[idx++] = (data[0] << 8) | data[1];
    }

    // ---------- CONVERT TO REAL VALUES ----------
    out->pm1_0      = values[0] / 10.0;
    out->pm2_5      = values[1] / 10.0;
    out->pm4_0      = values[2] / 10.0;
    out->pm10       = values[3] / 10.0;
    out->humidity   = values[4] / 100.0;
    out->temperature= values[5] / 200.0;

    return ESP_OK;
}