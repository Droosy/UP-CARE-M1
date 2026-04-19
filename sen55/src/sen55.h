#pragma once
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    float pm1_0;
    float pm2_5;
    float pm4_0;
    float pm10;
    float humidity;
    float temperature;
} sen55_data_t;

void sen55_init(void);
esp_err_t sen55_start_measurement(void);
esp_err_t sen55_read_values(sen55_data_t *out);