// include/SensorData.h
#ifndef SENSOR_DATA_H
#define SENSOR_DATA_H

#include <stdint.h>

// 确保结构体按1字节对齐
#pragma pack(push, 1)
struct SensorData {
    uint16_t id;
    float temperature;
    float humidity;
};
#pragma pack(pop)

#endif // SENSOR_DATA_H
