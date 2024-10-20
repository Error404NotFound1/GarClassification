// include/SensorData.h
#ifndef SERIAL_DATA_H
#define SERIAL_DATA_H

#include <stdint.h>

// 确保结构体按1字节对齐
#pragma pack(push, 1)
struct SendData {
    float x;
    float y;
    int class_id;
    float angle;
};
#pragma pack(pop)

struct ReceiveData {
    bool isFinished = 0;
    bool bucket1_full = 0;
    bool bucket2_full = 0;
    bool bucket3_full = 0;
    bool bucket4_full = 0;
};

#endif 
