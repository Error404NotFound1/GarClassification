// include/SerialPort.h
#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <vector>
#include <stdint.h>
#include <unistd.h>     // UNIX 标准函数定义
#include <fcntl.h>      // 文件控制定义
#include <string.h>

#include "SerialData.h"

// 串口配置函数
int configureSerialPort(int fd, int baud_rate);

// 计算校验和
uint8_t calculateChecksum(const std::vector<uint8_t>& data);

// 启动串口
int serialStart(int& serial_fd);

// 发送 SensorData 结构体的函数
bool sendSerialData(int serial_fd, const SendData& data);

// 接收 SensorData 结构体的函数
bool receiveSerialData(int serial_fd, ReceiveData& data);

#endif // SERIAL_PORT_H
