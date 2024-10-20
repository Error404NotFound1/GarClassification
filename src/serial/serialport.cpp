#include "serial/SerialPort.h"
#include "serial/SerialData.h"
#include <iostream>
#include <fcntl.h>      // 文件控制定义
#include <termios.h>    // POSIX 终端控制定义
#include <unistd.h>     // UNIX 标准函数定义
#include <errno.h>
#include <string.h>

#include "dataBase.h"

extern SendData final_send_data;

// 串口配置函数
int configureSerialPort(int fd, int baud_rate) {
    struct termios tty;

    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error from tcgetattr: " << strerror(errno) << std::endl;
        return -1;
    }

    // 设置波特率
    speed_t speed;
    switch (baud_rate) {
        case 9600: speed = B9600; break;
        case 19200: speed = B19200; break;
        case 38400: speed = B38400; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
        default:
            std::cerr << "Unsupported baud rate" << std::endl;
            return -1;
    }
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    // 配置串口参数
    tty.c_cflag &= ~PARENB;        // 无奇偶校验
    tty.c_cflag &= ~CSTOPB;        // 1 个停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8 个数据位
    tty.c_cflag &= ~CRTSCTS;       // 禁用硬件流控制
    tty.c_cflag |= CREAD | CLOCAL; // 启用接收器，忽略调制解调器状态行

    tty.c_lflag &= ~ICANON;        // 原始模式
    tty.c_lflag &= ~ECHO;          // 禁用回显
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ISIG;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // 禁用软件流控制
    tty.c_iflag &= ~(ICRNL | INLCR);        // 禁用回车转换

    tty.c_oflag &= ~OPOST;        // 原始输出

    // 设置超时和最小接收字符
    tty.c_cc[VMIN]  = 0;          // 最小字符数
    tty.c_cc[VTIME] = 10;         // 读取超时时间（单位: 0.1 秒）

    // 应用配置
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error from tcsetattr: " << strerror(errno) << std::endl;
        return -1;
    }

    return 0;
}

// 计算校验和
uint8_t calculateChecksum(const std::vector<uint8_t>& data) {
    uint8_t checksum = 0;
    for(auto byte : data) {
        checksum += byte;
    }
    return checksum;
}

// 发送 SensorData 结构体的函数
bool sendSerialData(int serial_fd, const SendData& data) {
    // 序列化结构体
    std::vector<uint8_t> data_bytes(reinterpret_cast<const uint8_t*>(&data),
                                    reinterpret_cast<const uint8_t*>(&data) + sizeof(data));

    // 构建数据包
    std::vector<uint8_t> packet;
    packet.push_back(0xAA); // 起始符
    packet.push_back(static_cast<uint8_t>(data_bytes.size())); // 数据长度
    packet.insert(packet.end(), data_bytes.begin(), data_bytes.end()); // 数据部分
    uint8_t checksum = calculateChecksum(data_bytes);
    packet.push_back(checksum); // 校验和

    // 发送数据包
    ssize_t bytes_written = write(serial_fd, packet.data(), packet.size());
    if (bytes_written < 0) {
        std::cerr << "Error writing to serial port: " << strerror(errno) << std::endl;
        return false;
    }

    // // 打印发送的数据包
    // std::cout << "Sent SensorData Packet: ";
    // for(auto byte : packet) {
    //     printf("0x%02X ", byte);
    // }
    // std::cout << std::endl;

    return true;
}


bool receiveSerialData(int serial_fd, ReceiveData& data) {
    const uint8_t START_BYTE = 0xAA;
    uint8_t byte;
    ssize_t bytes_read;

    // 寻找起始符
    while (true) {
        bytes_read = read(serial_fd, &byte, 1);
        if (bytes_read < 0) {
            std::cerr << "Error reading from serial port: " << strerror(errno) << std::endl;
            return false;
        } else if (bytes_read == 0) {
            // 超时
            std::cerr << "Read timeout while waiting for start byte." << std::endl;
            return false;
        }

        if (byte == START_BYTE) {
            break;
        }
    }

    // 读取数据长度
    uint8_t data_length;
    bytes_read = read(serial_fd, &data_length, 1);
    if (bytes_read != 1) {
        std::cerr << "Error reading data length." << std::endl;
        return false;
    }

    // 读取数据部分
    std::vector<uint8_t> data_bytes(data_length);
    bytes_read = read(serial_fd, data_bytes.data(), data_length);
    if (bytes_read != data_length) {
        std::cerr << "Error reading data bytes." << std::endl;
        return false;
    }

    // 读取校验和
    uint8_t received_checksum;
    bytes_read = read(serial_fd, &received_checksum, 1);
    if (bytes_read != 1) {
        std::cerr << "Error reading checksum." << std::endl;
        return false;
    }

    // 计算校验和
    uint8_t calculated_checksum = calculateChecksum(data_bytes);
    if (calculated_checksum != received_checksum) {
        std::cerr << "Checksum mismatch. Received: " << static_cast<int>(received_checksum)
                  << ", Calculated: " << static_cast<int>(calculated_checksum) << std::endl;
        return false;
    }

    // 反序列化数据
    if (data_length != sizeof(ReceiveData)) {
        std::cerr << "Data length mismatch. Expected: " << sizeof(ReceiveData)
                  << ", Received: " << static_cast<int>(data_length) << std::endl;
        return false;
    }

    memcpy(&data, data_bytes.data(), sizeof(ReceiveData));

    // // 打印接收到的数据
    // std::cout << "Received SensorData Packet: ID=" << data.id
    //           << ", Temperature=" << data.temperature
    //           << ", Humidity=" << data.humidity << std::endl;

    return true;
}

//启动串口
int serialStart(int& serial_fd){
    // SensorData data;
    // data.id = 1;
    // data.temperature = 25.5f;
    // data.humidity = 60.0f;

    // 发送结构体
    if (!sendSerialData(serial_fd, final_send_data)) {
        close(serial_fd);
        return 1;
    }

    // 关闭串口
    close(serial_fd);
    return 0;
}