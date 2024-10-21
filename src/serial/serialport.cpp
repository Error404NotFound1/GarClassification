#include "serial/SerialPort.h"
#include "serial/SerialData.h"
#include <iostream>
#include <fcntl.h>      // 文件控制定义
#include <termios.h>    // POSIX 终端控制定义
#include <unistd.h>     // UNIX 标准函数定义
#include <errno.h>
#include <string.h>
#include <thread>
#include <atomic>
#include <unistd.h> // for close

#include "dataBase.h"
#include "crc/crc_ccitt_modify.h" // 引入新的 CRC 计算头文件

// 外部数据
extern SendData final_send_data;
extern ReceiveData final_receive_data;

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
    
    std::cout << "Configuring serial port with baud rate: " << baud_rate << std::endl;

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

    std::cout << "Serial port configured successfully." << std::endl;

    return 0;
}

// 发送 SensorData 结构体的函数
bool sendSerialData(int serial_fd, const SendData& data) {
    rx_device_message rxMessage;
    rxMessage.header.HAED1 = RX_HEAD1;
    rxMessage.header.HAED2 = RX_HEAD2;
    rxMessage.send_data = data;

    // 计算 CRC 校验
    uint8_t* dataPtr = reinterpret_cast<uint8_t*>(&rxMessage);
    size_t dataSize = sizeof(rx_device_message) - sizeof(CRC16_CHECK_TX);
    rxMessage.CRCdata.crc_u = crc_ccitt_modify(0xFFFF, dataPtr, dataSize);

    // std::cout << "Calculated CRC: " << std::hex << txMessage.CRCdata.crc_u << std::dec << std::endl;

    // 发送整个结构体
    ssize_t bytes_written = write(serial_fd, &rxMessage, sizeof(rx_device_message));
    if (bytes_written < 0) {
        std::cerr << "Error writing to serial port: " << strerror(errno) << std::endl;
        return false;
    }

    std::cout << "Sent " << bytes_written << " bytes." << std::endl; // 输出发送字节数

    return true;
}

// 接收 SensorData 结构体的函数
bool receiveSerialData(int serial_fd, ReceiveData& data) {
    rx_device_message rxMessage;
    size_t expected_message_size = sizeof(rx_device_message);
    uint8_t buffer[expected_message_size];
    size_t bytes_received = 0;

    while (true) {
        // 逐字节读取数据
        ssize_t byte = read(serial_fd, buffer + bytes_received, 1);
        if (byte < 0) {
            std::cerr << "Error reading from serial port: " << strerror(errno) << std::endl;
            return false; // 读取出错，返回 false
        }

        if (byte == 0) {
            continue; // 没有数据，继续尝试
        }

        // 更新接收的字节数
        bytes_received++;

        // 如果接收到的字节数小于头部大小，继续读取
        if (bytes_received < sizeof(rxMessage.header)) {
            continue;
        }

        // 检查文件头是否匹配
        rx_device_message* tempMessage = reinterpret_cast<rx_device_message*>(buffer);
        if (tempMessage->header.HAED1 == RX_HEAD1 && tempMessage->header.HAED2 == RX_HEAD2) {
            // 找到匹配的文件头，继续读取剩余的数据
            while (bytes_received < expected_message_size) {
                ssize_t more_bytes = read(serial_fd, buffer + bytes_received, expected_message_size - bytes_received);
                if (more_bytes < 0) {
                    std::cerr << "Error reading from serial port: " << strerror(errno) << std::endl;
                    return false; // 读取出错，返回 false
                }
                if (more_bytes == 0) {
                    continue; // 没有数据，继续尝试
                }

                bytes_received += more_bytes;
            }

            // 完整的消息已接收，验证 CRC
            uint8_t* dataPtr = reinterpret_cast<uint8_t*>(buffer);
            size_t dataSize = expected_message_size - sizeof(CRC16_CHECK_TX);
            uint16_t calculated_crc = crc_ccitt_modify(0xFFFF, dataPtr, dataSize);

            // 检查 CRC 是否匹配
            if (calculated_crc == tempMessage->CRCdata.crc_u) {
                // CRC 校验成功，将接收到的数据复制到 ReceiveData
                data = tempMessage->receive_data;
                return true; // 成功接收数据
            } else {
                std::cerr << "CRC mismatch. Calculated: " << calculated_crc
                          << ", Received: " << tempMessage->CRCdata.crc_u << std::endl;
            }
        }

        // 如果文件头不匹配，向前移动一个字节，并继续尝试匹配
        memmove(buffer, buffer + 1, bytes_received - 1);
        bytes_received -= 1;
    }
}

// 原子变量用于控制线程
std::atomic<bool> running(true);

// 发送线程函数
void sendThread(int serial_fd) {
    while (running) {
        // 发送结构体
        if (!sendSerialData(serial_fd, final_send_data)) {
            std::cerr << "Failed to send data." << std::endl;
            break; // 发送失败，退出循环
        }
        std::cout << "Data sent successfully." << std::endl;

        // 可以添加适当的延时
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100毫秒
    }
}

// 接收线程函数
void receiveThread(int serial_fd) {
    while (running) {
        // 接收数据
        if (!receiveSerialData(serial_fd, final_receive_data)) {
            std::cerr << "Failed to receive data." << std::endl;
            break; // 接收失败，退出循环
        }

        // 处理接收到的数据（如果需要）
        // std::cout << "Received valid data." << std::endl;

        // 可以添加适当的延时
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100毫秒
    }
}

// 启动串口
int serialStart(int& serial_fd) {
    std::cout << "Starting serial communication..." << std::endl;

    // 创建发送和接收线程
    std::thread sender(sendThread, serial_fd);
    std::thread receiver(receiveThread, serial_fd);

    // 等待线程完成
    sender.join();
    receiver.join();

    // 关闭串口
    close(serial_fd);
    std::cout << "Serial port closed." << std::endl;

    return 0;
}
