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
#include <cstdlib>

#include "dataBase.h"
#include "crc/crc_ccitt_modify.h" // 引入新的 CRC 计算头文件

// 外部数据
extern std::vector<YoloRect> GarbageList;
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

int iii=0;

// 发送 SensorData 结构体的函数
bool sendSerialData(int serial_fd, const SendData& data) {
    send_data_message send_message;
    send_message.head.head1 = HEAD1;
    send_message.head.head2 = HEAD2;
    send_message.send_data = data;
    send_message.tail.tail1 = TAIL1;
    send_message.tail.tail2 = TAIL2;

    send_message.send_data.x = 18.58765f + (iii++);
    // send_message.send_data.x = 18.58765f;
    send_message.send_data.y = 17.58765f;
    send_message.send_data.class_id = 3;
    send_message.send_data.angle = 16.98765f;

    // uint8_t buffer2[30]={0xFE,0xEE, 0xAA, 0x90};

    // convertToBigEndian(send_message);

    // 发送整个结构体
    ssize_t bytes_written = write(serial_fd, &send_message, sizeof(send_message));
    // ssize_t bytes_written = write(serial_fd, buffer2, sizeof(buffer2));
    if (bytes_written < 0) {
        std::cerr << "Error writing to serial port: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

// 接收 SensorData 结构体的函数
bool receiveSerialData(int serial_fd, ReceiveData& data) {
    receive_data_message receive_message;
    size_t expected_message_size = sizeof(receive_message);
    uint8_t buffer[expected_message_size];
    size_t bytes_received = 0;

    // uint8_t buffer2[30];

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
        if (bytes_received < sizeof(receive_message.head)) {
            continue;
        }

        // 检查文件头是否匹配
        receive_data_message* tempMessage = reinterpret_cast<receive_data_message*>(buffer);
        if (tempMessage->head.head1 == HEAD1 && tempMessage->head.head2 == HEAD2) {
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
            if(tempMessage->tail.tail1 == TAIL1 && tempMessage->tail.tail2 == TAIL2) {
                std::cout << "Received valid data." << std::endl;
                data = tempMessage->receive_data; // 接收到有效数据

                // for (size_t i = 0; i < 20; ++i) {
                //     std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)buffer[i] << " ";
                // }
                // std::cout << std::endl;

                printReceiveData(*tempMessage);
                system("clear");

                return true; // 成功接收数据
            } else {
                std::cerr << "Received invalid data." << std::endl;
                return false; // 接收到无效数据
            }
        }

        // 如果文件头不匹配，向前移动一个字节，并继续尝试匹配
        memmove(buffer, buffer + 1, bytes_received - 1);
        bytes_received -= 1;

        
        // ssize_t byte = read(serial_fd, buffer2, sizeof(buffer2));
        // std::cout << "buffer: ";
        // for (ssize_t i = 0; i < byte; ++i) {
        //     // 输出为十六进制格式
        //     std::cout << std::hex << static_cast<int>(buffer2[i]) << " ";
        // }
        // std::cout << std::endl;
    }
}

// 原子变量用于控制线程
std::atomic<bool> running(true);
std::vector<YoloRect> GarbageBuffer;
// 发送线程函数
void sendThread(int serial_fd) {
    while (running) {
        // // 对垃圾列表进行排序
        // std::sort(GarbageList.begin(), GarbageList.end(), [](const YoloRect& a, const YoloRect& b) {
        //     if(a.rect.y != b.rect.y) {
        //         return a.rect.y < b.rect.y;
        //     }
        //     return a.rect.x < b.rect.x;
        // });
        // 发送结构体
        if (!sendSerialData(serial_fd, final_send_data)) {
            std::cerr << "Failed to send data." << std::endl;
            break; // 发送失败，退出循环
        }
        // std::cout << "Data sent successfully." << std::endl;

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
            // break; // 接收失败，退出循环
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
