#include "serial/SerialData.h"

// 32位整数转大端字节序
uint32_t toBigEndian32(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |
           ((value >> 8) & 0x0000FF00) |
           ((value << 8) & 0x00FF0000) |
           ((value << 24) & 0xFF000000);
}
// 浮点数转大端字节序
void floatToBigEndian(float value, uint8_t* buffer) {
    union {
        float f;
        uint8_t bytes[4];
    } converter;

    converter.f = value;

    buffer[0] = converter.bytes[3];
    buffer[1] = converter.bytes[2];
    buffer[2] = converter.bytes[1];
    buffer[3] = converter.bytes[0];
}
// 转换结构体中的 int 和 float 为大端字节序
void convertToBigEndian(send_data_message& message) {
    // 转换浮点数到大端
    uint8_t buffer[4];

    floatToBigEndian(message.send_data.x, buffer);
    std::memcpy(&message.send_data.x, buffer, sizeof(float));

    floatToBigEndian(message.send_data.y, buffer);
    std::memcpy(&message.send_data.y, buffer, sizeof(float));

    // 转换整数到大端
    uint32_t bigEndianClassId = toBigEndian32(message.send_data.class_id);
    std::memcpy(&message.send_data.class_id, &bigEndianClassId, sizeof(int));

    floatToBigEndian(message.send_data.angle, buffer);
    std::memcpy(&message.send_data.angle, buffer, sizeof(float));
}

void printReceiveData(const receive_data_message& message) {
    std::cout << "Header:" << std::endl;
    std::cout << "  head1: 0x" << std::hex << std::setw(2) << std::setfill('0') << +message.head.head1 << std::endl;
    std::cout << "  head2: 0x" << std::hex << std::setw(2) << std::setfill('0') << +message.head.head2 << std::endl;

    std::cout << "Receive Data:" << std::endl;
    std::cout << "  isReady: " << message.receive_data.isReady << std::endl;
    std::cout << "  isWorking: " << message.receive_data.isWorking << std::endl;
    std::cout << "  temple: " << message.receive_data.temple << std::endl;
    std::cout << "  bucket1_full: " << message.receive_data.bucket1_full << std::endl;

    std::cout << "Tail:" << std::endl;
    std::cout << "  tail1: 0x" << std::hex << std::setw(2) << std::setfill('0') << +message.tail.tail1 << std::endl;
    std::cout << "  tail2: 0x" << std::hex << std::setw(2) << std::setfill('0') << +message.tail.tail2 << std::endl;
}
