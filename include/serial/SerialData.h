// include/SensorData.h
#ifndef SERIAL_DATA_H
#define SERIAL_DATA_H

#include <stdint.h>
#include <cstring>
#include <iostream>
#include <iomanip>

#define	HEAD1 0xFE 
#define	HEAD2 0xEE 
#define	TAIL1 0xFF
#define	TAIL2 0xFF 

// 32位整数转大端字节序
uint32_t toBigEndian32(uint32_t value);
// 浮点数转大端字节序
void floatToBigEndian(float value, uint8_t* buffer);

#pragma pack(1)
struct SendData {
    float x = 1;
    float y = 2;
    int class_id = 3;
    float angle = 4;
};

struct ReceiveData {
    int isReady = 0;
	int isWorking = 0;
    int bucket1_full = 0;
	float temple = 0;
};

struct log_text{
	int num = 0;
	std::string class_id = " ";
	std::string text = " ";
};

typedef struct
{
		uint8_t head1;
		uint8_t head2;
} header;

typedef struct
{
		uint8_t tail1;
		uint8_t tail2;
} tailer;

typedef struct
{		
		header head;
		SendData send_data;
		tailer tail;		
} send_data_message;


typedef struct
{		
		header head;
		ReceiveData receive_data;
		tailer tail;		
} receive_data_message;

#pragma pack()
// 转换结构体中的 int 和 float 为大端字节序
void convertToBigEndian(send_data_message& message);

void printReceiveData(const receive_data_message& message);

#endif 
