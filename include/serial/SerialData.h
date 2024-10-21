// include/SensorData.h
#ifndef SERIAL_DATA_H
#define SERIAL_DATA_H

#include <stdint.h>

#define	RX_HEAD1 0xFE 
#define	RX_HEAD2 0xEE 
#define	TX_HEAD1 0xFD 
#define	TX_HEAD2 0xEE 

struct SendData {
    float x = 1;
    float y = 2;
    int class_id = 3;
    float angle = 4;
};

struct ReceiveData {
    bool isFinished = 0;
    bool bucket1_full = 0;
    bool bucket2_full = 0;
    bool bucket3_full = 0;
    bool bucket4_full = 0;
};

typedef struct
{
		uint8_t HAED1;
		uint8_t HAED2;
} rx_header;

typedef struct
{
		uint8_t HAED1;
		uint8_t HAED2;
} tx_header;

typedef struct
{
		uint16_t crc_u;
} CRC16_CHECK_TX;

typedef struct
{		
		rx_header  header;
		SendData send_data;
		CRC16_CHECK_TX CRCdata;		
} rx_device_message;

typedef struct
{		
		tx_header  header;
		ReceiveData receive_data;
		CRC16_CHECK_TX CRCdata;		
} tx_device_message;;


#endif 
