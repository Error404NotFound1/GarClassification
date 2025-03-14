#ifndef __DATABASE_H__
#define __DATABASE_H__

#include <string>
#include "Yolo.h"
#include "serial/SerialData.h"
#include "Camera.h"
#include <vector>

extern std::vector<YoloRect> GarbageList;
extern SendData final_send_data;
extern ReceiveData final_receive_data;
extern Camera cam;


#endif // __DATABASE_H__