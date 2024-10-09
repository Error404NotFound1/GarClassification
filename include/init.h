#ifndef __INIT_H__
#define __INIT_H__

#include "Camera.h"
#include "Tools.h"
#include "serial/SerialData.h"
#include "serial/SerialPort.h"

void CameraInit(Camera &cam);

std::string ModelInit();

int initSerial(const std::string& port, int baud_rate);

#endif // __INIT_H__