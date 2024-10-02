#include "init.h"


void CameraInit(Camera &cam) {
    cam.setlenth(640);
    cam.setwidth(480);
}

std::string ModelInit(){
    std::string onnx_file = "/home/ybw/GarClassification/model/best.onnx";
    std::string engine_file = "/home/ybw/GarClassification/model/bestV2.engine";
    if(!checkModel(engine_file)) {
        ONNX2Engine(onnx_file, engine_file);
    }
    return engine_file;
}

