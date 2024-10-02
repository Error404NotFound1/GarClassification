#include "init.h"


void CameraInit(Camera &cam) {
    cam.lenth = 0;
    cam.width = 0;
    cam.frame = cv::Mat::zeros(0, 0, CV_8UC3);
    cam.Intrinsics = (cv::Mat_<double>(3, 3) << 800, 0, 320, 
                                                0, 800, 240, 
                                                0, 0, 1);
    cam.Distortion = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
}

std::string ModelInit(){
    std::string onnx_file = "/home/ybw/GarClassification/model/best.onnx";
    std::string engine_file = "/home/ybw/GarClassification/model/bestV2.engine";
    if(!checkModel(engine_file)) {
        ONNX2Engine(onnx_file, engine_file);
    }
    return engine_file;
}

