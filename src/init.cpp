#include "init.h"


void CameraInit(Camera &cam) {
    cam.lenth = 0;
    cam.width = 0;
    cam.frame = cv::Mat::zeros(0, 0, CV_8UC3);
    cam.Intrinsics = (cv::Mat_<double>(3, 3) << 1.0666e3, 0, 961.8031, 
                                                0, 1.0625e3, 592.6813, 
                                                0, 0, 1);
    cam.Distortion = (cv::Mat_<double>(1, 5) << -0.4337, 0.2218, 1e-6, 1e-6, -0.0612);
}
cv::Mat undistortImage(const cv::Mat& inputImage, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    cv::Mat undistortedImage;
    
    // 去畸变
    cv::undistort(inputImage, undistortedImage, cameraMatrix, distCoeffs);

    return undistortedImage;
}

std::string ModelInit(){
    std::string onnx_file = "/home/ybw/GarClassification/model/GarbageV6_640_16.onnx";
    std::string engine_file = "/home/ybw/GarClassification/model/GarbageV6_640_16.engine";
    if(!checkModel(engine_file)) {
        ONNX2Engine(onnx_file, engine_file);
    }
    return engine_file;
}

// 初始化串口函数
int initSerial(const std::string& port, int baud_rate) {
    // 打开串口
    int serial_fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (serial_fd < 0) {
        std::cerr << "Error opening " << port << ": " << strerror(errno) << std::endl;
        return -1;
    }

    // 配置串口
    if (configureSerialPort(serial_fd, baud_rate) != 0) {
        close(serial_fd);
        return -1;
    }

    return serial_fd;
}

