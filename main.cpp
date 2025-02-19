#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

#include "init.h"
#include "Camera.h"
#include "Tools.h"
#include "dataBase.h"
#include "Runningtime.h"
#include "serial/SerialPort.h"

int main() {
    Camera cam;
    CameraInit(cam);

    cv::VideoCapture cap(0);
    
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    unique_ptr<YoloModel> yolo = std::make_unique<YoloModel>(ModelInit());
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string port = "/dev/ttyUSB0"; // 根据实际情况修改
    int baud_rate = 115200;

    // 初始化串口
    int serial_fd = initSerial(port, baud_rate);
    if (serial_fd < 0) {
        return 1;
    }
    serialStart(serial_fd);

     // 创建窗口并设置为全屏
    // cv::namedWindow("FullScreenWindow", cv::WINDOW_NORMAL);
    // cv::setWindowProperty("FullScreenWindow", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    
   
    while (1)
    {
        // TimePoint t0;
        // cv::Mat tempframe;
        // cap >> tempframe;
    
        // cam.frame = cv::imread("/home/ybw/GarClassification/test/5890bf472a8cc3f05cb036ef933d3f5.jpg");
        // cam.frame = undistortImage(tempframe, cam.Intrinsics, cam.Distortion);

        // cam.lenth = cam.frame.rows;
        // cam.width = cam.frame.cols;
        // if(cam.frame.empty()) {
        //     std::cerr << "frame is empty" << std::endl;
        //     break;
        // }
        // DetectStart(cam.frame, *yolo, stream);
        
        // drawDetections(cam.frame, GarbageList);
        // TimePoint t1;
        // cam.fps = round(1000.f / t0.getTimeDiffms(t1));
        // cv::putText(cam.frame, "fps: " + std::to_string(cam.fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        // post_resize(cam.frame, cam.lenth, cam.width);
        // cv::imshow("FullScreenWindow", cam.frame);
        // cv::waitKey(1);
    }
    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    
    return 0;
}