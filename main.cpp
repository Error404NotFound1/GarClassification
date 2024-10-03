#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

#include "init.h"
#include "Camera.h"
#include "Tools.h"
#include "dataBase.h"
#include "Runningtime.h"

int main() {
    Camera cam;
    CameraInit(cam);
    cv::VideoCapture cap(0);
    unique_ptr<YoloModel> yolo = std::make_unique<YoloModel>(ModelInit());
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (1)
    {
        TimePoint t0;
        /* code */
        // cv::Mat frame = cv::imread("/home/ybw/GarClassification/2eb99947c7b455a35283973b3d2a4fb.jpg");
        cap >> cam.frame;
        cam.lenth = cam.frame.rows;
        cam.width = cam.frame.cols;
        if(cam.frame.empty()) {
            std::cerr << "frame is empty" << std::endl;
            break;
        }
        DetectStart(cam.frame, *yolo, stream);
        
        drawDetections(cam.frame, GarbageList);
        TimePoint t1;
        cam.fps = round(1000.f / t0.getTimeDiffms(t1));
        cv::putText(cam.frame, "fps: " + std::to_string(cam.fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        cv::resize(cam.frame, cam.frame, cv::Size(cam.width, cam.lenth));
        cv::imshow("frame", cam.frame);
        cv::waitKey(1);
    }
    // cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    
    return 0;
}