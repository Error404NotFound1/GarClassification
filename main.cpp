#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

#include "init.h"
#include "Camera.h"
#include "Tools.h"
#include "dataBase.h"

int main() {
    Camera cam;
    CameraInit(cam);
    // cv::VideoCapture cap(0);
    unique_ptr<YoloModel> yolo = std::make_unique<YoloModel>(ModelInit());
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    while (1)
    {
        /* code */
        cv::Mat frame = cv::imread("/home/ybw/GarClassification/2eb99947c7b455a35283973b3d2a4fb.jpg");
        // cap >> frame;
        if(frame.empty()) {
            std::cerr << "frame is empty" << std::endl;
            break;
        }
        DetectStart(frame, *yolo, stream);
        // if (frame.cols > 0) {
        //     float scale = 640.0f / frame.cols;  // 计算缩放比例
        //     int new_width = static_cast<int>(frame.cols * scale);  // 新宽度
        //     int new_height = static_cast<int>(frame.rows * scale); // 新高度

        //     // 使用 cv::resize 进行缩放
        //     cv::resize(frame, frame, cv::Size(new_width, new_height));
        // }
        // drawDetections(frame, GarbageList);
        // cv::imshow("frame", frame);
        // cv::waitKey(1);
        break;
    }
    // cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    
    return 0;
}