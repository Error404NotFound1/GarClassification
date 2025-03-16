#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>  // for sleep_for
//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
using namespace std;
using namespace std::chrono;

#include "init.h"
#include "Camera.h"
#include "Tools.h"
#include "dataBase.h"
#include "Runningtime.h"
#include "serial/SerialPort.h"
extern Camera cam;
extern ReceiveData final_receive_data;
extern log_text log_text_data;
static bool imshow_flag = true;

void MakePort(){
    std::string port = "/dev/ttyUSB0"; // 根据实际情况修改
    int baud_rate = 115200;

    // 初始化串口
    int serial_fd = initSerial(port, baud_rate);
    if (serial_fd < 0) {
        printf("Serial port initialization failed.\n");
        return;
    }
    serialStart(serial_fd);
}

void showVideo(cv::Mat& frame, cv::VideoCapture& cap){
    // 读取一帧
    cap >> frame;

    // 如果视频播放完毕，重置到第一帧
    if (frame.empty()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        // final_receive_data.isReady = true;
        return;
    }

    // 显示当前帧
    if(imshow_flag){
        cv::imshow("FullScreenWindow", frame);
    }
    // cv::imshow("FullScreenWindow", frame);
    cv::waitKey(1);
}


int main() {
    
    MakePort();

    printf("串口初始化完毕\n");
    // 创建窗口并设置为全屏
    if(imshow_flag){
        cv::namedWindow("FullScreenWindow", cv::WINDOW_NORMAL);
        cv::setWindowProperty("FullScreenWindow", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }

    // 初始化播放宣传片
    cv::VideoCapture cap1("/home/ybw/GarClassification/garbageClassify.mp4");
    cv::Mat frame;
    if (!cap1.isOpened()) {
        std::cerr << "Error: Cannot open video file." << std::endl;
        return -1;
    }

    // 初始化摄像头
    CameraInit(cam);
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    printf("摄像头初始化完毕\n");
   
   // 初始化模型
    unique_ptr<YoloModel> yolo = std::make_unique<YoloModel>(ModelInit());
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置超时限制（例如10秒）
    auto timeout_duration = seconds(10);  // 设置为10秒
    auto start_time = steady_clock::now(); // 记录开始时间

    while (1)
    {
        // // 检查超时
        // auto current_time = steady_clock::now();
        // if (duration_cast<seconds>(current_time - start_time) >= timeout_duration) {
        //     std::cout << "Timeout reached. Exiting..." << std::endl;
        //     break; // 超过超时时间则退出
        // }
        // 播放宣传片
        while(!final_receive_data.isReady){
            showVideo(frame, cap1);
            // cout << "isReady: " << final_receive_data.isReady << endl;
        }
        // printf("播放完毕\n");
        // cout << "isReady: " << final_receive_data.isReady << endl;
        TimePoint t0;
        cv::Mat tempframe;
        cap >> tempframe;
    
        // cam.frame = cv::imread("/home/ybw/GarClassification/test/5890bf472a8cc3f05cb036ef933d3f5.jpg");
        if(!tempframe.empty()) {
            cam.frame = undistortImage(tempframe, cam.Intrinsics, cam.Distortion);
        }
        
        // cam.frame = tempframe;

        cam.lenth = cam.frame.rows;
        cam.width = cam.frame.cols;
        if(cam.frame.empty()) {
            // std::cout << "frame is empty" << std::endl;
            // cap.release();
            continue;
        }
        DetectStart(cam.frame, *yolo, stream);
        
        drawDetections(cam.frame, GarbageList);
        TimePoint t1;
        // cam.fps = round(1000.f / t0.getTimeDiffms(t1));
        
        cv::putText(cam.frame," num: " + std::to_string(log_text_data.num) + " class: " + log_text_data.class_id + log_text_data.text, 
                    cv::Point(10,  30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        post_resize(cam.frame, cam.lenth, cam.width);
        if(imshow_flag){
            cv::imshow("FullScreenWindow", cam.frame);
        }
        // cv::imshow("FullScreenWindow", cam.frame);
        cv::waitKey(30);
    }
    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    
    return 0;
}
