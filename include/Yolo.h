#ifndef __YOLO_H__
#define __YOLO_H__

#include <NvInfer.h>
#include <string>
#include <opencv2/opencv.hpp>

class YoloModel {
    public:
        int INPUT_H = 640;  //输入图像尺寸
        int INPUT_W = 640;  //输入图像尺寸
        int INPUT_C = 3;    //输入通道数
        float CONF_THRESH = 0.6; //置信度阈值
        int CLASSES = 10;   //类别数
        int MAX_BOXES = 25200;    //最大检测框数
        float IOU_THRESH = 0.3;   //IOU阈值
        YoloModel(const std::string& engineFile);
        ~YoloModel();
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
};

class YoloRect {
    public:
        cv::Rect rect;  //检测框
        cv::Point center;   //检测框中心
        float confidence = 0; //置信度
        int class_id = -1;   //类别
        int angle = -1;    //角度: 0是横放，1是竖放
};

#endif // __YOLO_H__