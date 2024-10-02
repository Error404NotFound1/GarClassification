#ifndef __TOOLS_H__
#define __TOOLS_H__
#include <string>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <cassert>
using namespace nvonnxparser;
using namespace nvinfer1;
#include "Yolo.h"
#include <cuda_runtime_api.h>


// 实例化记录器界面。捕获所有警告消息，但忽略信息性消息
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// 使用 extern 声明 gLogger
extern Logger gLogger;

void ONNX2Engine(const std::string &onnx_file, const std::string &engine_file);

bool checkModel(const std::string &engine_file);

void pre_resize(const cv::Mat& img, cv::Mat &output, int h, int w);

void preprocess(const cv::Mat& img, std::vector<float>& output);

void DetectStart(cv::Mat &frame, YoloModel &yolo, cudaStream_t stream);

void drawDetections(cv::Mat& image, const std::vector<YoloRect>& detections);

std::vector<YoloRect> postProcess(float* output_data, int num_detections, float confidence_threshold, float iou_threshold, int class_num);

std::vector<YoloRect> nonMaximumSuppression(const std::vector<YoloRect>& detections, float iouThreshold);

float computeIoU(const cv::Rect& box1, const cv::Rect& box2);

cv::Point3f getObjectPosition(const YoloRect& detection, const cv::Mat& intrinsic, const cv::Mat& distCoeffs, const float& FIXED_DISTANCE);


#endif
