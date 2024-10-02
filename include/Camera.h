#ifndef __CAMERA_H__
#define __CAMERA_H__
#include <opencv2/opencv.hpp>

class Camera {  
public:
    Camera();
    ~Camera();
    int lenth = 0;
    int width = 0;
    int fps = 0;
    cv::Mat frame = cv::Mat::zeros(0, 0, CV_8UC3);
    cv::Mat Intrinsics = (cv::Mat_<double>(3, 3) << 0, 0, 0, 
                                                    0, 0, 0, 
                                                    0, 0, 0);
    cv::Mat Distortion = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
};

#endif // __TEST_H__
