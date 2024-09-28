#ifndef __CAMERA_H__
#define __CAMERA_H__
#include <opencv2/opencv.hpp>

class Camera {  
public:
    Camera();
    ~Camera();
    int &getlenth();
    int &getwidth();
    int &getfps();
    void setlenth(int l);
    void setwidth(int w);
    // void setfps(int f);
    cv::Mat &getframe();
private:
    int lenth = 0;
    int width = 0;
    int fps = 0;
    cv::Mat frame = cv::Mat::zeros(0, 0, CV_8UC3);
};

#endif // __TEST_H__
