#include "Camera.h"

Camera::Camera() {
    lenth = 0;
    width = 0;
    fps = 0;
    frame = cv::Mat::zeros(0, 0, CV_8UC3);
}

Camera::~Camera() {
    lenth = 0;
    width = 0;
    fps = 0;
    frame.release();
}
