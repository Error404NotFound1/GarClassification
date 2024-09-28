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

int &Camera::getlenth() {
    return lenth;
}

int &Camera::getwidth() {
    return width;
}

int &Camera::getfps() {
    return fps;
}

void Camera::setlenth(int l) {
    this->lenth = l;
}

void Camera::setwidth(int w) {
    this->width = w;
}

cv::Mat &Camera::getframe() {
    return frame;
}