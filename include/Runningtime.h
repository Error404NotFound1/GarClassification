#ifndef __RUNNINGTIME_H__
#define __RUNNINGTIME_H__

#include <chrono>
#include <iostream>

class TimePoint {
public:
    // 构造函数，获取当前时间
    TimePoint() {
        time_point = std::chrono::system_clock::now();
    }

    // 返回两个 TimePoint 对象之间的时间差（以秒为单位）
    double getTimeDiffs(const TimePoint& time) {
        std::chrono::duration<double> diff = time.time_point - this->time_point;
        return abs(diff.count());  // 返回秒为单位的时间差
    }

    // 返回两个 TimePoint 对象之间的时间差（以毫秒为单位）
    double getTimeDiffms(const TimePoint& time) {
        std::chrono::duration<double, std::milli> diff = time.time_point - this->time_point;
        return abs(diff.count());  // 返回毫秒为单位的时间差
    }

private:
    std::chrono::system_clock::time_point time_point;  // 存储时间点
};

#endif // __RUNNINGTIME_H__
