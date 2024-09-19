cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 设置全局编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含头文件目录
include_directories(include)

# 添加子目录
add_subdirectory(src)

# 创建可执行文件并链接库
add_executable(GarbageClassify main.cpp)
target_link_libraries(GarbageClassify src)
