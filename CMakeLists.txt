cmake_minimum_required(VERSION 3.10)
project(GarbageClassify)

# 设置 C++ 和 CUDA 的标准版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置全局编译选项
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
add_compile_options(-w)  # 忽略所有警告

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 CUDA 库
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 查找 TensorRT 库
find_library(TENSORRT_LIB nvinfer PATHS /usr/lib/x86_64-linux-gnu/)
find_library(NVONNXPARSER_LIB nvonnxparser PATHS /usr/lib/x86_64-linux-gnu/)

# 包含 TensorRT 头文件目录
include_directories(/usr/include/aarch64-linux-gnu)

# 包含头文件目录
include_directories(include)

# 添加子目录
add_subdirectory(src)

# 创建可执行文件并链接库
add_executable(GarbageClassify main.cpp)

# 链接 OpenCV、CUDA 和 TensorRT 库
target_link_libraries(GarbageClassify src ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIB} ${NVONNXPARSER_LIB})
