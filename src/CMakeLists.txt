file(GLOB SOURCES "*.cpp")   # 查找 src 目录下的源文件

# 将这些源文件添加到父级项目的构建中
add_library(src ${SOURCES})