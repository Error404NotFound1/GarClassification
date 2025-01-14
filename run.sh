#!/bin/bash

# 定义颜色变量
RED="\033[31m"
YELLOW="\033[33m"
GREEN="\033[32m"
NC="\033[0m"
# 获取最大核心数
MAX_CORES=$(nproc)
# 显示用法信息
usage() {
    echo "Usage: $0 [-g] [-c] [-r] [-h]"
    echo "  -g    github push"
    echo "  -c    Clean the build directory"
    echo "  -r    rebuild the project"
    echo "  -h    Display this help message"
    exit 1
}

# 解析命令行选项
while getopts "g:crh" opt; do
    case $opt in
        g)
            # Push to github
            CommitMessage=$OPTARG
            echo -e "${YELLOW}Commit Message: $CommitMessage${NC}"

             # 执行 Git 操作
            if git pull; then
                if git add .; then
                    if git commit -m "$CommitMessage"; then
                        if git push; then
                            echo -e "${GREEN}Changes pushed successfully!${NC}"
                        else
                            echo -e "${RED}Failed to push changes.${NC}"
                            exit 1
                        fi
                    else
                        echo -e "${RED}Failed to commit changes.${NC}"
                        exit 1
                    fi
                else
                    echo -e "${RED}Failed to stage changes.${NC}"
                    exit 1
                fi
            else
                echo -e "${RED}Failed to pull changes.${NC}"
                exit 1
            fi

            exit 0
            shift
            ;;
        c)
            # Clean the build directory
            echo -e "${YELLOW}Cleaning the build directory...${NC}"
            if [ -d "build" ]; then
                rm -rf build
                echo -e "${GREEN}Build directory removed!${NC}"
            else
                echo -e "${YELLOW}Build directory does not exist.${NC}"
            fi
            exit 0
            shift
            ;;
        r)
            # Rebuild the project
            echo -e "${YELLOW}Rebuilding the project...${NC}"
            if [ -d "build" ]; then
                rm -rf build
                echo -e "${GREEN}Build directory removed!${NC}"
            else
                echo -e "${YELLOW}Build directory does not exist.${NC}"
            fi
            shift
            ;;
        h)
            # Display help message
            usage
            exit 0
            shift
            ;;
        *)
            usage
            exit 0
            shift
            ;;
    esac
done

# 创建 build 目录
if [ ! -d "build" ]; then
  mkdir build
fi

# 进入 build 目录
cd build

echo -e "${YELLOW}Start Cmake ...${NC}"
# 运行 cmake 生成构建文件
cmake ..
if [ $? -eq 0 ]; then
    echo -e "${GREEN}CMake configuration successful!${NC}" 
else
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# 编译项目
make -j${MAX_CORES}

# 检查是否编译成功
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Compilation successful!${NC}"
else
  echo -e "${RED}Compilation failed!${NC}"
  exit 1
fi

  
  # 运行可执行文件
sudo pkill GarbageClassify
sudo ./GarbageClassify
