#include "Yolo.h"
#include <iostream>
#include <fstream>
#include "Tools.h"

extern Logger gLogger;

using namespace nvinfer1;

#include <memory>

YoloModel::YoloModel(const std::string& engineFile) {
    runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create runtime." << std::endl;
        return;
    }

    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open engine file: " << engineFile << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 使用智能指针
    std::unique_ptr<char[]> buffer(new char[size]);
    file.read(buffer.get(), size);
    file.close();

    engine = runtime->deserializeCudaEngine(buffer.get(), size, nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        engine->destroy();
        return;
    }
}

YoloModel::~YoloModel() {
    std::cout << "destroying YoloModel" << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
}


