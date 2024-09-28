#include "Tools.h"
#include <vector>
#include "Runningtime.h"
#include "dataBase.h"
#include <algorithm> 

Logger gLogger;
void ONNX2Engine(const std::string &onnx_file, const std::string &engine_file)
{
    std::cout << "Converting onnx model to tensorrt engine model..." << std::endl;
	//这个函数接收一个Logger对象gLogger作为参数，返回一个IBuilder对象，即推理构建器。
	IBuilder* builder = createInferBuilder(gLogger);
	//将数字 1（作为 uint32_t 类型）左移
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	//explicitBatch是一个布尔值参数，指示是否显式地在网络中包含批处理维度
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	//ONNX解析器库来创建一个解析器对象
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	//加载onnx模型
	const char* onnx_filename = onnx_file.c_str();
	//解析模型，并且只记录警告级别及以上的日志
	parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
	//getNbErrors方法返回在解析过程中遇到的错误数量。
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		//打印错误信息
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	//成功加载和解析onnx模型
	std::cout << "successfully load the onnx model" << std::endl;
 
 
	//定义最大批次
	unsigned int maxBatchSize = 1;
	//设置最大批处理大小为
	builder->setMaxBatchSize(maxBatchSize);
	//创建一个新的配置对象
	IBuilderConfig* config = builder->createBuilderConfig();
	//设置最大工作空间
	config->setMaxWorkspaceSize(1 << 20);
	//在构建过程中使用16位浮点数精度
	config->setFlag(BuilderFlag::kFP16);
	//根据给定的网络（network）和配置（config）构建一个TensorRT引擎（engine）
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
 
 
	//尝试序列化一个引擎模型。engine->serialize()方法被用来将TensorRT引擎模型转换为可以存储或传输的格式。
	IHostMemory* gieModelStream = engine->serialize();
	std::ofstream p(engine_file, std::ios::binary);
	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	
	//的返回值转换为一个指向const char*类型的指针，该指针指向要写入的数据的起始位置
	p.write(reinterpret_cast<const char*>(gieModelStream->data()), gieModelStream->size());
	//销毁流，释放内存
	gieModelStream->destroy();
 
 
	std::cout << "successfully generate the trt engine model" << std::endl;
	return;
}

bool checkModel(const std::string &engine_file)
{
    std::ifstream file(engine_file);
    return file.good();  // 如果文件成功打开，则返回 true
}

void preprocess(const cv::Mat& img, cv::Mat &output, int h, int w)
{
    // 图像预处理
    output = img.clone();
    int max_side_len = std::max(output.cols, output.rows);
    cv::Mat pad_img = cv::Mat::zeros(cv::Size(max_side_len, max_side_len), CV_8UC3);
    cv::Rect roi = cv::Rect(0, 0, output.cols, output.rows);
    output.copyTo(pad_img(roi));

    cv::Size input_node_shape(w, h);
    cv::Mat resized_img;

    cv::resize(output, output, cv::Size(w, h));
	// cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    output.convertTo(output, CV_32FC3, 1.0 / 255); // 归一化
}

// 计算两个矩形的重叠区域（IoU）
float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;  // 计算交集
    float intersectionArea = intersection.area();
    float box1Area = box1.area();
    float box2Area = box2.area();

    return intersectionArea / (box1Area + box2Area - intersectionArea);  // 计算 IoU
}

// 执行非极大值抑制
std::vector<YoloRect> nonMaximumSuppression(std::vector<YoloRect>& detections, float iouThreshold) {
    std::vector<YoloRect> result;
    std::sort(detections.begin(),detections.end(),
              //Lambda表达式作为第三个参数输入。
              [](const YoloRect& a,const YoloRect& b){return a.confidence > b.confidence;}
              );
 
    // 创建一个布尔类型的std::vector is_suppressed，其长度等于detections的大小。
    // 所有元素被初始化为false，表示初始时所有检测结果都没有被抑制。
    std::vector<bool> is_suppressed(detections.size(),false);
 
    for(size_t i = 0; i < detections.size(); ++i){
        if(is_suppressed[i]) continue;
 
        //依次遍历整个vector，先选择i，然后依次和（i+1）开始对比iou
        //每次只从第一个循环中写值
        result.push_back(detections[i]);
 
        for(size_t j = i + 1; j < detections.size(); ++j){
 
            if(is_suppressed[j]) continue;
 
            if(detections[i].class_id == detections[j].class_id){
 
               float iou = computeIoU(detections[i].rect, detections[j].rect);
               if(iou > iouThreshold){
                is_suppressed[j] = true;
               }
            }
        }
    }
    return result;
}

// 后处理函数
std::vector<YoloRect> postProcess(float* output_data, int num_detections, float confidence_threshold, float iou_threshold, int class_num) {
    std::vector<YoloRect> detections;

    for (int i = 0; i < num_detections; ++i) {
        const float *data = &output_data[i * (class_num + 5)];

        float x = data[0];
        float y = data[1];
        float width = data[2];
        float height = data[3];

        float confidence = data[4];
        int class_id = -1;
        float class_score = 0;
        for(int j = 0; j < class_num; ++j) {
            if(data[5 + j] > class_score) {
                class_id = j;
                class_score = data[5 + j];
            }
        }
        confidence *= class_score;
        if(confidence>0.005)
        std::cout<<"confidence: "<<confidence<<std::endl;
        // 根据置信度阈值筛选检测框
        if (confidence > confidence_threshold) {
            float x_c = x;
            float y_c = y;
            float w = width;
            float h = height;

            float x0 = x_c - 0.5 * w;
            float y0 = y_c - 0.5 * h;
            float x1 = x_c + 0.5 * w;
            float y1 = y_c + 0.5 * h;
            detections.push_back({cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)), confidence, class_id});
        }
    }
    std::cout<<"detections size: "<<detections.size()<<std::endl;
    // 执行非极大值抑制
    return nonMaximumSuppression(detections, iou_threshold);
}

// 绘制检测框
void drawDetections(cv::Mat& image, const std::vector<YoloRect>& detections) {
    std::cout<<"drawDetections size: "<<detections.size()<<std::endl;
    for (const auto& detection : detections) {
        // 对 detection.rect 进行边界检查
        // int x = std::clamp(detection.rect.x, 0, image.cols - 1);
        // int y = std::clamp(detection.rect.y, 0, image.rows - 1);
        // int width = std::clamp(detection.rect.width, 0, image.cols - x);
        // int height = std::clamp(detection.rect.height, 0, image.rows - y);

        // // 创建新的矩形框，确保不超出边界
        // cv::Rect boundedRect(x, y, width, height);

        // 绘制矩形框
        cv::rectangle(image, detection.rect, cv::Scalar(0, 255, 0), 2); // 绿色框

        // // 绘制置信度
        // std::string label = "Class: " + std::to_string(detection.class_id) + 
        //                     ", Conf: " + std::to_string(detection.confidence);
        // int baseLine;
        // cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // // 计算标签的绘制位置
        // int label_x = x;
        // int label_y = y - labelSize.height;

        // // 确保标签不超出图像边界
        // label_y = std::max(label_y, y + height); // 如果超出顶部，放到框底部
        // label_x = std::clamp(label_x, 0, image.cols - labelSize.width); // 确保x坐标不小于0且不超出右边界

        // // 绘制标签背景框
        // cv::rectangle(image, 
        //             cv::Point(label_x, label_y),
        //             cv::Point(label_x + labelSize.width, label_y + labelSize.height + baseLine),
        //             cv::Scalar(0, 255, 0), cv::FILLED);
        
        // // 在图像上绘制标签
        // cv::putText(image, label, 
        //             cv::Point(label_x, label_y + labelSize.height), // 让文本位置在背景框内
        //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void DetectStart(const cv::Mat &frame, YoloModel &yolo, cudaStream_t stream) {
    // 创建一个时间点用于测量
    TimePoint t0;

    // 获取输入输出绑定索引
    int input_index = yolo.engine->getBindingIndex("input");
    int output_index1 = yolo.engine->getBindingIndex("output"); // 第一个输出
    int output_index2 = yolo.engine->getBindingIndex("onnx::Sigmoid_345"); // 第二个输出
    int output_index3 = yolo.engine->getBindingIndex("onnx::Sigmoid_435"); // 第三个输出
    int output_index4 = yolo.engine->getBindingIndex("onnx::Sigmoid_525"); // 第四个输出

    // 获取输入输出维度
    nvinfer1::Dims input_dims = yolo.engine->getBindingDimensions(input_index);
    nvinfer1::Dims output_dims1 = yolo.engine->getBindingDimensions(output_index1);
    nvinfer1::Dims output_dims2 = yolo.engine->getBindingDimensions(output_index2);
    nvinfer1::Dims output_dims3 = yolo.engine->getBindingDimensions(output_index3);
    nvinfer1::Dims output_dims4 = yolo.engine->getBindingDimensions(output_index4);

    // 计算输入输出数据大小
    size_t input_size = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float);
    size_t output_size1 = (output_dims1.d[0] * output_dims1.d[1] * output_dims1.d[2]) * sizeof(float);
    size_t output_size2 = (output_dims2.d[0] * output_dims2.d[1] * output_dims2.d[2] * output_dims2.d[3] * output_dims2.d[4]) * sizeof(float);
    size_t output_size3 = (output_dims3.d[0] * output_dims3.d[1] * output_dims3.d[2] * output_dims3.d[3] * output_dims3.d[4]) * sizeof(float);
    size_t output_size4 = (output_dims4.d[0] * output_dims4.d[1] * output_dims4.d[2] * output_dims4.d[3] * output_dims4.d[4]) * sizeof(float);

    float* input_buffer = nullptr;
    float* output_buffer1 = nullptr;
    float* output_buffer2 = nullptr;
    float* output_buffer3 = nullptr;
    float* output_buffer4 = nullptr;

    cudaError_t err;

    // 分配GPU内存
    err = cudaMalloc(reinterpret_cast<void**>(&input_buffer), input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&output_buffer1), output_size1);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for output 1: " << cudaGetErrorString(err) << std::endl;
        cudaFree(input_buffer);
        return;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&output_buffer2), output_size2);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for output 2: " << cudaGetErrorString(err) << std::endl;
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        return;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&output_buffer3), output_size3);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for output 3: " << cudaGetErrorString(err) << std::endl;
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        cudaFree(output_buffer2);
        return;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&output_buffer4), output_size4);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for output 4: " << cudaGetErrorString(err) << std::endl;
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        cudaFree(output_buffer2);
        cudaFree(output_buffer3);
        return;
    }

    // 使用 cudaMallocHost 分配页锁定内存
    float* output_data1 = nullptr;
    float* output_data2 = nullptr;
    float* output_data3 = nullptr;
    float* output_data4 = nullptr;
    err = cudaMallocHost(reinterpret_cast<void**>(&output_data1), output_size1); // 页锁定内存，供主机使用
    err = cudaMallocHost(reinterpret_cast<void**>(&output_data2), output_size2); 
    err = cudaMallocHost(reinterpret_cast<void**>(&output_data3), output_size3); 
    err = cudaMallocHost(reinterpret_cast<void**>(&output_data4), output_size4); 

    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMallocHost for output 1: " << cudaGetErrorString(err) << std::endl;
        // 释放已分配的资源
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        cudaFree(output_buffer2);
        cudaFree(output_buffer3);
        cudaFree(output_buffer4);
        return;
    }

   
    TimePoint t1;
    cv::Mat resized_img;
    preprocess(frame, resized_img, yolo.INPUT_H, yolo.INPUT_W);
    

    TimePoint t2;
    // 将数据异步复制到GPU
    err = cudaMemcpyAsync(input_buffer, resized_img.data, input_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMemcpyAsync to input: " << cudaGetErrorString(err) << std::endl;
        // 释放资源
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        cudaFree(output_buffer2);
        cudaFree(output_buffer3);
        cudaFree(output_buffer4);
        cudaFreeHost(output_data1);
        return;
    }

    // 执行推理
    void* buffers[5]; // 需要为所有输出分配空间
    // 设置缓冲区
    buffers[input_index] = input_buffer;
    buffers[output_index1] = output_buffer1;
    buffers[output_index2] = output_buffer2;
    buffers[output_index3] = output_buffer3;
    buffers[output_index4] = output_buffer4;

    // 使用流执行推理
    yolo.context->enqueueV2(buffers, stream, nullptr);

    TimePoint t3;

    // 输出缓冲区1
    err = cudaMemcpyAsync(output_data1, output_buffer1, output_size1, cudaMemcpyDeviceToHost, stream);
    err = cudaMemcpyAsync(output_data2, output_buffer2, output_size2, cudaMemcpyDeviceToHost, stream);
    err = cudaMemcpyAsync(output_data3, output_buffer3, output_size3, cudaMemcpyDeviceToHost, stream);
    err = cudaMemcpyAsync(output_data4, output_buffer4, output_size4, cudaMemcpyDeviceToHost, stream);

    // 等待所有操作完成
    cudaStreamSynchronize(stream);

    TimePoint t4;
    // 打印时间差
    // std::cout << "t0-t1: " << t0.getTimeDiffms(t1) 
    //         << " t1-t2: " << t1.getTimeDiffms(t2) 
    //         << " t2-t3: " << t2.getTimeDiffms(t3) 
    //         << " t3-t4: " << t3.getTimeDiffms(t4) << std::endl;
    // 检查同步后的错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after cudaStreamSynchronize: " << cudaGetErrorString(err) << std::endl;
    }

    GarbageList.clear();
    // 清理
    cudaFree(input_buffer);
    cudaFree(output_buffer1);
    cudaFree(output_buffer2);
    cudaFree(output_buffer3);
    cudaFree(output_buffer4);

    // 释放页锁定内存
    cudaFreeHost(output_data2);
    cudaFreeHost(output_data3);
    cudaFreeHost(output_data4);

    GarbageList = postProcess(output_data1, yolo.MAX_BOXES, yolo.CONF_THRESH, yolo.IOU_THRESH, yolo.CLASSES);
    std::cout << "GarbageList size: " << GarbageList.size() << std::endl;
    // for(auto &i : GarbageList) {
    //     std::cout << "class_id: " << i.class_id << " confidence: " << i.confidence << std::endl;
    //     std::cout << "rect: " << i.rect << std::endl;
    // }
    drawDetections(resized_img, GarbageList);
    cv::imshow("frame", resized_img);
    cv::waitKey(0);
    // 释放页锁定内存
    cudaFreeHost(output_data1);

}