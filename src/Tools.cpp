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

// 制作预处理图像
void pre_resize(const cv::Mat& img, cv::Mat &output, int h, int w){
    // 1. 调整图像大小并保持长宽比
    int max_side_len = std::max(img.cols, img.rows);
    cv::Mat pad_img = cv::Mat::zeros(cv::Size(max_side_len, max_side_len), img.type());
    img.copyTo(pad_img(cv::Rect(0, 0, img.cols, img.rows)));

    // 2. 缩放图像到指定大小
    cv::resize(pad_img, output, cv::Size(w, h));
}

// 预处理函数
void preprocess(const cv::Mat& img, std::vector<float>& output)
{
    int h = img.rows;
    int w = img.cols;
    cv::Mat resized_img = img.clone();
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    // 5. 分离通道
    std::vector<cv::Mat> channels(3);
    cv::split(resized_img, channels); // channels[0] = R, channels[1] = G, channels[2] = B

    // 6. 将每个通道的数据按顺序排列为 RRR...GGG...BBB
    output.resize(3 * h * w); // 确保输出向量有足够的空间

    // 复制 R 通道
    std::memcpy(output.data(), channels[2].data, sizeof(float) * h * w);

    // 复制 G 通道
    std::memcpy(output.data() + h * w, channels[1].data, sizeof(float) * h * w);

    // 复制 B 通道
    std::memcpy(output.data() + 2 * h * w, channels[0].data, sizeof(float) * h * w);
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
                switch (j)
                {
                case 0: class_id = 0;
                    break;
                case 8: class_id = 0;
                    break;
                case 1: class_id = 1;
                    break;
                case 2: class_id = 1;
                    break;
                case 3: class_id = 2;
                    break;
                case 4: class_id = 2;
                    break;
                case 5: class_id = 2;
                    break;
                case 6: class_id = 3;
                    break;
                case 7: class_id = 3;
                    break;
                case 9: class_id = 3;
                    break;
                default:
                    break;
                }
                
                class_score = data[5 + j];
            }
        }
        
        confidence *= class_score;
        
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

            int angle = (w > h) ? 0 : 1;
            detections.push_back({cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)), cv::Point(x_c, y_c), confidence, class_id, angle});
        }
    }
    // std::cout<<"detections size: "<<detections.size()<<std::endl;
    // 执行非极大值抑制
    return nonMaximumSuppression(detections, iou_threshold);
}

// 绘制检测框
void drawDetections(cv::Mat& image, const std::vector<YoloRect>& detections) {
    std::cout << "drawDetections size: " << detections.size() << std::endl;
    for (const auto& detection : detections) {
        // 绘制矩形框
        cv::rectangle(image, detection.rect, cv::Scalar(0, 255, 0), 2); // 绿色框

        // 绘制置信度和类别ID
        std::string label = "Class: " + std::to_string(detection.class_id) + 
                            ", Conf: " + std::to_string(detection.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // 计算标签的绘制位置（框的左上角）
        int label_x = detection.rect.x;
        int label_y = detection.rect.y - labelSize.height;

        // 确保标签不会超出顶部边界
        if (label_y < 0) {
            label_y = detection.rect.y + labelSize.height;
        }

        // 确保标签不超出左右边界
        label_x = std::clamp(label_x, 0, image.cols - labelSize.width);

        // 绘制标签背景框
        cv::rectangle(image, 
                      cv::Point(label_x, label_y),
                      cv::Point(label_x + labelSize.width, label_y + labelSize.height + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        
        // 在图像上绘制标签
        cv::putText(image, label, 
                    cv::Point(label_x, label_y + labelSize.height), // 让文本在背景框内
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// 检测函数
void DetectStart(cv::Mat &frame, YoloModel &yolo, cudaStream_t stream) {
    // 创建一个时间点用于测量
    TimePoint t0;

    // 获取输入输出绑定索引
    int input_index = yolo.engine->getBindingIndex("images");
    int output_index1 = yolo.engine->getBindingIndex("output0"); // 第一个输出

    // 获取输入输出维度
    nvinfer1::Dims input_dims = yolo.engine->getBindingDimensions(input_index);
    nvinfer1::Dims output_dims1 = yolo.engine->getBindingDimensions(output_index1);

    // 计算输入输出数据大小
    size_t input_size = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float);
    size_t output_size1 = (output_dims1.d[0] * output_dims1.d[1] * output_dims1.d[2]) * sizeof(float);

    float* input_buffer = nullptr;
    float* output_buffer1 = nullptr;

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

    // 使用 cudaMallocHost 分配页锁定内存
    float* output_data1 = nullptr;
    
    err = cudaMallocHost(reinterpret_cast<void**>(&output_data1), output_size1); // 页锁定内存，供主机使用
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMallocHost for output 1: " << cudaGetErrorString(err) << std::endl;
        // 释放已分配的资源
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        return;
    }

   
    TimePoint t1;
    // 预处理
    std::vector<float> preprocessed_data;
    pre_resize(frame, frame, yolo.INPUT_H, yolo.INPUT_W);
    preprocess(frame, preprocessed_data);
    

    TimePoint t2;
    // 将数据异步复制到GPU
    err = cudaMemcpyAsync(input_buffer, preprocessed_data.data(), input_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during cudaMemcpyAsync to input: " << cudaGetErrorString(err) << std::endl;
        // 释放资源
        cudaFree(input_buffer);
        cudaFree(output_buffer1);
        cudaFreeHost(output_data1);
        return;
    }

    // 执行推理
    void* buffers[2]; // 需要为所有输出分配空间
    // 设置缓冲区
    buffers[input_index] = input_buffer;
    buffers[output_index1] = output_buffer1;

    // 使用流执行推理
    yolo.context->enqueueV2(buffers, stream, nullptr);

    TimePoint t3;

    // 输出缓冲区1
    err = cudaMemcpyAsync(output_data1, output_buffer1, output_size1, cudaMemcpyDeviceToHost, stream);

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


    GarbageList = postProcess(output_data1, yolo.MAX_BOXES, yolo.CONF_THRESH, yolo.IOU_THRESH, yolo.CLASSES);
    std::cout << "GarbageList size: " << GarbageList.size() << std::endl;
    // for(auto &i : GarbageList) {
    //     std::cout << "class_id: " << i.class_id << " confidence: " << i.confidence << std::endl;
    //     std::cout << "rect: " << i.rect << std::endl;
    // }
    
    // // 释放页锁定内存
    cudaFreeHost(output_data1);

}

// 函数：根据检测框、相机内参和畸变系数计算物体的实际位置
cv::Point3f getObjectPosition(const YoloRect& detection, const cv::Mat& intrinsic, const cv::Mat& distCoeffs, const float& FIXED_DISTANCE) {
    // 将单个点存储在一个向量中
    std::vector<cv::Point2f> points = { detection.center };
    std::vector<cv::Point2f> undistortedPoints;

    // 执行畸变校正
    undistortPoints(points, undistortedPoints, intrinsic, distCoeffs, cv::Mat(), intrinsic);

    // 获取校正后的点
    cv::Point2f undistortedPoint = undistortedPoints[0];

    // 从内参矩阵中获取焦距和光心
    float fx = intrinsic.at<double>(0, 0);
    float fy = intrinsic.at<double>(1, 1);
    float cx = intrinsic.at<double>(0, 2);
    float cy = intrinsic.at<double>(1, 2);

    // 计算实际位置
    float z = FIXED_DISTANCE; // 固定前向距离
    float x = (undistortedPoint.x - cx) * z / fx;
    float y = (undistortedPoint.y - cy) * z / fy;

    return cv::Point3f(x, y, z);
}