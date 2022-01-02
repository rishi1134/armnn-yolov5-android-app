#include <android/asset_manager.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <string>
#include <vector>

#include "armnn/ArmNN.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/IRuntime.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace yoloUtil {
    struct NMSConfig {
        unsigned int num_classes{0};
        unsigned int num_boxes{0};
        float confidence_threshold{0.8f};
        float iou_threshold{0.8f};
    };
    struct Box {
        float x;
        float y;
        float w;
        float h;
    };
    struct Detection {
        Box box;
        float confidence;
        std::vector<float> classes;
    };

    namespace {
        constexpr int box_elements = 4;
        constexpr int confidence_elements = 1;
        float iou(const Box& box1, const Box& box2)
        {
            const float area1 = box1.w * box1.h;
            const float area2 = box2.w * box2.h;
            float overlap;
            if (area1 <= 0 || area2 <= 0)
            {
                overlap = 0.0f;
            }
            else
            {
                float box1_xmin = box1.x - box1.w / 2;
                float box1_xmax = box1.x + box1.w / 2;
                float box2_xmin = box2.x - box2.w / 2;
                float box2_xmax = box2.x + box2.w / 2;
                float box1_ymin = box1.y - box1.h / 2;
                float box1_ymax = box1.y + box1.h / 2;
                float box2_ymin = box2.y - box2.h / 2;
                float box2_ymax = box2.y + box2.h / 2;
                const auto y_min_intersection = std::max<float>(box1_ymin, box2_ymin);
                const auto x_min_intersection = std::max<float>(box1_xmin, box2_xmin);
                const auto y_max_intersection = std::min<float>(box1_ymax, box2_ymax);
                const auto x_max_intersection = std::min<float>(box1_xmax, box2_xmax);
                const auto area_intersection =
                        std::max<float>(y_max_intersection - y_min_intersection, 0.0f) *
                        std::max<float>(x_max_intersection - x_min_intersection, 0.0f);
                overlap = area_intersection / (area1 + area2 - area_intersection);
            }
            return overlap;
        }
        std::vector<Detection> convert_to_detections(const NMSConfig& config,
                                                     const std::vector<float>& detected_boxes)
        {
            const size_t element_step = static_cast<size_t>(
                    box_elements + confidence_elements + config.num_classes);
            std::vector<Detection> detections;

            for (unsigned int i = 0; i < config.num_boxes; ++i)
            {
                const float* cur_box = &detected_boxes[i * element_step];
                if (cur_box[4] > config.confidence_threshold)
                {
                    Detection det;
                    det.box = {cur_box[0], cur_box[1], cur_box[2],
                               cur_box[3]};
                    det.confidence = cur_box[4];
                    det.classes.resize(static_cast<size_t>(config.num_classes), 0);
                    for (unsigned int c = 0; c < config.num_classes; ++c)
                    {
                        const float class_prob = det.confidence * cur_box[5 + c];
                        if (class_prob > config.confidence_threshold)
                        {
                            det.classes[c] = class_prob;
                        }
                    }
                    detections.emplace_back(std::move(det));
                }
            }
            return detections;
        }
    }

    std::vector<Detection> nms(const NMSConfig& config,
                               const std::vector<float>& detected_boxes) {
        std::vector<Detection> detections =
                convert_to_detections(config, detected_boxes);
        std::vector<Detection> pickedDetections;
        const unsigned int num_detections = static_cast<unsigned int>(detections.size());
        std::sort(detections.begin(), detections.begin() + static_cast<std::ptrdiff_t>(num_detections),
                  [](Detection& detection1, Detection& detection2)
                  {
                      return (detection1.confidence - detection2.confidence) > 0;
                  });

        for (unsigned int i = 0; i < num_detections; ++i){
            const Box& box1 = detections[i].box;
            int keep = 1;
            for (unsigned int j = 0; j< (unsigned int)pickedDetections.size(); ++j){
                const Box& box2 = pickedDetections[j].box;
                __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "IoU : %f", iou(box1, box2));
                if (iou(box1, box2) > config.iou_threshold){
                        keep = 0;
                }
            }
            if(keep)
                pickedDetections.push_back(detections[i]);
        }
        return pickedDetections;
    }
}

void LoadImage(cv::Mat& matrix, char* pixelValues, int h, int w, int c)
{
    if (h == 0 || w==0 || c==0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Empty Image Loaded");
    }

    cv::Mat img;
    cv::Mat tmp(h, w, CV_8UC4, pixelValues);
    tmp.copyTo(img);
    cv::resize(img, matrix, cv::Size(640, 640), cv::INTER_LINEAR);
    cv::cvtColor(matrix, matrix, cv::COLOR_RGBA2RGB);
    cv::Vec3b bgrPixel = matrix.at<cv::Vec3b>(10, 10);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%d %d %d", bgrPixel[0],bgrPixel[1],bgrPixel[2]);
}

template<typename TContainer>
inline armnn::OutputTensors MakeOutputTensors(
        const std::vector<armnn::BindingPointInfo>& outputBindings,
        const std::vector<std::reference_wrapper<TContainer>>& outputDataContainers)
{
    armnn::OutputTensors outputTensors;

    const size_t numOutputs = outputBindings.size();

    outputTensors.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; i++)
    {
        const armnn::BindingPointInfo& outputBinding = outputBindings[i];
        const TContainer& outputData = outputDataContainers[i].get();

        armnn::Tensor outputTensor(outputBinding.second, const_cast<float*>(outputData.data()));
        outputTensors.push_back(std::make_pair(outputBinding.first, outputTensor));
    }

    return outputTensors;
}

bool createNetwork(uint8_t* modelBufferAddr, size_t modelOffset,
                                             std::shared_ptr<armnn::IRuntime>& runtime,
                                             armnn::INetworkPtr& network,
                                             armnn::NetworkId& networkId,
                                             armnn::BindingPointInfo& outputBindingInfo,
                                             armnn::BindingPointInfo& inputBindingInfo) {

    std::vector<uint8_t> modelBuffer(modelBufferAddr, modelBufferAddr + modelOffset);

    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    network = parser->CreateNetworkFromBinary(modelBuffer);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%lu", modelBuffer.size());
    modelBuffer.clear();

    runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
    std::vector<armnn::BackendId> backends{"CpuAcc", "GpuAcc", "CpuRef"};

    armnn::IOptimizedNetworkPtr optNet = Optimize(*network,
                                                  backends,
                                                  runtime->GetDeviceSpec(),
                                                  armnn::OptimizerOptions());
    if (optNet == nullptr) {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Optimizer Failed");
    }
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Optimizer Selected");
    std::string errorMessage;

    armnn::Status status = runtime->LoadNetwork(networkId, std::move(optNet), errorMessage);
    if (status != armnn::Status::Success) {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%s", errorMessage.c_str());
    }
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Network Loaded");
    size_t graph_id = 0;
    std::vector<std::string> inputNames = parser->GetSubgraphInputTensorNames(graph_id);
    for (int i = 0; i < inputNames.size(); i++)
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%s", inputNames[i].c_str());
    inputBindingInfo = parser->GetNetworkInputBindingInfo(0, inputNames[0]);
    /* ArmNN dtype enumerations
     Float16  = 0,
     Float32  = 1,
     QAsymmU8 = 2,
     Signed32 = 3,
     Boolean  = 4,
     QSymmS16 = 5,
     QSymmS8  = 6,
     QAsymmS8 = 7,
     BFloat16 = 8,
     Signed64 = 9,
     */
    std::vector<std::string> outputNames = parser->GetSubgraphOutputTensorNames(graph_id);
    for (int i = 0; i < outputNames.size(); i++)
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%s", outputNames[i].c_str());
    outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, outputNames[0]);
    return true;
}

std::vector<yoloUtil::Detection> runInference(const std::shared_ptr<armnn::IRuntime>& runtime,
                                            const cv::Mat& image,
                                            const armnn::INetworkPtr& network,
                                            const armnn::NetworkId& networkId,
                                            const armnn::BindingPointInfo& outputBindingInfo,
                                            const armnn::BindingPointInfo& inputBindingInfo){

    std::vector<float> intermediateMem0(outputBindingInfo.second.GetNumElements());

    using BindingInfos = std::vector<armnn::BindingPointInfo>;
    using FloatTensors = std::vector<std::reference_wrapper<std::vector<float>>>;

    armnn::InputTensors inputTensors = {{ inputBindingInfo.first,armnn::ConstTensor(inputBindingInfo.second, image.data)}};
    armnn::OutputTensors outputTensors = MakeOutputTensors(BindingInfos{ outputBindingInfo },FloatTensors{ intermediateMem0 });

    runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Run Model");

    /*float outputScale = outputBindingInfo.second.GetQuantizationScale();
    int32_t outputZeroPoint = outputBindingInfo.second.GetQuantizationOffset();
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%f %d", outputScale, outputZeroPoint);
    std::transform(intermediateMem0.begin(), intermediateMem0.end(), intermediateMem0.begin(), [outputScale,outputZeroPoint](float &c){ return (c-outputZeroPoint)*outputScale; });*/

    yoloUtil::NMSConfig config;
    config.num_boxes = 25200;
    config.num_classes = 4;
    config.confidence_threshold = 0.4f;
    config.iou_threshold = 0.25f;
    std::vector<yoloUtil::Detection> filtered_boxes;
    filtered_boxes = yoloUtil::nms(config, intermediateMem0);

    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Exit NMS");

    return filtered_boxes;
}


extern "C" {

    static jclass objCls = NULL;
    static jmethodID constructortorId;
    static jfieldID xId;
    static jfieldID yId;
    static jfieldID wId;
    static jfieldID hId;
    static jfieldID labelId;
    static jfieldID probId;

    static std::shared_ptr<armnn::IRuntime> runtime;
    static armnn::NetworkId networkId;
    static armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};
    static armnn::BindingPointInfo outputBindingInfo;
    static armnn::BindingPointInfo inputBindingInfo;

    JNIEXPORT jint JNI_OnLoad(JavaVM * vm , void *reserved ) {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn" , "JNI_OnLoad" ) ;
        return JNI_VERSION_1_4;
    }

    JNIEXPORT void JNI_OnUnload(JavaVM * vm, void * reserved) {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnUnload");
    }

    JNIEXPORT jboolean JNICALL Java_com_brandanalytics_yolov5ncnn_YoloV5Ncnn_Init(JNIEnv *env, jobject thiz, jobject assetManager, jobject modelBuffer) {
        jclass localObjCls = env->FindClass("com/brandanalytics/yolov5ncnn/YoloV5Ncnn$Obj");
        objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

        constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/brandanalytics/yolov5ncnn/YoloV5Ncnn;)V");
        xId = env->GetFieldID(objCls, "x", "F");
        yId = env->GetFieldID(objCls, "y", "F");
        wId = env->GetFieldID(objCls, "w", "F");
        hId = env->GetFieldID(objCls, "h", "F");
        labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
        probId = env->GetFieldID(objCls, "prob", "F");

        auto* modelBufferAddr = static_cast<uint8_t*>(env->GetDirectBufferAddress(modelBuffer));
        auto modelOffset = static_cast<size_t>(env->GetDirectBufferCapacity(modelBuffer));
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Model size(bytes): %lu", modelOffset);
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Model Loaded");
        using clock = std::chrono::steady_clock;
        auto networkStartTime = clock::now();
        if(!createNetwork(modelBufferAddr, modelOffset,
                runtime, network, networkId, outputBindingInfo, inputBindingInfo)){
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Failed to create Network");
            return JNI_FALSE;
        }
        auto networkStopTime = clock::now();
        std::chrono::duration<double> networkDuration = networkStopTime - networkStartTime;
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn/Network Time", "%f", networkDuration.count());
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Network created");

        return JNI_TRUE;
    }

    JNIEXPORT jobjectArray JNICALL Java_com_brandanalytics_yolov5ncnn_YoloV5Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap) {

        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        const int width = info.width;
        const int height = info.height;
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Image size : %d,%d", width, height);
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        char *pixelValues;
        cv::Mat cvMatImage;
        int ret;
        if ((ret = AndroidBitmap_lockPixels(env, bitmap, (void**)&pixelValues)) < 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "AndroidBitmap_lockPixels() failed(src)! error = %d", ret);
        }
        LoadImage(cvMatImage, pixelValues, height, width, 3);
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Image Loaded");
        AndroidBitmap_unlockPixels(env, bitmap);

        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Resized Image :(%d %d)", cvMatImage.cols, cvMatImage.rows);

        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Detect entered");

        using clock = std::chrono::steady_clock;
        auto detectStartTime = clock::now();
        std::vector<yoloUtil::Detection> detections = runInference(runtime, cvMatImage, network, networkId, outputBindingInfo, inputBindingInfo);
        auto detectStopTime = clock::now();
        std::chrono::duration<double> detectDuration = detectStopTime - detectStartTime;
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn/Inference Time", "%f", detectDuration.count());

        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Detect Finished");
        jobjectArray jObjArray = env->NewObjectArray(detections.size(), objCls, NULL);
        static const char* class_names[] = {"none","adidas","nike","asics"};
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%lu", detections.size());

        for (size_t i=0; i<detections.size(); i++)
        {
            jobject jObj = env->NewObject(objCls, constructortorId, thiz);
            env->SetFloatField(jObj, xId, detections[i].box.x);
            env->SetFloatField(jObj, yId, detections[i].box.y);
            env->SetFloatField(jObj, wId, detections[i].box.w);
            env->SetFloatField(jObj, hId, detections[i].box.h);
            jstring label_str = env->NewStringUTF(class_names[std::max_element(detections[i].classes.begin(),detections[i].classes.end()) - detections[i].classes.begin()]);
            //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%f %f", detections[i].box.x,detections[i].box.y);
            //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%s", class_names[std::max_element(detections[i].classes.begin(),detections[i].classes.end()) - detections[i].classes.begin()]);
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%f", detections[i].confidence);
            env->SetObjectField(jObj, labelId, label_str);
            env->SetFloatField(jObj, probId, (float)*std::max_element(detections[i].classes.begin(),detections[i].classes.end()));

            env->SetObjectArrayElement(jObjArray, i, jObj);
            env->DeleteLocalRef(jObj);
            env->DeleteLocalRef(label_str);
        }

        return jObjArray;
    }
}

