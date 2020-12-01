#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
//#include <opencv2/imgproc.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <fstream>

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void printDims(int idx, const std::string& name, const nvinfer1::Dims& dims, bool isInput)
{
    std::string str = isInput ? "Input" : "Output";
    std::cout << str << " " << idx << ", name = " << name << ": {";
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0) std::cout << ", ";
        std::cout << dims.d[i];
    }
    std::cout << "}" << std::endl;
}

class Logger : public nvinfer1::ILogger
{
    public:
        void log(Severity severity, const char* msg) override {
            if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
            {
                std::cout << msg << "n";
            }
        }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
 
template< class T >
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

void printCudaEngine(TRTUniquePtr<nvinfer1::ICudaEngine>& engine)
{
    std::cout << "PrintCudaEngine===============" << std::endl;
    auto num_layers = engine->getNbLayers();
    std::cout << "num_of_layers = " << num_layers << std::endl;
    auto max_batch_size = engine->getMaxBatchSize();
    std::cout << "max_batch_size = " << max_batch_size << std::endl;
    auto workspace_size = engine->getWorkspaceSize();
    std::cout << "workspace_size = " << workspace_size << std::endl;

    int num_bindings = engine->getNbBindings();
    for (int i = 0; i < num_bindings; ++i)
    {
        if (engine->bindingIsInput(i))
        {
            std::cout << "binding " << i << engine->getBindingName(i) << " is input" << std::endl;
        }
        else
        {
            std::cout << "binding " << i << engine->getBindingName(i) << " is output" << std::endl;
        }
    }
}

void printDims(const nvinfer1::Dims& dims)
{
    std::cout << "{";
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0) std::cout << ", ";
        std::cout << dims.d[i];
    }
    std::cout << "}";
}

void printTensor(nvinfer1::ITensor* tensor)
{
    const char *name = tensor->getName();
    std::cout << "name = " << name << ", dim = ";
    nvinfer1::Dims dims = tensor->getDimensions();
    printDims(dims);
}

void printLayerOutput(nvinfer1::ILayer* layer)
{
    auto num_outputs = layer->getNbOutputs();
    for (int i = 0; i < num_outputs; ++i)
    {
        nvinfer1::ITensor* tensor = layer->getOutput(i);
        std::cout << "\tOutput " << i << ": ";
        printTensor(tensor);
        std::cout << std::endl;
    }
}

void printNetwork(TRTUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    std::cout << "PrintNetwork==================" << std::endl;
    auto num_layers = network->getNbLayers();
    std::cout << "num_of_layers = " << num_layers << std::endl;
    std::cout << "Layer info:" << std::endl;
    for (int i = 0; i < num_layers; ++i)
    {
        nvinfer1::ILayer* layer = network->getLayer(i);
        const char* name = layer->getName();
        std::cout << "Layer " << i << ": " << name << std::endl;
        printLayerOutput(layer);
    }

    std::cout << std::endl;
}

bool isDynamicShape(TRTUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    bool dynamic_shape = false;
    int num_inputs = network->getNbInputs();
    for (int i = 0; i < num_inputs; ++i)
    {
        auto input = network->getInput(i);
        nvinfer1::Dims dims = input->getDimensions();
        int nb_dims = dims.nbDims;
        if (input->isShapeTensor())
        {
            dynamic_shape = true;
        }
        else
        {
            for (int j = 0; j < nb_dims; ++j)
            {
                if (dims.d[j] == -1)
                {
                    dynamic_shape = true;
                }
            }
        }
    }

    return dynamic_shape;
}

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    nvinfer1::IOptimizationProfile* profile = nullptr;

    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        return; 
    }

    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    config->setMaxWorkspaceSize(1ULL << 30);
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(1);
    if (profile == nullptr)
    {
        profile = builder->createOptimizationProfile();
    }

    int num_inputs = network->getNbInputs();
    for (int i = 0; i < num_inputs; ++i)
    {
        auto input = network->getInput(i);
        const std::string& input_name = input->getName();
        nvinfer1::Dims dims = input->getDimensions();
        int nb_dims = dims.nbDims;

        nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
        for (int j = 0; j < nb_dims; ++j)
        {
            dims_min.d[j] = 2;
            dims_opt.d[j] = 3;
            dims_max.d[j] = 4;
        }

        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
    }

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
    printNetwork(network);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " model.onnx" << std::endl;
        return -1;
    }

    std::string model_path(argv[1]);

    int batch_size = 1;

    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context);

    printCudaEngine(engine);

    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers(engine->getNbBindings());

    for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        std::cout << "Loc1, i = " << i << std::endl;
        auto&& name = engine->getBindingName(i);
        auto&& dims = engine->getBindingDimensions(i);

        auto binding_size = getSizeByDim(dims) * batch_size * sizeof(float);
        std::cout << "Loc2" << std::endl;
        cudaMalloc(&buffers[i], binding_size);
        printDims(i, name, dims, engine->bindingIsInput(i));
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(dims);
        }
        else
        {
            output_dims.emplace_back(dims);
        }
        std::cout << "Loc3" << std::endl;
    }

    if (input_dims.empty() or output_dims.empty())
    {
        std::cout << "Expect at lease one input and one output for networkn" << std::endl;
        return -1;
    }

    // inference
    //context->enqueue(batch_size, buffers.data(), 0, nullptr);

    //for (void *buf : buffers)
    //{
    //    cudaFree(buf);
    //}

    return 0;
}


