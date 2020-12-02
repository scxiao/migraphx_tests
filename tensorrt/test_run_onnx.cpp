#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>

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

std::unordered_map<nvinfer1::DataType, std::string> map_types = 
{{nvinfer1::DataType::kFLOAT, "float"},
 {nvinfer1::DataType::kHALF, "half"},
 {nvinfer1::DataType::kINT8, "int8"},
 {nvinfer1::DataType::kINT32, "int"},
 {nvinfer1::DataType::kBOOL, "bool"}};

void printTensor(nvinfer1::ITensor* tensor)
{
    const char *name = tensor->getName();
    std::cout << "name = " << name << ", type = " << map_types[tensor->getType()] << ", dim = ";
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


template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

namespace detail
{
template <class T, bool = std::is_integral<T>::value>
struct helper { };

template<class T>
struct helper<T, false>
{
    static std::vector<T> gen_input(const std::string& name, std::size_t elem_num)
    {
        std::vector<T> vec(elem_num);
        uint32_t seed = static_cast<uint32_t>(get_hash(name));
        std::srand(seed);
        for (std::size_t i = 0; i < elem_num; ++i)
        {
            float val = 1.0f * rand() / RAND_MAX;
            vec[i] = static_cast<T>(val);
        }
       
        return std::move(vec);
    }
};

template <class T>
struct helper<T, true>
{
    static std::vector<T> gen_input(const std::string& name, std::size_t elem_num)
    {
        std::vector<T> vec(elem_num);
        for (std::size_t i = 0; i < elem_num; ++i)
        {
            vec[i] = 1;
        }
       
        return std::move(vec);
    }
};
}

template<class T>
std::vector<T> gen_input(const std::string& name, std::size_t elem_num)
{
    auto vec = detail::helper<T>::gen_input(name, elem_num);
    return std::move(vec);
}

void wrapup_inputs(const std::string &name, const nvinfer1::Dims& dims, nvinfer1::DataType type, void* &buffer)
{
    std::size_t size = getSizeByDim(dims);
    std::size_t type_size = 1;
    std::size_t binding_size = 0;
    if (type == nvinfer1::DataType::kINT8)
    {
        binding_size = sizeof(int8_t) * size;
        cudaMalloc(&buffer, binding_size);
        auto vec = gen_input<int8_t>(name, size);
        cudaMemcpy(buffer, vec.data(), binding_size, cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kFLOAT)
    {
        binding_size = sizeof(float) * size;
        cudaMalloc(&buffer, binding_size);
        auto vec = gen_input<float>(name, size);
        cudaMemcpy(buffer, vec.data(), binding_size, cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kBOOL)
    {
        binding_size = sizeof(bool) * size;
        cudaMalloc(&buffer, binding_size);
        auto vec = gen_input<int8_t>(name, size);
        cudaMemcpy(buffer, vec.data(), binding_size, cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kINT32)
    {
        binding_size = sizeof(int32_t) * size;
        cudaMalloc(&buffer, binding_size);
        auto vec = gen_input<int32_t>(name, size);
        cudaMemcpy(buffer, vec.data(), binding_size, cudaMemcpyHostToDevice);
    }
    else
    {
        std::abort();
    }
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
        auto&& name = engine->getBindingName(i);
        auto&& dims = engine->getBindingDimensions(i);
        auto&& type = engine->getBindingDataType(i);
        std::size_t binding_size = 1;

        printDims(i, name, dims, engine->bindingIsInput(i));
        if (engine->bindingIsInput(i))
        {
            wrapup_inputs(name, dims, type, buffers[i]);
            input_dims.emplace_back(dims);
        }
        else
        {
            output_dims.emplace_back(dims);
        }
    }

    if (input_dims.empty() or output_dims.empty())
    {
        std::cout << "Expect at lease one input and one output for networkn" << std::endl;
        return -1;
    }

    // inference
    // context->enqueue(batch_size, buffers.data(), 0, nullptr);

    for (void *buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}


