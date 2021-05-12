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
#include <iomanip>
#include <chrono>

using namespace std::chrono;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::ostream& operator << (std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "{";
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0) std::cout << ", ";
        os << dims.d[i];
    }
    os << "}";

    return os;
}

void printDims(int idx, const std::string& name, const nvinfer1::Dims& dims, bool isInput)
{
    std::string str = isInput ? "Input" : "Output";
    std::cout << str << " " << idx << ": name = " << name << ", dim = " << dims << std::endl;
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

std::unordered_map<nvinfer1::TensorLocation, std::string> map_loc = {
    {nvinfer1::TensorLocation::kDEVICE, "device"}, {nvinfer1::TensorLocation::kHOST, "host"}};

template< class T >
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

void printCudaEngine(TRTUniquePtr<nvinfer1::ICudaEngine>& engine)
{
    std::cout << "PrintCudaEngine ===============" << std::endl;
    auto num_layers = engine->getNbLayers();
    std::cout << "num_of_layers = " << num_layers << std::endl;
    auto max_batch_size = engine->getMaxBatchSize();
    std::cout << "max_batch_size = " << max_batch_size << std::endl;
    auto workspace_size = engine->getWorkspaceSize();
    std::cout << "workspace_size = " << workspace_size << std::endl;
    auto dev_mem_size = engine->getDeviceMemorySize();
    std::cout << "Device_mem_size = " << dev_mem_size << std::endl;

    int num_bindings = engine->getNbBindings();
    for (int i = 0; i < num_bindings; ++i)
    {
        std::string inout = engine->bindingIsInput(i) ? "input" : "output";
        std::cout << "binding " << i << ", name: " << engine->getBindingName(i) << " is " << inout << std::endl;
        auto loc = engine->getLocation(i);
        std::cout << "loc = " << map_loc[loc] << std::endl;
    }
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
    nvinfer1::Dims dims = tensor->getDimensions();
    std::cout << "name = " << name;
    std::cout << ", type = " << map_types[tensor->getType()];
    std::cout << ", dim = " << dims << std::endl;
}

void printLayerOutput(nvinfer1::ILayer* layer)
{
    auto num_outputs = layer->getNbOutputs();
    for (int i = 0; i < num_outputs; ++i)
    {
        nvinfer1::ITensor* tensor = layer->getOutput(i);
        std::cout << "\tOutput " << i << ": ";
        printTensor(tensor);
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
    TRTUniquePtr<nvinfer1::IExecutionContext>& context, const std::vector<int>& dim_range, int load_engine)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    milliseconds ms = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Initialize objects takes " << ms.count() << " ms" << std::endl;


    if (load_engine)
    {
        std::string eng_buf;
        std::ifstream eng_fs("bertsquad.engine", std::ios::binary | std::ios::in);
        if(eng_fs.is_open())
        {
            eng_fs.seekg(0, std::ios::end);
            int eng_size = eng_fs.tellg();
            eng_fs.seekg(0, std::ios::beg);
            eng_buf.resize(eng_size);
            eng_fs.read((char *)eng_buf.data(), eng_buf.size());

            t1 = high_resolution_clock::now();            
            engine = TRTUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(eng_buf.data(), eng_buf.size()));
            t2 = high_resolution_clock::now();
            ms = duration_cast<milliseconds>(t2 - t1);
            std::cout << "deserialize_engine takes " << ms.count() << " ms" << std::endl;
        }
    }
    else
    {
        // parse ONNX
        t1 = high_resolution_clock::now();
        if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
        {
            std::cerr << "ERROR: could not parse the model." << std::endl;
            return; 
        }
        t2 = high_resolution_clock::now();
        ms = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Parse model takes " << ms.count() << " ms" << std::endl;


        t1 = high_resolution_clock::now();
        TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        config->setMaxWorkspaceSize(1ULL << 24);
        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        builder->setMaxBatchSize(1);
        if (isDynamicShape(network))
        {
            nvinfer1::IOptimizationProfile* profile = nullptr;
            profile = builder->createOptimizationProfile();
            config->addOptimizationProfile(profile);

            int num_inputs = network->getNbInputs();
            for (int i = 0; i < num_inputs; ++i)
            {
                auto input = network->getInput(i);
                const std::string& input_name = input->getName();
                nvinfer1::Dims dims = input->getDimensions();
                int nb_dims = dims.nbDims;
                std::cout << "i = " << i << " : " << dims << std::endl;

                nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
                for (int j = 0; j < nb_dims; ++j)
                {
                    if (dims.d[j] == -1)
                    {
                        dims_min.d[j] = dim_range[0];
                        dims_opt.d[j] = dim_range[1];
                        dims_max.d[j] = dim_range[2];
                    }
                }

                std::cout << "dims_min = " << dims_min << std::endl;
                std::cout << "dims_max = " << dims_max << std::endl;
                std::cout << "dims_opt = " << dims_opt << std::endl;
                profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
                profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
                profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
            }
        }
        t2 = high_resolution_clock::now();
        ms = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Set dim ranges takes " << ms.count() << " ms" << std::endl;

        t1 = high_resolution_clock::now();
        engine.reset(builder->buildEngineWithConfig(*network, *config));
        t2 = high_resolution_clock::now();
        ms = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Create engine takes " << ms.count() << " ms" << std::endl;

        t1 = high_resolution_clock::now();
        context.reset(engine->createExecutionContext());
        t2 = high_resolution_clock::now();
        ms = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Create context takes " << ms.count() << " ms" << std::endl;

        // save engine
        std::ofstream eng_fs("bertsquad.engine", std::ios::binary | std::ios::out);
        t1 = high_resolution_clock::now();
        nvinfer1::IHostMemory* serialized_model = engine->serialize();
        t2 = high_resolution_clock::now();
        ms = duration_cast<milliseconds>(t2 - t1);
        std::cout << "serialize_engine takes " << ms.count() << " ms, engine_size = " << serialized_model->size() << std::endl;
        eng_fs.write(reinterpret_cast<char *>(serialized_model->data()), serialized_model->size());
        serialized_model->destroy();
    }

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
void print_vec(const std::vector<T>& vec)
{
    std::cout << "elem_num = " << vec.size() << "\n[";
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (i != 0) std::cout << ", ";
        if ((i + 1) % 8 == 0)
            std::cout << std::endl;

        std::cout << std::setw(12) << vec[i];
    }
    std::cout << "]" << std::endl;
}

template<class T>
std::vector<T> gen_input(const std::string& name, std::size_t elem_num)
{
    auto vec = detail::helper<T>::gen_input(name, elem_num);
    std::cout << "Input: ";
    print_vec(vec);
    return std::move(vec);
}

nvinfer1::Dims specify_dynamic_dims(const nvinfer1::Dims& dims, int dim_size)
{
    nvinfer1::Dims ret_dims(dims);

    std::cout << "Dim_before_tune = " << dims << std::endl;
    int nb = dims.nbDims;
    for (int i = 0; i < nb; ++i)
    {
        std::cout << "i = " << i << std::endl;
        if (dims.d[i] == -1)
        {
            ret_dims.d[i] = dim_size;
        }
    }

    std::cout << "Dim_after_tune = " << ret_dims << std::endl;

    return ret_dims;
}

void malloc_nbbinding_buffer(const nvinfer1::Dims& dims, nvinfer1::DataType type, void* &buffer)
{
    std::cout << "malloc, dim = " << dims << std::endl;
    std::size_t elem_num = getSizeByDim(dims);
    std::size_t type_size = 1;
    if (type == nvinfer1::DataType::kINT8)
    {
        type_size = 1;
    }
    else if (type == nvinfer1::DataType::kFLOAT)
    {
        type_size = sizeof(float);
    }
    else if (type == nvinfer1::DataType::kBOOL)
    {
        type_size = sizeof(bool);
    }
    else if (type == nvinfer1::DataType::kINT32)
    {
        type_size = sizeof(int);
    }
    else
    {
        std::abort();
    }

    std::size_t malloc_size = type_size * elem_num;
    cudaMalloc(&buffer, malloc_size);
}

void malloc_all_binding_buffers(TRTUniquePtr<nvinfer1::ICudaEngine>& engine, const std::vector<int>& dim_range, std::vector<void*>& buffers)
{
    buffers.resize(engine->getNbBindings());
    for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto&& dims = engine->getBindingDimensions(i);
        auto&& type = engine->getBindingDataType(i);
        auto specified_dims = specify_dynamic_dims(dims, dim_range[2]);
        malloc_nbbinding_buffer(specified_dims, type, buffers[i]);
    }
}

void generate_input(const std::string& name, nvinfer1::Dims& dims, nvinfer1::DataType type, void* &buffer)
{
    auto elem_num = getSizeByDim(dims);
    if (type == nvinfer1::DataType::kINT8)
    {
        auto vec = gen_input<int8_t>(name, elem_num);
        cudaMemcpy(buffer, vec.data(), elem_num * sizeof(int8_t), cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kFLOAT)
    {
        auto vec = gen_input<float>(name, elem_num);
        cudaMemcpy(buffer, vec.data(), elem_num * sizeof(float), cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kBOOL)
    {
        auto vec = gen_input<int8_t>(name, elem_num);
        cudaMemcpy(buffer, vec.data(), elem_num * sizeof(bool), cudaMemcpyHostToDevice);
    }
    else if (type == nvinfer1::DataType::kINT32)
    {
        auto vec = gen_input<int32_t>(name, elem_num);
        cudaMemcpy(buffer, vec.data(), elem_num * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    else
    {
        std::abort();
    }
}

void generate_all_inputs(TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context, int dim_size, std::vector<void *>& buffers)
{
    for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto&& name = engine->getBindingName(i);
        auto&& dims = engine->getBindingDimensions(i);
        auto&& type = engine->getBindingDataType(i);
        bool is_input = engine->bindingIsInput(i);
        auto tuned_dims = specify_dynamic_dims(dims, dim_size);
        generate_input(name, tuned_dims, type, buffers[i]);
        context->setBindingDimensions(i, tuned_dims);
    }
}

void get_output(const nvinfer1::Dims& dims, nvinfer1::DataType type, void* buffer)
{
    std::size_t size = getSizeByDim(dims);
    std::cout << "Output: ";
    if (type == nvinfer1::DataType::kINT8 or type == nvinfer1::DataType::kBOOL)
    {
        std::vector<int8_t> output(size);
        cudaMemcpy(output.data(), buffer, output.size() * sizeof(int8_t), cudaMemcpyDeviceToHost);
        print_vec(output);
    }
    else if (type == nvinfer1::DataType::kFLOAT)
    {
        std::vector<float> output(size);
        cudaMemcpy(output.data(), buffer, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        print_vec(output);;
    }
    else if (type == nvinfer1::DataType::kINT32)
    {
        std::vector<int32_t> output(size);
        cudaMemcpy(output.data(), buffer, output.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        print_vec(output);
    }
    else
    {
        std::abort();
    }
}

void get_all_outputs(TRTUniquePtr<nvinfer1::ICudaEngine>& engine, int dim_size, std::vector<void*>& buffers)
{
    for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto&& dims = engine->getBindingDimensions(i);
        auto&& type = engine->getBindingDataType(i);
        bool is_input = engine->bindingIsInput(i);
        if (!is_input)
        {
            auto tuned_dims = specify_dynamic_dims(dims, dim_size);
            get_output(tuned_dims, type, buffers[i]);
        }
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
    std::vector<int> dim_range = {1, 3, 10};

    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context, dim_range, false);

    std::cout << "Before execution, engine is:" << std::endl;
    printCudaEngine(engine);

    std::vector<void*> buffers;
    // allocate buffer for input/output on device
    malloc_all_binding_buffers(engine, dim_range, buffers);

    for (int i = dim_range[0]; i < dim_range[2]; ++i)
    {
        generate_all_inputs(engine, context, i, buffers);
        context->executeV2(buffers.data());
        get_all_outputs(engine, i, buffers);
    }

    std::cout << "After execution, engine is:" << std::endl;
    printCudaEngine(engine);

    // free all buffers
    for (void *buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}


