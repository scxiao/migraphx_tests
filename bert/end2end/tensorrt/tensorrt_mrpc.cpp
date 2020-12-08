#include <iostream>
#include <vector>
#include <ctime>
#include <unordered_map>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

using namespace std::chrono;

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


std::ostream& operator << (std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "{";
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0) os << ", ";
        os << dims.d[i];
    }
    os << "}"; 

	return os;
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

    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        return;
    }

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
            std::cout << "i = " << i << ", dim = " << dims << std::endl;

            nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
            for (int j = 0; j < nb_dims; ++j)
            {
                if (dims.d[j] == -1)
                {
                    dims_min.d[j] = 1;
                    dims_opt.d[j] = 2;
                    dims_max.d[j] = 3;
                }
            }

            profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
            profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
            profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
        }
    }

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}


std::unordered_map<nvinfer1::DataType, std::size_t> map_type_sizes =
{{nvinfer1::DataType::kFLOAT, 4},
 {nvinfer1::DataType::kHALF, 2},
 {nvinfer1::DataType::kINT8, 1},
 {nvinfer1::DataType::kINT32, 4},
 {nvinfer1::DataType::kBOOL, 1}};

std::unordered_map<nvinfer1::DataType, std::string> map_type_names =
{{nvinfer1::DataType::kFLOAT, "float"},
 {nvinfer1::DataType::kHALF, "half"},
 {nvinfer1::DataType::kINT8, "int8"},
 {nvinfer1::DataType::kINT32, "int"},
 {nvinfer1::DataType::kBOOL, "bool"}};

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::size_t malloc_nbbinding_buffer(const nvinfer1::Dims& dims, const nvinfer1::DataType type, void* &buffer)
{
    std::size_t elem_num = getSizeByDim(dims);
    std::size_t type_size = 1;
	if (map_type_sizes.count(type) == 0)
	{
        std::cout << "Type does not exist!" << std::endl;
        std::abort();
    }

    type_size = map_type_sizes.at(type);
    auto size = elem_num * type_size;
    cudaMalloc(&buffer, size);

    return size;
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void parse_sentence(const std::string& sent, std::vector<int64_t>& vec_feature)
{
    size_t pos = 0, pos_next;
    size_t index = 0;
    while ((pos_next = sent.find(',', pos)) != std::string::npos)
    {
        auto word_feature = sent.substr(pos, pos_next);
        vec_feature.push_back(std::stoll(word_feature));
        pos = pos_next + 1;
    }
    vec_feature.push_back(std::stoll(sent.substr(pos)));
    vec_feature.push_back(102);
}

int parse_line(std::string& line, std::size_t sent_size, 
        std::unordered_map<std::string, std::vector<int64_t>>& input_map)
{
    auto& vec_feature = input_map["input.1"];
    auto& vec_id = input_map["input.3"];
    auto& seg_id = input_map["2"];
    vec_feature.clear();
    vec_id.clear();
    seg_id.clear();

    size_t pos = line.find('\t');
    int label = std::stoi(line.substr(0, pos));

    ++pos;
    size_t pos_next = line.find('\t', pos);
    vec_feature.push_back(101);
    parse_sentence(line.substr(pos, pos_next), vec_feature);
    vec_id.resize(vec_feature.size(), 0);

    pos = pos_next + 1;
    parse_sentence(line.substr(pos), vec_feature);
    vec_id.resize(vec_feature.size(), 1);
    seg_id.resize(vec_feature.size(), 1);

    vec_feature.resize(sent_size, 0);
    vec_id.resize(sent_size, 0);
    seg_id.resize(sent_size, 0);

    return label;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " onnx_file input_data" << std::endl;
        return 0;
    }

	// load model
	TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    std::string model_path(argv[1]);
    parseOnnxModel(model_path, engine, context);

    std::unordered_map<std::string, void*> map_name_buffers;
    std::vector<void*> buffers(engine->getNbBindings());
    std::unordered_map<std::string, std::size_t> map_name_size;
	std::unordered_map<std::string, std::size_t> map_name_elem_num;
    std::size_t batch_size = engine->getBindingDimensions(0).d[0];

    // allocate buffer for input and output
    for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto&& name = engine->getBindingName(i);
        auto&& type = engine->getBindingDataType(i);
        auto&& dims = engine->getBindingDimensions(i);
        bool is_input = engine->bindingIsInput(i);
        std::cout << "i = " << i << ", name = " << name;
		std::cout << ", type = " << map_type_names.at(type);
		std::cout << ", dim =  ";
        std::cout << dims;
        std::string inout = is_input ? "input" : "output";
        std::cout << ", is " << inout << std::endl;
        map_name_buffers[name] = (void *)nullptr;
        auto size = malloc_nbbinding_buffer(dims, type, map_name_buffers[name]);
        buffers.push_back(map_name_buffers.at(name));
        map_name_size[name] = size;
		map_name_elem_num[name] = getSizeByDim(dims);
    }

	// load input file
    std::ifstream ifs(argv[2]);
    if (!ifs.is_open())
    {
        std::cout << "Open file " << argv[2] << " error!" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(ifs, line);
    std::size_t accu_count = 0, total_count = 0;
    std::vector<float> vec_output;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while (true)
    {
        std::unordered_map<std::string, std::vector<int64_t>> input_map;
        input_map["input.1"];
        input_map["input.3"];
        input_map["2"];
        std::unordered_map<std::string, std::vector<int64_t>> sent_tokens;
        sent_tokens["input.1"];
        sent_tokens["input.3"];
        sent_tokens["2"];
        std::vector<int> vec_labels;

        for (std::size_t batch_no = 0; batch_no < batch_size; batch_no++)
        {
            std::getline(ifs, line);
            if (line.empty())
            {
                break;
            }
            int label = parse_line(line, 128, sent_tokens);
            vec_labels.push_back(label);
            input_map["input.1"].insert(input_map["input.1"].end(), sent_tokens["input.1"].begin(),
                    sent_tokens["input.1"].end());
            input_map["input.3"].insert(input_map["input.3"].end(), sent_tokens["input.3"].begin(),
                    sent_tokens["input.3"].end());
            input_map["2"].insert(input_map["2"].end(), sent_tokens["2"].begin(),
                    sent_tokens["2"].end());
        }
       
        if (line.empty())
        {
            break;
        }

		// assign input data
		for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
		{
			auto&& name = engine->getBindingName(i);
			auto&& dims = engine->getBindingDimensions(i);
			bool is_input = engine->bindingIsInput(i); 
			if (is_input)
            {
                cudaMemcpy(map_name_buffers[name], sent_tokens[name].data(), map_name_size[name],
                        cudaMemcpyHostToDevice);
            }
		}

        context->executeV2(buffers.data());

		for (std::size_t i = 0; i < engine->getNbBindings(); ++i)
		{
			auto&& name = engine->getBindingName(i);
			auto&& dims = engine->getBindingDimensions(i);
			bool is_input = engine->bindingIsInput(i); 
			if (!is_input)
            {
				vec_output.resize(map_name_elem_num[name]);
                cudaMemcpy(vec_output.data(), map_name_buffers[name], map_name_size[name],
                        cudaMemcpyDeviceToHost);
            }
		}

        for (std::size_t batch_no = 0; batch_no < batch_size; ++batch_no)
        {
            std::cout << "[" << vec_output[2 * batch_no] << ", " << vec_output[2 * batch_no + 1] << "]" << std::endl;
            int calc_label = (vec_output[2 * batch_no] >= vec_output[2 * batch_no + 1]) ? 0 : 1;
            accu_count += (calc_label == vec_labels[batch_no]) ? 1 : 0;
            ++total_count;
        }
    }

    std::cout << "accuracy rate = " << 1.0 * accu_count / total_count << std::endl;

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    milliseconds ms = duration_cast<milliseconds>(t2 - t1);
    std::cout << "It takes " << ms.count() << " ms" << std::endl;
    return 0;
}

