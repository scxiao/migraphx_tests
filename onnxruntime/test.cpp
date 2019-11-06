// cxx include
#include <iostream>
#include <map>

// onnx runtime include
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/allocator.h>
#include <core/providers/migraphx/migraphx_provider_factory.h>

std::string get_type_name(ONNXTensorElementDataType type)
{
    static std::map<ONNXTensorElementDataType, std::string> table = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "half"}, 
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "float"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "double"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "int8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "int16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "int32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "int64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "uint8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "uint16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "uint32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "uint64"}
    };

    if (table.count(type) > 0) {
        return table.at(type);
    }

    return "unknown";
}

template<class T>
void print_vec(std::vector<T>& vec)
{
    std::cout << "{";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (it != vec.begin())
            std::cout << ", ";
        std::cout << *it;
    }

    std::cout << "}" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnxfile" << std::endl;
        return 0;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions sess_options;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MiGraphX(sess_options, 0));

    Ort::Session sess{env, argv[1], sess_options};
    //std::size_t input_num = sess.GetInputCount();
    const char * input_names[] = {"input.1", "input.3", "2"};
    const char * output_names[] = {"1627"};

    std::vector<std::array<int64_t, 128>> input_data(3);
    std::vector<int64_t> in_dims = {1, 128};

    std::vector<Ort::Value> inputs;
    std::cout << "Input names: " << std::endl;
    for (size_t i = 0; i < sess.GetInputCount(); ++i)
    {
        std::cout << "input " << i << "\'s name: " << sess.GetInputName(i, ort_alloc);
        Ort::TypeInfo info = sess.GetInputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        auto onnx_type = tensor_info.GetElementType();
        std::cout << ", type: " << get_type_name(onnx_type) << ", shape = ";
        print_vec(dims);

        if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        {
            std::fill(input_data[i].begin(), input_data[i].end(), 1);
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_data[i].data(), 
                        input_data[i].size(), in_dims.data(), in_dims.size()));
        }
    }
    std::cout << std::endl;

    std::vector<Ort::Value> outputs;
    std::array<float, 2> output_data;
    std::cout << "Output names:" << std::endl;
    for (size_t i = 0; i < sess.GetOutputCount(); ++i)
    {
        std::cout << "Out " << i << "'s name: " << sess.GetOutputName(i, ort_alloc);
        Ort::TypeInfo info = sess.GetOutputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_dims = tensor_info.GetShape();
        auto onnx_type = tensor_info.GetElementType();
        std::cout << ", type: " << get_type_name(onnx_type) << ", shape = ";
        print_vec(out_dims);

        if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            outputs.push_back(Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(),
                        out_dims.data(), out_dims.size()));
        }
    }
    std::cout << std::endl;

    sess.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names,
            outputs.data(), outputs.size());

    std::cout << "outputs = " << std::endl;
    for (auto val : output_data)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

