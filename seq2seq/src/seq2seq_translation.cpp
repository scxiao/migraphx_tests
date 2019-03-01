#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include "load_onnx.hpp"
#include "language.hpp"
#include "s2s_utilities.hpp"

std::vector<std::pair<std::string, std::string>> all_sentences;
const int SOS_token = 0;
const int EOS_token = 1;

std::pair<std::vector<int>, std::vector<int>> indices_of_pair(const CLanguage& lang, 
        std::pair<std::string, std::string> &sent_pair)
{
    auto indices1 = lang.get_sentence_indices(sent_pair.first);
    auto indices2 = lang.get_sentence_indices(sent_pair.second);

    return std::make_pair(indices1, indices2);
}



int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " encoder.onnx decoder.onnx lang1 lang2" << std::endl;
        return 0;
    }

    std::string file_name("data//");
    file_name += std::string(argv[1]) + "-" + std::string(argv[2]) + "_procd.txt";
    std::cout << "file_nam = " << file_name << std::endl;

    std::ifstream ifs(file_name);
    if (!ifs.is_open())
    {
        std::cout << "File " << file_name << " open error!" << std::endl;
        return 1;
    }

    // two language objects: one is for lang1, 
    // the other is for lang2
    CLanguage lang1(argv[3]);
    CLanguage lang2(argv[4]);

    std::string line;
    std::size_t line_index = 0;
    while (std::getline(ifs, line, '\n'))
    {
        std::size_t pos = 0;
        pos = line.find('\t', pos);
        std::cout << "line = " << line << std::endl;
        std::string l1_sent{}, l2_sent{};
        if (pos != std::string::npos)
        {
            l1_sent = line.substr(0, pos);
            l2_sent = line.substr(pos + 1);
        }
        else
        {
            std::cout << "File " << file_name << ", Line " << line_index;
            std::cout << " does not contain two languages!" << std::endl;
            return 1;
        }

        all_sentences.push_back(std::make_pair(l1_sent, l2_sent));
        lang1.add_sentence(l1_sent);
        lang2.add_sentence(l2_sent);
    }

    // load the models for the encoder and decoder
    migraphx::program encoder = load_onnx("s2s_encoder.onnx");
    migraphx::program decoder = load_onnx("s2s_decoder.onnx");

    std::size_t hidden_size = 256;

    return 0;
}


