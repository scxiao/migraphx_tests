#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "load_onnx.hpp"
#include "language.hpp"
#include "s2s_utilities.hpp"
#include <migraphx/cpu/target.hpp>
#include <migraphx/generate.hpp>

std::vector<std::pair<std::string, std::string>> all_sentences;
const std::size_t max_sent_len = 10;

std::pair<std::vector<int>, std::vector<int>> indices_of_sentence(
        CLanguage& input_lang, 
        CLanguage& output_lang,
        const std::string &sent)
{
    auto pos = sent.find('\t', 0);
    assert (pos != std::string::npos);
    auto sent_first = sent.substr(0, pos);
    auto sent_second = sent.substr(pos + 1);
    auto input_indices = input_lang.get_sentence_indices(sent_first);
    auto output_indices = output_lang.get_sentence_indices(sent_second);

    return std::make_pair(input_indices, output_indices);
}

std::pair<std::string, std::string>& get_random_sentence_pair(std::vector<std::pair<std::string, std::string>> &sentences)
{
    int index = rand() % sentences.size();
    return sentences.at(index);
}

std::vector<std::string> evaluate_cpu(migraphx::program& encoder, migraphx::program& decoder, 
        CLanguage& input_lang, CLanguage& output_lang, 
        const size_t hidden_size, const std::size_t max_sent_len, const std::string& sent)
{
    encoder.compile(migraphx::cpu::target{});
    decoder.compile(migraphx::cpu::target{});

    auto indices_pair = indices_of_sentence(input_lang, output_lang, sent);
    auto& input_indices = indices_pair.first;
    auto& output_indices = indices_pair.second;

    std::size_t input_len = input_indices.size();
    std::vector<float> encoder_hidden(hidden_size, 0.0f);
    std::vector<float> encoder_outputs{};

    // run the encoder
    for (std::size_t i = 0; i < input_len; ++i)
    {
        migraphx::program::parameter_map m;
        for (auto&& x : encoder.get_parameter_shapes())
        {
            if (x.first == "input.1")
            {
                m[x.first] = migraphx::argument{x.second, &input_indices.at(i)};
            }
            else if (x.first == "hidden")
            {
                m[x.first] = migraphx::argument{x.second, encoder_hidden.data()};
            }
            else
            {
                m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
            }
        }

        auto concat_hiddens = encoder.eval(m);
        // from the encoder source code, seq_size is 1, so output is the
        // same as the hidden states
        concat_hiddens.visit([&](auto output) { encoder_hidden.assign(output.begin(), output.end()); });
        encoder_outputs.insert(encoder_outputs.end(), encoder_hidden.begin(), encoder_hidden.end());
    }

    // run the decoder
    std::vector<int> decoder_input{SOS_token};
    std::vector<float> decoder_hidden(encoder_hidden);
    std::vector<std::string> decoder_words{};

    for (std::size_t i = 0; i < max_sent_len; ++i)
    {
        migraphx::program::parameter_map m;
        for (auto&& x : decoder.get_parameter_shapes())
        {
            if (x.first == "input.1")
            {
                m[x.first] = migraphx::argument{x.second, decoder_input.data()};
            }
            else if (x.first == "hidden")
            {
                m[x.first] = migraphx::argument{x.second, decoder_hidden.data()};
            }
            else if (x.first == "2")
            {
                m[x.first] = migraphx::argument{x.second, encoder_outputs.data()};
            }
            else
            {
                m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
            }
        }

        auto outputs_arg = decoder.eval(m);
        std::vector<float> outputs;
        outputs_arg.visit([&](auto output) { outputs.assign(output.begin(), output.end()); });

        std::vector<float> decoder_output(outputs.begin(), outputs.begin() + output_lang.get_word_num());
        decoder_hidden.assign(outputs.begin() + output_lang.get_word_num(), outputs.end());

        // compute the words from the decoder output
        std::size_t max_index = std::distance(decoder_output.begin(), std::max_element(decoder_output.begin(),
                    decoder_output.end()));
        if (max_index == static_cast<std::size_t>(EOS_token))
        {
            break;
        }
        else
        {
            decoder_words.push_back(output_lang.get_word(max_index));
        }
    }

    return decoder_words;
}

std::string convert_to_sentence(std::vector<std::string> vec_words)
{
    std::string ret_sent;
    for_each(vec_words.begin(), vec_words.end(), [&](auto word) {
        ret_sent.append(word);
        ret_sent.append(1, ' ');
    });

    return ret_sent;
}


int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " encoder.onnx decoder.onnx input_lang output_lang" << std::endl;
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
    CLanguage input_lang(argv[3]);
    CLanguage output_lang(argv[4]);

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
        input_lang.add_sentence(l1_sent);
        output_lang.add_sentence(l2_sent);
    }

    // load the models for the encoder and decoder
    migraphx::program encoder = load_onnx_file("s2s_encoder.onnx");
    migraphx::program decoder = load_onnx_file("s2s_decoder.onnx");

    int hidden_size = 256;
    int n_words_in_lan = input_lang.get_word_num();
    int n_words_out_lan = output_lang.get_word_num();

    srand(time(nullptr));

    int sent_num = 100;
    for (int sent_no = 0; sent_no < sent_num; ++sent_no)
    {
        auto sent_pair = get_random_sentence_pair(all_sentences);
        auto vec_words = evaluate_cpu(encoder, decoder, input_lang, output_lang, 
                            hidden_size, max_sent_len, sent_pair.first);
        auto output_sentence = convert_to_sentence(vec_words);
        std::cout << "Input    sentence: " << sent_pair.first << std::endl;
        std::cout << "Output   sentence: " << output_sentence << std::endl;
        std::cout << "Expected sentence: " << sent_pair.second << std::endl;
    }

    return 0;
}

