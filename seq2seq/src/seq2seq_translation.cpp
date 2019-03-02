#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include "load_onnx.hpp"
#include "language.hpp"
#include "s2s_utilities.hpp"

std::vector<std::pair<std::string, std::string>> all_sentences;

std::pair<std::vector<int>, std::vector<int>> indices_of_sentence(
        const CLanguage& input_lang, 
        const CLanguage& output_lang,
        std::pair<std::string, std::string> &sent_pair)
{
    auto input_indices = input_lang.get_sentence_indices(sent_pair.first);
    auto output_indices = output_lang.get_sentence_indices(sent_pair.second);

    return std::make_pair(input_indices, output_indices);
}

std::vector<std::string> evaluate_cpu(migarphx::program& encoder, migraphx::program& decoder, 
        const CLanguage& input_lang, const CLanguage& output_lang, 
        const size_t hidden_size, const std::size_t max_sent_len, std::string& sent)
{
    encoder.compile(migraphx::cpu::target{});
    decoder.compile(migraphx::cpu::target{});
    migraphx::program logsoftmax_prog = create_program(1);

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
    std::vector<float> decoder_input{SOS_token};
    std::vector<float> decoder_hidden(encoder_hidden);
    std::vector<std::string> decoder_words{};

    for (st::size_t i = 0; i < max_sent_len; ++i)
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
        decode_hidden.assign(outputs.begin() + output_lang.get_word_num(), outputs.end());

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
        lang1.add_sentence(l1_sent);
        lang2.add_sentence(l2_sent);
    }

    // load the models for the encoder and decoder
    migraphx::program encoder = load_onnx("s2s_encoder.onnx");
    migraphx::program decoder = load_onnx("s2s_decoder.onnx");

    std::size_t hidden_size = 256;
    

    return 0;
}


