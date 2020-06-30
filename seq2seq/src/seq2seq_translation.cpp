#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "load_onnx.hpp"
#include "language.hpp"
#include "s2s_utilities.hpp"
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <chrono>

using namespace std::chrono;

std::vector<std::pair<std::string, std::string>> all_sentences;
const std::size_t max_sent_len = 10;

template<typename T>
void print_vec(std::vector<T> &res)
{
    for(std::size_t i = 0; i < res.size(); ++i)
    {
        std::cout << std::setw(12) << res.at(i) << ", ";
        if ((i + 1) % 8 == 0)
        {
            std::cout << std::endl;
        }
    }

    return;
}

std::pair<std::string, std::string>& get_random_sentence_pair(std::vector<std::pair<std::string, std::string>> &sentences)
{
    int index = rand() % sentences.size();
    return sentences.at(index);
}

std::vector<std::string> evaluate(migraphx::program encoder, migraphx::program decoder, 
        migraphx::target& t, CLanguage& input_lang, CLanguage& output_lang, 
        const size_t hidden_size, const std::size_t max_sent_len, std::string& sent)
{

    auto tmp = input_lang.get_sentence_indices(sent);
    std::vector<long> input_indices(tmp.begin(), tmp.end());

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
                m[x.first] = t.copy_to(migraphx::argument(x.second, &input_indices.at(i)));
            }
            else if (x.first == "hidden")
            {
                m[x.first] = t.copy_to(migraphx::argument(x.second, encoder_hidden.data()));
            }
            else
            {
                m[x.first] = t.copy_to(migraphx::generate_argument(x.second, get_hash(x.first)));
            }
        }

        auto prog_outputs = encoder.eval(m);
        // first output is the output to be concatenated
        auto output_arg = t.copy_from(prog_outputs[0]);
        // second output is hidden state used as input for next char
        auto hiddens_arg = t.copy_from(prog_outputs[1]);

        // from the encoder source code, seq_size is 1, so output is the
        // same as the hidden states
        hiddens_arg.visit([&](auto hs) { encoder_hidden.assign(hs.begin(), hs.end()); });

        std::vector<float> cur_output;
        output_arg.visit([&](auto out) { cur_output.assign(out.begin(), out.end()); });

        encoder_outputs.insert(encoder_outputs.end(), cur_output.begin(), cur_output.end());
    }

    // expected size of the encoder output is hidden_size * max_sent_len
    if (hidden_size * max_sent_len > encoder_outputs.size())
    {
        std::size_t elem_num = hidden_size * max_sent_len - encoder_outputs.size();
        encoder_outputs.insert(encoder_outputs.end(), elem_num, 0.0f);
    }

    // run the decoder
    std::vector<long> decoder_input{SOS_token};
    std::vector<float> decoder_hidden(encoder_hidden);
    std::vector<std::string> decoder_words{};

    for (std::size_t i = 0; i < max_sent_len; ++i)
    {
        migraphx::program::parameter_map m;
        for (auto&& x : decoder.get_parameter_shapes())
        {
            if (x.first == "input.1")
            {
                m[x.first] = t.copy_to(migraphx::argument(x.second, decoder_input.data()));
            }
            else if (x.first == "hidden")
            {
                m[x.first] = t.copy_to(migraphx::argument(x.second, decoder_hidden.data()));
            }
            else if (x.first == "2")
            {
                m[x.first] = t.copy_to(migraphx::argument(x.second, encoder_outputs.data()));
            }
            else
            {
                m[x.first] = t.copy_to(migraphx::generate_argument(x.second, get_hash(x.first)));
            }
        }

        auto prog_outputs = decoder.eval(m);

        auto output_arg = t.copy_from(prog_outputs[0]);
        std::vector<float> decoder_output;
        output_arg.visit([&](auto opt) { decoder_output.assign(opt.begin(), opt.end()); });

        auto hidden_arg = t.copy_from(prog_outputs[1]);
        hidden_arg.visit([&](auto opt) { decoder_hidden.assign(opt.begin(), opt.end()); });

        // compute the words from the decoder output
        std::size_t max_index = std::distance(decoder_output.begin(), std::max_element(decoder_output.begin(),
                    decoder_output.end()));
        if (max_index == static_cast<std::size_t>(EOS_token))
        {
            //decoder_words.push_back("<EOS>");
            break;
        }
        else
        {
            decoder_words.push_back(output_lang.get_word(max_index));
            decoder_input.at(0) = static_cast<long>(max_index);
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
    ret_sent.pop_back();

    return ret_sent;
}


int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cout << "Usage: " << argv[0] << " encoder.onnx decoder.onnx input_lang output_lang gpu/cpu" << std::endl;
        return 0;
    }

    std::string file_name("..//data//");
    //file_name += std::string(argv[4]) + "-" + std::string(argv[3]) + "_procd.txt";
    file_name += std::string("eng") + "-" + std::string("fra") + "_procd.txt";
    std::cout << "file_nam = " << file_name << std::endl;

    std::ifstream ifs(file_name);
    if (!ifs.is_open())
    {
        std::cout << "File " << file_name << " open error!" << std::endl;
        return 1;
    }

    std::string use_gpu(argv[5]);
    bool b_use_gpu = false;
    migraphx::target t = migraphx::cpu::target{};
    if (use_gpu == "gpu")
    {
        b_use_gpu = true;
        t = migraphx::gpu::target{};
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
        //std::cout << "line " << line_index++ << ": " << line << std::endl;
        std::string input_sent{}, output_sent{};
        if (pos != std::string::npos)
        {
            input_sent = line.substr(0, pos);
            output_sent = line.substr(pos + 1);
        }
        else
        {
            std::cout << "File " << file_name << ", Line " << line_index;
            std::cout << " does not contain two languages!" << std::endl;
            return 1;
        }

        all_sentences.push_back(std::make_pair(input_sent, output_sent));
        input_lang.add_sentence(input_sent);
        output_lang.add_sentence(output_sent);
    }

    // load the models for the encoder and decoder
    migraphx::program encoder = load_onnx_file(argv[1]);
    migraphx::program decoder = load_onnx_file(argv[2]);

    // compile the encoder and decoder programs
    encoder.compile(t);
    decoder.compile(t);

    int hidden_size = 256;
    int n_words_in_lan = input_lang.get_word_num();
    int n_words_out_lan = output_lang.get_word_num();

    srand(time(nullptr));

    int sent_num = 500;
    // get start time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int sent_no = 0; sent_no < sent_num; ++sent_no)
    {
        std::cout << "sent_no = " << sent_no << std::endl;
        //auto sent_pair = get_random_sentence_pair(all_sentences);
        auto sent_pair = all_sentences.at(sent_no * 10);
        std::vector<std::string> vec_words{};
        vec_words = evaluate(encoder, decoder, t, input_lang, output_lang, 
                        hidden_size, max_sent_len, sent_pair.first);
        auto output_sentence = convert_to_sentence(vec_words);
        std::cout << "> " << sent_pair.first << std::endl;
        std::cout << "= " << sent_pair.second << std::endl;
        std::cout << "< " << output_sentence << std::endl;
        std::cout << std::endl;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    milliseconds ms = duration_cast<milliseconds>(t2 - t1);
    std::cout << "It takes " << ms.count() << " ms to translate " << sent_num << " sentences!" << std::endl;

    return 0;
}
