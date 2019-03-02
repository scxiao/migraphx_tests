#ifndef __MIGRAPHX_EXAMPLE_LANGUAGE_HPP_
#define __MIGRAPHX_EXAMPLE_LANGUAGE_HPP_
#include <string>
#include <unordered_map>
#include <vector>


class CLanguage
{
public:
    CLanguage(std::string nm);
    void add_word(std::string &word);
    void add_sentence(std::string &sent);
    std::string get_word(std::int index);
    int get_word_index(std::string &word);
    int get_word_num() { return static_cast<int>(index2word.size()); }

    std::vector<int> get_sentence_indices(std::string &sent)
private:
    void init();

private:
    std::string name;
    std::unordered_map<std::string, int> word2count;
    std::unordered_map<std::string, int> word2index;
    std::vector<std::string> index2word;
};

#endif
