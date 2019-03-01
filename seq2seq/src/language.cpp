#include "language.hpp"
#include "s2s_utilities.hpp"

CLanguage::CLanguage(std::string nm) : name(nm)
{
    init();
}

void CLanguage::add_word(std::string &word)
{
    // word not in diectory
    if (word2index.count(word) == 0)
    {
        word2index[word] = index2word.size();
        word2count[word] = 1;
        index2word.push_back(word);
    }
    else
    {
        word2count[word]++;
    }
}

void CLanguage::add_sentence(std::string &sent)
{
    auto words = convert_sent_to_words(sent);
    for_each(words.begin(), words.end(), [](auto &word) {
        add_word(word);
    });
}

std::string CLanguage::get_word(std::int index) {
    if (index < index2word.size() && index >= 0)
    {
        return index2word.at(index);
    }
    else 
    {
        return {};
    }
}

int CLanguage::get_word_index(std::string &word)
{
    if (word2index.count(word) > 0)
    {
        return word2index[word];
    }
    else
    {
        return -1;
    }
}

std::vector<int> CLanguage::get_sentence_indices(std::string &sent)
{
    auto words = convert_sent_to_words(sent);
    std::vector<int> sent_indices(words.size());
    std::transform(words.begin(), words.end(), sent_indices.begin(), [](auto &word) {
        return get_word_index(word);
    });

    return sent_indices;
}


void CLanguage::init() 
{
    index2word.push_back("SOS");
    index2word.push_back("EOS");
}

