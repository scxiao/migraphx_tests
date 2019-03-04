#include "s2s_utilities.hpp"

std::vector<std::string> convert_sent_to_words(std::string &sent, char delim)
{
    std::vector<std::string> words;
    std::size_t start_loc = sent.find_first_not_of(delim, 0);
    while (true)
    {
        std::size_t end_loc = sent.find(delim, start_loc);
        if (end_loc != std::string::npos)
        {
            words.push_back(sent.substr(start_loc, end_loc - start_loc));
            start_loc = sent.find_first_not_of(delim, end_loc);
            if (start_loc == std::string::npos)
            {
                break;
            }
        }
        else
        {
            words.push_back(sent.substr(start_loc));
            break;
        }
    }

    return words;
}


