#ifndef __MIGRAPHX_EXAMPLE_S2S_UTILITIES_HPP__
#define __MIGRAPHX_EXAMPLE_S2S_UTILITIES_HPP__

#include <vector>
#include <string>

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

extern const int SOS_token;
extern const int EOS_token;

std::vector<std::string> convert_sent_to_words(std::string &sent, char delim = ' ');

#endif

