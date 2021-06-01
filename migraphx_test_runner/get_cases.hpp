#ifndef __GET_CASES_HPP__
#define __GET_CASES_HPP__

#include <string>
#include <vector>

std::string last_folder_file(const std::string& path_str);
std::vector<std::string> get_test_cases(const std::string& path_str);
std::string get_path_last_part(const std::string& path_str);
std::string get_model_name(const std::string& path_str);

#endif

