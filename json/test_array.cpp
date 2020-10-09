#include <iostream>
#include <nlohmann/json.hpp>

using nlohmann::json;

int main() {
    json j;
    std::vector<int> vec = {1, 2, 3, 4};
    j = vec;

    for (auto& v : j)
    {
        int jv = v;
        std::cout << "jv = " << jv << std::endl;
    }

    json jnull;
    std::string str = "null";
    jnull = json::parse(str);
    std::cout << "is_null = " << jnull.is_null() << std::endl;
    std::cout << "null_str = " << jnull.dump() << std::endl;
    return 0;
}

