#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using nlohmann::json;

namespace ns {
    struct person {
        std::string name;
        std::string address;
        int age;
    };
}

void to_json(ns::person& p, json& j)
{
    j["name"] = p.name;
    j["address abc"] = p.address;
    j["age"] = p.age;
}

void from_json(const json& j, ns::person& p)
{
    p = {j["name"].get<std::string>(), j["address abc"].get<std::string>(), j["age"].get<int>()};
}

int main() {
    ns::person p = {"Ned Flanders", "744 Evergreen Terrace", 60};

    json j;
    to_json(p, j);
    std::cout << "j = " << j.dump(4) << std::endl;

    return 0;
}
