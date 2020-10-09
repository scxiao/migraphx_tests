#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using nlohmann::json;

void json_check(const json& jj)
{

    std::cout << "jj = " << jj.dump(6) << std::endl;
    std::cout << "jj_size = " << jj.size() << std::endl;
    std::cout << "jj.is_null = "  << jj.is_null() << std::endl;
	std::cout << "jj.is_boolean = " << jj.is_boolean() << std::endl;
	std::cout << "jj.is_number = "  << jj.is_number() << std::endl;
	std::cout << "jj.is_object = "  << jj.is_object() << std::endl;
	std::cout << "jj.is_array = "   << jj.is_array() << std::endl;
	std::cout << "jj.is_string = "  << jj.is_string() << std::endl;
}

int main() {
    std::string str = "{ \"happy\": true, \"pi\": 3.141 }";
    json jj = json::parse(str);
    for (auto& i : jj.items())
    {
        std::cout << "key = " << i.key() << ", value = " << i.value() << std::endl;
    }
    auto type = jj.type();
    int t = static_cast<int>(type);
    std::cout << "t = " << t << std::endl;
    if (type == json::value_t::number_integer)
    {
        std::cout << "integer" << std::endl;
    }
    json_check(jj);

    json j = 3;
    std::cout << "j = " << j << std::endl;
    json_check(j);

    json m;
    m["ab"] = 2;
    for (auto kv : m.items())
    {
        std::cout << "m_key = " << kv.key() << std::endl;
        json v = kv.value();
        std::cout << "m_val = " << v << std::endl;
    }

    return 0;
}
