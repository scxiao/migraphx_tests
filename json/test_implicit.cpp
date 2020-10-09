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

namespace nlohmann {
	template <>
	struct adl_serializer<ns::person> {
		static void to_json(json& j, const ns::person& p) {
			// calls the "to_json" method in T's namespace
            j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
		}

		static void from_json(const json& j, ns::person& p) {
			// same thing, but with the "from_json" method
			j.at("name").get_to(p.name);
			j.at("address").get_to(p.address);
			j.at("age").get_to(p.age);
		}
	};
}

namespace ns {
//    void to_json(json& j, const person& p) {
//        j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
//    }
//
//    void from_json(const json& j, person& p) {
//        j.at("name").get_to(p.name);
//        j.at("address").get_to(p.address);
//        j.at("age").get_to(p.age);
//    }

    std::string convert(person& p)
    {
        json j = p;
        return j.dump();
    }
} // namespace ns


int main() {

	ns::person p = {"abc", "111 address", 32};
    std::string str = ns::convert(p);
    std::cout << "str = " << str << std::endl;

    json j1 = json::array();
    std::cout << "j1 = " << j1 << std::endl;
    json j2 = json::object();
    std::cout << "j2 = " << j2 << std::endl;

    return 0;
}
