#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using nlohmann::json;


int main() {
	json o;
	o["foo"] = 23;
	o["bar"] = false;
	o["baz"] = 3.141;

	// also use emplace
	//o.emplace("weather", "sunny");

	// special iterator member functions for objects
	for (json::iterator it = o.begin(); it != o.end(); ++it) {
	  std::cout << it.key() << " : " << it.value() << "\n";
	}

	// the same code as range for
	for (auto& el : o.items()) {
	  std::cout << el.key() << " : " << el.value() << "\n";
	}

//	// even easier with structured bindings (C++17)
//	for (auto& [key, value] : o.items()) {
//	  std::cout << key << " : " << value << "\n";
//	}
//
	// find an entry
	if (o.contains("foo")) {
	  // there is an entry with key "foo"
	}

	// or via find and an iterator
	if (o.find("foo") != o.end()) {
	  // there is an entry with key "foo"
	}

	// or simpler using count()
	int foo_present = o.count("foo"); // 1
	int fob_present = o.count("fob"); // 0

    json bval = false;
    bool b = bval.get<bool>();
    std::cout << "b = " << b << std::endl;

	// delete an entry
	o.erase("foo");

    return 0;
}
