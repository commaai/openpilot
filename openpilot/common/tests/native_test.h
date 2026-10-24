#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

inline void native_test_check(bool condition, const char *expression, const char *file, int line) {
  if (!condition) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": check failed: " + expression);
  }
}

#define CHECK(condition) native_test_check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
#define REQUIRE(...) CHECK((__VA_ARGS__))

template <typename Function>
int run_native_test(Function &&function) {
  try {
    function();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
