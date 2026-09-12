#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace test_runner {

struct TestCase {
  const char *name;
  void (*function)();
};

inline std::vector<TestCase> &tests() {
  static std::vector<TestCase> registered_tests;
  return registered_tests;
}

struct Registrar {
  Registrar(const char *name, void (*function)()) {
    tests().push_back({name, function});
  }
};

struct TestState {
  std::size_t assertion_count = 0;
  bool failed = false;
  std::string failure_message;
};

inline TestState &state() {
  static TestState test_state;
  return test_state;
}

inline bool require(bool result, const char *expression, const char *file, int line) {
  state().assertion_count++;
  if (!result) {
    state().failed = true;
    state().failure_message = std::string(file) + ":" + std::to_string(line) +
                              ": REQUIRE(" + expression + ") failed";
  }
  return result;
}

inline int run_all() {
  if (tests().empty()) {
    std::cerr << "[FAIL] no tests registered\n";
    return 1;
  }
  std::size_t passed = 0;

  for (const TestCase &test : tests()) {
    state() = {};
    bool threw = false;
    try {
      test.function();
    } catch (const std::exception &error) {
      threw = true;
      std::cerr << "[FAIL] " << test.name << "\n  " << error.what() << '\n';
    } catch (...) {
      threw = true;
      std::cerr << "[FAIL] " << test.name << "\n  unknown exception\n";
    }

    if (threw) {
      continue;
    } else if (state().failed) {
      std::cerr << "[FAIL] " << test.name << "\n  " << state().failure_message << '\n';
    } else if (state().assertion_count == 0) {
      std::cerr << "[FAIL] " << test.name << "\n  no assertions executed\n";
    } else {
      ++passed;
      std::cout << "[PASS] " << test.name << '\n';
    }
  }

  std::cout << '\n' << passed << "/" << tests().size() << " tests passed\n";
  return passed == tests().size() ? 0 : 1;
}

}  // namespace test_runner

#define TEST_RUNNER_JOIN_IMPL(left, right) left##right
#define TEST_RUNNER_JOIN(left, right) TEST_RUNNER_JOIN_IMPL(left, right)

#define TEST_CASE(name)                                                         \
  static void TEST_RUNNER_JOIN(test_case_, __LINE__)();                         \
  static const test_runner::Registrar TEST_RUNNER_JOIN(test_registrar_,         \
                                                        __LINE__)(               \
      name, TEST_RUNNER_JOIN(test_case_, __LINE__));                            \
  static void TEST_RUNNER_JOIN(test_case_, __LINE__)()

#define REQUIRE(expression)                                                     \
  do {                                                                          \
    if (!test_runner::require(static_cast<bool>(expression), #expression,       \
                              __FILE__, __LINE__)) {                             \
      return;                                                                   \
    }                                                                           \
  } while (false)

int main() {
  return test_runner::run_all();
}
