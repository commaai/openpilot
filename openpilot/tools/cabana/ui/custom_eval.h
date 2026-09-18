#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cabana {

struct CustomPythonSeries {
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code;
};

struct PythonEvalResult {
  std::vector<double> xs;
  std::vector<double> ys;
};

PythonEvalResult evaluateCustomPythonSeries(
    const CustomPythonSeries &spec,
    const std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> &inputs);

}  // namespace cabana
