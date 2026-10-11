#pragma once
#include <string>
#include <vector>
#include "tools/cabana/streams/cereal_series.h"

struct PythonSnippet {
  std::string name;
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string code;  // must define: def derive(value, v1, v2, v3, v4, time): return <number>
};

// Evaluates snippet over the series in `map`, appends result under snippet.name.
// Returns empty string on success, error message on failure.
std::string pythonEvaluate(const PythonSnippet &snippet, const CerealSeriesMap &map, CerealSeriesMap &out);
