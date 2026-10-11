#pragma once
#include <string>
#include <vector>
#include "tools/cabana/streams/cereal_series.h"

struct LuaSnippet {
  std::string name;
  std::string global_init;   // runs once to set up state
  std::string function_body; // runs per sample; receives value, v1..v4, time; must return a number
  std::string linked_source; // primary cereal path
  std::vector<std::string> additional_sources; // v1..v4
};

// Evaluates snippet over the series in `map`, appends result under snippet.name.
// Returns empty string on success, error message on failure.
std::string luaEvaluate(const LuaSnippet &snippet, const CerealSeriesMap &map, CerealSeriesMap &out);
