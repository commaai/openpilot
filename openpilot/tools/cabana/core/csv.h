#pragma once
#include <map>
#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

namespace cabana {
struct CsvPoint { double time, value; };
struct CsvSeries { std::string name; std::vector<CsvPoint> points; std::map<uint16_t, std::string> labels; };
void writeSeriesCsv(std::ostream &output, const std::vector<CsvSeries> &series);
}  // namespace cabana
