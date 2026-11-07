#include "tools/cabana/core/csv.h"
#include <cmath>
#include <iomanip>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace cabana {
namespace {
void quote(std::ostream &output, const std::string &value) {
  output << '"';
  for (char c : value) { if (c == '"') output << '"'; output << c; }
  output << '"';
}
}
void writeSeriesCsv(std::ostream &output, const std::vector<CsvSeries> &series) {
  output << "time_s";
  for (const auto &item : series) { output << ','; quote(output, item.name); }
  output << '\n' << std::setprecision(17);
  using Entry = std::tuple<double, size_t, size_t>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> next;
  std::vector<std::optional<double>> previous(series.size());
  for (size_t i = 0; i < series.size(); ++i) {
    if (!series[i].points.empty()) next.emplace(series[i].points[0].time, i, 0);
  }
  while (!next.empty()) {
    const double time = std::get<0>(next.top());
    if (!std::isfinite(time)) throw std::invalid_argument("Cannot export nonfinite timestamps");
    while (!next.empty() && std::get<0>(next.top()) == time) {
      const auto [sample_time, index, point] = next.top(); next.pop();
      previous[index] = series[index].points[point].value;
      const auto following = point + 1;
      if (following < series[index].points.size()) {
        if (series[index].points[following].time < sample_time) throw std::invalid_argument("Cannot export unsorted samples");
        next.emplace(series[index].points[following].time, index, following);
      }
    }
    output << time;
    for (size_t i = 0; i < series.size(); ++i) {
      output << ',';
      if (!previous[i] || !std::isfinite(*previous[i])) continue;
      const double value = *previous[i];
      if (value >= 0 && value <= 65535 && std::floor(value) == value && series[i].labels.count(value)) quote(output, series[i].labels.at(value));
      else output << value;
    }
    output << '\n';
    if (!output) throw std::runtime_error("Cannot write CSV output (check free disk space)");
  }
  output.flush();
  if (!output) throw std::runtime_error("Cannot flush CSV output (check free disk space)");
}
}  // namespace cabana
