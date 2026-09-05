#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <capnp/dynamic.h>

namespace cabana {

struct LogSample {
  uint64_t mono_time;
  double value;
};
using LogSeries = std::vector<LogSample>;
using LogSignals = std::map<std::string, LogSeries>;
// Immutable segments are shared with the UI; evicted replay segments release their plotted data too.
using LogSegments = std::map<int, std::shared_ptr<const LogSignals>>;

struct LogPoint {
  double x, y;
};

struct LogCurve {
  std::string signal;
  double scale = 1.0;
  double offset = 0.0;
  bool derivative = false;
};
using LogPlot = std::vector<LogCurve>;

// Flatten active numeric fields, including groups and lists, using PlotJuggler's slash-separated names.
// Text/data and absent pointers are omitted. Non-finite values become gaps, not zero-valued samples.
void appendLogSignal(capnp::DynamicValue::Reader value, const std::string &path, uint64_t mono_time, LogSignals &signals);
std::vector<LogPoint> logPoints(const LogSegments &segments, const LogCurve &curve, uint64_t origin);

// Layout parsing is transactional: invalid input throws before the caller replaces its current plots.
std::vector<LogPlot> parseLogLayout(const std::string &text);
std::string serializeLogLayout(const std::vector<LogPlot> &plots);
std::string logCsv(const LogSegments &segments, const std::vector<LogPlot> &plots, uint64_t origin, double begin, double end);

}  // namespace cabana
