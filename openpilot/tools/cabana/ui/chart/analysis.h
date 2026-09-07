#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace chart {

enum class Transform { None, Derivative, Integral, MovingAverage };
inline constexpr const char *TRANSFORM_NAMES[] = {"Original", "Derivative", "Integral", "Moving average"};

struct TransformSettings {
  Transform type = Transform::None;
  double scale = 1;
  double offset = 0;
  int window = 10;
  bool original() const { return type == Transform::None && scale == 1 && offset == 0; }
};

// State is retained across live batches and reset when earlier route segments arrive.
struct TransformState {
  double integral = 0, sum = 0, previous_time = 0, previous_value = 0;
  bool has_previous = false;
  std::deque<double> window;

  std::optional<double> append(double time, double raw, const TransformSettings &settings) {
    double value = raw * settings.scale + settings.offset;
    const double dt = has_previous ? time - previous_time : 0;
    const double previous = has_previous ? previous_value : value;
    previous_time = time;
    previous_value = value;
    has_previous = true;
    switch (settings.type) {
      case Transform::Derivative:
        if (dt <= 0) return std::nullopt;
        value = (value - previous) / dt;
        break;
      case Transform::Integral:
        integral += dt * (value / 2 + previous / 2);
        value = integral;
        break;
      case Transform::MovingAverage:
        window.push_back(value);
        sum += value;
        if (window.size() > std::max(settings.window, 1)) { sum -= window.front(); window.pop_front(); }
        value = sum / window.size();
        break;
      case Transform::None: break;
    }
    return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
  }
};

// Apply scale/offset before the transform. Points must be ordered by time.
// Duplicate timestamps have no derivative; integration uses the trapezoidal rule.
template <class Point>
std::vector<Point> transform(const std::vector<Point> &raw, const TransformSettings &settings) {
  std::vector<Point> result;
  result.reserve(raw.size());
  TransformState state;
  for (const auto &pt : raw) {
    if (auto value = state.append(pt.x, pt.y, settings)) result.emplace_back(pt.x, *value);
  }
  return result;
}

inline std::string csvField(const std::string &text) {
  std::string escaped = "\"";
  for (char c : text) {
    if (c == '"') escaped += '"';
    escaped += c;
  }
  return escaped + '"';
}

}  // namespace chart
