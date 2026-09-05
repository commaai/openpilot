#include "tools/cabana/core/logsignals.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "json11/json11.hpp"

namespace cabana {

void appendLogSignal(capnp::DynamicValue::Reader value, const std::string &path, uint64_t mono_time, LogSignals &signals) {
  double number;
  switch (value.getType()) {
    case capnp::DynamicValue::BOOL: number = value.as<bool>(); break;
    case capnp::DynamicValue::INT: number = value.as<int64_t>(); break;
    case capnp::DynamicValue::UINT: number = value.as<uint64_t>(); break;
    case capnp::DynamicValue::FLOAT: number = value.as<double>(); break;
    case capnp::DynamicValue::ENUM: number = value.as<capnp::DynamicEnum>().getRaw(); break;
    case capnp::DynamicValue::STRUCT: {
      auto reader = value.as<capnp::DynamicStruct>();
      auto append_field = [&](capnp::StructSchema::Field field) {
        auto type = field.getType();
        if ((type.isStruct() || type.isList()) && !reader.has(field)) return;
        appendLogSignal(reader.get(field), path + "/" + field.getProto().getName().cStr(), mono_time, signals);
      };
      for (auto field : reader.getSchema().getNonUnionFields()) append_field(field);
      KJ_IF_MAYBE(field, reader.which()) { append_field(*field); }
      return;
    }
    case capnp::DynamicValue::LIST: {
      auto list = value.as<capnp::DynamicList>();
      for (uint32_t i = 0; i < list.size(); ++i) {
        appendLogSignal(list[i], path + "/" + std::to_string(i), mono_time, signals);
      }
      return;
    }
    default: return;
  }
  signals[path].push_back({mono_time, std::isfinite(number) ? number : std::numeric_limits<double>::quiet_NaN()});
}

std::vector<LogPoint> logPoints(const LogSegments &segments, const LogCurve &curve, uint64_t origin) {
  std::vector<std::pair<LogSample, int>> samples;
  for (const auto &[segment, signals] : segments) {
    if (auto it = signals->find(curve.signal); it != signals->end()) {
      for (const auto &sample : it->second) samples.emplace_back(sample, segment);
    }
  }
  std::stable_sort(samples.begin(), samples.end(), [](const auto &a, const auto &b) { return a.first.mono_time < b.first.mono_time; });
  std::vector<LogPoint> points;
  points.reserve(samples.size());
  int previous_segment = 0;
  for (const auto &[sample, segment] : samples) {
    // Subtract as integers first to retain nanosecond resolution without unsigned underflow.
    double sec = sample.mono_time >= origin ? (sample.mono_time - origin) / 1e9 : -((origin - sample.mono_time) / 1e9);
    double value = sample.value * curve.scale + curve.offset;
    if (!std::isfinite(value)) value = std::numeric_limits<double>::quiet_NaN();
    if (!points.empty() && points.back().x == sec) points.back().y = value;
    else {
      // Do not imply a continuous trace (or a derivative) across an unloaded segment.
      if (!points.empty() && std::abs(int64_t(segment) - previous_segment) > 1) {
        points.push_back({sec, std::numeric_limits<double>::quiet_NaN()});
      }
      points.push_back({sec, value});
    }
    previous_segment = segment;
  }
  if (curve.derivative) {
    for (size_t i = points.size(); i > 0; --i) {
      points[i - 1].y = i > 1 && points[i - 1].x > points[i - 2].x
                             ? (points[i - 1].y - points[i - 2].y) / (points[i - 1].x - points[i - 2].x)
                             : std::numeric_limits<double>::quiet_NaN();
      if (!std::isfinite(points[i - 1].y)) points[i - 1].y = std::numeric_limits<double>::quiet_NaN();
    }
  }
  return points;
}

std::vector<LogPlot> parseLogLayout(const std::string &text) {
  std::string error;
  auto root = json11::Json::parse(text, error);
  if (!error.empty() || !root.is_object() || root["version"].number_value() != 1 || !root["plots"].is_array()) {
    throw std::runtime_error("Expected a Cabana log layout with version 1 and a plots array.");
  }
  std::vector<LogPlot> plots;
  for (const auto &plot : root["plots"].array_items()) {
    if (!plot.is_array() || plot.array_items().empty()) throw std::runtime_error("Each plot must contain at least one curve.");
    LogPlot curves;
    for (const auto &item : plot.array_items()) {
      if (!item.is_object() || !item["signal"].is_string() || item["signal"].string_value().empty()) {
        throw std::runtime_error("Every curve needs a nonempty signal name.");
      }
      LogCurve curve{item["signal"].string_value()};
      for (auto entry : {std::pair{"scale", &curve.scale}, std::pair{"offset", &curve.offset}}) {
        if (item.object_items().count(entry.first)) {
          if (!item[entry.first].is_number() || !std::isfinite(item[entry.first].number_value())) {
            throw std::runtime_error(std::string(entry.first) + " must be a finite number.");
          }
          *entry.second = item[entry.first].number_value();
        }
      }
      if (item.object_items().count("derivative")) {
        if (!item["derivative"].is_bool()) throw std::runtime_error("derivative must be a boolean.");
        curve.derivative = item["derivative"].bool_value();
      }
      curves.push_back(std::move(curve));
    }
    plots.push_back(std::move(curves));
  }
  return plots;
}

std::string serializeLogLayout(const std::vector<LogPlot> &plots) {
  json11::Json::array result;
  for (const auto &plot : plots) {
    json11::Json::array curves;
    for (const auto &curve : plot) {
      curves.push_back(json11::Json::object{{"signal", curve.signal}, {"scale", curve.scale},
                                          {"offset", curve.offset}, {"derivative", curve.derivative}});
    }
    result.push_back(std::move(curves));
  }
  return json11::Json(json11::Json::object{{"version", 1}, {"plots", std::move(result)}}).dump();
}

std::string logCsv(const LogSegments &segments, const std::vector<LogPlot> &plots, uint64_t origin, double begin, double end) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "plot,signal,time,value,scale,offset,derivative\n" << std::setprecision(17);
  for (size_t i = 0; i < plots.size(); ++i) {
    for (const auto &curve : plots[i]) {
      std::string name = "\"";
      for (char c : curve.signal) name += c == '"' ? "\"\"" : std::string(1, c);
      name += '"';
      for (const auto &p : logPoints(segments, curve, origin)) {
        if (p.x < begin || p.x > end) continue;
        out << i + 1 << ',' << name << ',' << p.x << ',';
        if (std::isfinite(p.y)) out << p.y;
        out << ',' << curve.scale << ',' << curve.offset << ',' << curve.derivative << '\n';
      }
    }
  }
  return out.str();
}

}  // namespace cabana
