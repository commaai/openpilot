#include "tools/cabana/analysis/telemetry.h"

#include <algorithm>
#include <cmath>

namespace cabana {
namespace {
void extract(const std::string &path, capnp::DynamicValue::Reader value, double time, Telemetry &out) {
  double number = 0;
  switch (value.getType()) {
    case capnp::DynamicValue::BOOL: number = value.as<bool>(); break;
    case capnp::DynamicValue::INT: number = value.as<int64_t>(); break;
    case capnp::DynamicValue::UINT: number = value.as<uint64_t>(); break;
    case capnp::DynamicValue::FLOAT: number = value.as<double>(); break;
    case capnp::DynamicValue::ENUM: number = value.as<capnp::DynamicEnum>().getRaw(); break;
    case capnp::DynamicValue::LIST: {
      auto list = value.as<capnp::DynamicList>();
      for (size_t i = 0; i < list.size(); ++i) extract(path + '/' + std::to_string(i), list[i], time, out);
      return;
    }
    case capnp::DynamicValue::STRUCT: {
      auto node = value.as<capnp::DynamicStruct>();
      for (auto field : node.getSchema().getFields()) {
        // has() tests union membership; scalars with their default (zero/false) remain meaningful.
        if (node.has(field)) extract(path + '/' + field.getProto().getName().cStr(), node.get(field), time, out);
      }
      return;
    }
    default: return;
  }
  if (std::isfinite(number)) out[path].emplace_back(time, number);
}
}  // namespace

void extractTelemetry(cereal::Event::Reader event, Telemetry &out) {
  auto node = capnp::toDynamic(event);
  KJ_IF_MAYBE(field, node.which()) {
    const std::string name = field->getProto().getName().cStr();
    if (name == "can" || name == "sendcan") return;  // Cabana's DBC decoder owns CAN.
    const double time = event.getLogMonoTime() * 1e-9;
    const std::string path = '/' + name;
    extract(path, node.get(*field), time, out);
    out[path + "/__logMonoTime"].emplace_back(time, event.getLogMonoTime());
    out[path + "/__logMonoTimeSeconds"].emplace_back(time, time);
    out[path + "/__valid"].emplace_back(time, event.getValid());
  }
}

void mergeTelemetry(Telemetry &destination, Telemetry source) {
  for (auto &[path, samples] : source) {
    auto &points = destination[path];
    const size_t previous = points.size();
    points.insert(points.end(), samples.begin(), samples.end());
    std::inplace_merge(points.begin(), points.begin() + previous, points.end(), [](const auto &a, const auto &b) { return a.x < b.x; });
  }
}
}  // namespace cabana
