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

struct TelemetryExtractor::Impl {
  struct Node {
    explicit Node(std::string path) : path(std::move(path)) {}
    std::string path;
    Samples *samples = nullptr;
    std::vector<std::pair<capnp::StructSchema::Field, std::unique_ptr<Node>>> fields;
    std::vector<std::unique_ptr<Node>> elements;

    void read(capnp::DynamicValue::Reader value, double time, Telemetry &out) {
      double number;
      switch (value.getType()) {
        case capnp::DynamicValue::BOOL: number = value.as<bool>(); break;
        case capnp::DynamicValue::INT: number = value.as<int64_t>(); break;
        case capnp::DynamicValue::UINT: number = value.as<uint64_t>(); break;
        case capnp::DynamicValue::FLOAT: number = value.as<double>(); break;
        case capnp::DynamicValue::ENUM: number = value.as<capnp::DynamicEnum>().getRaw(); break;
        case capnp::DynamicValue::STRUCT: {
          auto node = value.as<capnp::DynamicStruct>();
          if (fields.empty()) {
            for (auto field : node.getSchema().getFields()) {
              fields.emplace_back(field, std::make_unique<Node>(path + '/' + field.getProto().getName().cStr()));
            }
          }
          for (auto &[field, child] : fields) {
            if (node.has(field)) child->read(node.get(field), time, out);
          }
          return;
        }
        case capnp::DynamicValue::LIST: {
          auto list = value.as<capnp::DynamicList>();
          while (elements.size() < list.size()) elements.push_back(std::make_unique<Node>(path + '/' + std::to_string(elements.size())));
          for (size_t i = 0; i < list.size(); ++i) elements[i]->read(list[i], time, out);
          return;
        }
        default: return;
      }
      if (std::isfinite(number)) {
        if (!samples) samples = &out[path];
        samples->emplace_back(time, number);
      }
    }
  };
  struct Root {
    std::unique_ptr<Node> node;
    Samples *mono_time, *seconds, *valid;
  };
  explicit Impl(Telemetry &out) : out(out) {}
  Telemetry &out;
  std::map<uint16_t, Root> roots;
};

TelemetryExtractor::TelemetryExtractor(Telemetry &destination) : impl_(std::make_unique<Impl>(destination)) {}
TelemetryExtractor::~TelemetryExtractor() = default;

void TelemetryExtractor::extract(cereal::Event::Reader event) {
  if (event.which() == cereal::Event::Which::CAN || event.which() == cereal::Event::Which::SENDCAN) return;
  auto node = capnp::toDynamic(event);
  KJ_IF_MAYBE(field, node.which()) {
    const uint16_t index = field->getIndex();
    auto it = impl_->roots.find(index);
    if (it == impl_->roots.end()) {
      const std::string name = field->getProto().getName().cStr();
      const std::string path = '/' + name;
      Impl::Root root{std::make_unique<Impl::Node>(path), &impl_->out[path + "/__logMonoTime"],
                      &impl_->out[path + "/__logMonoTimeSeconds"], &impl_->out[path + "/__valid"]};
      it = impl_->roots.emplace(index, std::move(root)).first;
    }
    const double time = event.getLogMonoTime() * 1e-9;
    auto &root = it->second;
    root.node->read(node.get(*field), time, impl_->out);
    root.mono_time->emplace_back(time, event.getLogMonoTime());
    root.seconds->emplace_back(time, time);
    root.valid->emplace_back(time, event.getValid());
  }
}

void prepareTelemetryMerge(const TelemetrySnapshot &destination, Telemetry &batch) {
  for (auto &[path, samples] : batch) {
    auto old = destination.find(path);
    if (old == destination.end()) continue;
    std::vector<Sample> merged;
    merged.reserve(old->second->size() + samples.size());
    std::merge(old->second->begin(), old->second->end(), samples.begin(), samples.end(),
               std::back_inserter(merged), [](const auto &a, const auto &b) { return a.x < b.x; });
    samples.swap(merged);
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
