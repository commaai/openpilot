#include "tools/cabana/streams/cerealseries.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <string_view>

namespace cabana {
namespace {

std::optional<double> numericValue(capnp::DynamicValue::Reader value) {
  switch (value.getType()) {
    case capnp::DynamicValue::BOOL: return value.as<bool>() ? 1.0 : 0.0;
    case capnp::DynamicValue::INT: return static_cast<double>(value.as<int64_t>());
    case capnp::DynamicValue::UINT: return static_cast<double>(value.as<uint64_t>());
    case capnp::DynamicValue::FLOAT: {
      double result = value.as<double>();
      return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
    }
    case capnp::DynamicValue::ENUM: return value.as<capnp::DynamicEnum>().getRaw();
    default: return std::nullopt;
  }
}

bool activeField(capnp::DynamicStruct::Reader reader, capnp::StructSchema::Field field) {
  if (field.getProto().getDiscriminantValue() == 65535) return true;
  KJ_IF_MAYBE(active, reader.which()) {
    return active->getIndex() == field.getIndex();
  }
  return false;
}

}  // namespace

struct CerealDiscovery::Impl {
  struct Node {
    std::string path;
    bool complete = false;
    struct Field {
      capnp::StructSchema::Field schema;
      std::unique_ptr<Node> node;
    };
    std::vector<Field> fields;
    std::vector<std::unique_ptr<Node>> elements;
    explicit Node(std::string path) : path(std::move(path)) {}

    void observe(capnp::DynamicValue::Reader value, std::set<std::string> &names, int depth) {
      if (complete) return;
      if (depth > 64) throw std::runtime_error("Cereal schema exceeds the supported nesting depth");
      switch (value.getType()) {
        case capnp::DynamicValue::TEXT:
        case capnp::DynamicValue::BOOL:
        case capnp::DynamicValue::INT:
        case capnp::DynamicValue::UINT:
        case capnp::DynamicValue::FLOAT:
        case capnp::DynamicValue::ENUM:
          names.insert(path);
          complete = true;
          return;
        case capnp::DynamicValue::STRUCT: {
          auto reader = value.as<capnp::DynamicStruct>();
          if (fields.empty()) {
            for (auto field : reader.getSchema().getFields()) {
              auto type = field.getType();
              if (type.isVoid() || type.isData() || type.isInterface() || type.isAnyPointer()) continue;
              fields.push_back({field, std::make_unique<Node>(path + "/" + field.getProto().getName().cStr())});
            }
          }
          bool fixed = true;
          for (auto &field : fields) {
            if (!field.node->complete && activeField(reader, field.schema)) {
              auto type = field.schema.getType();
              if (!(type.isStruct() || type.isList()) || reader.has(field.schema)) {
                field.node->observe(reader.get(field.schema), names, depth + 1);
              }
            }
            fixed &= field.node->complete;
          }
          complete = fixed;
          return;
        }
        case capnp::DynamicValue::LIST: {
          auto list = value.as<capnp::DynamicList>();
          while (elements.size() < list.size()) {
            elements.push_back(std::make_unique<Node>(path + "/" + std::to_string(elements.size())));
          }
          for (uint i = 0; i < list.size(); ++i) {
            if (!elements[i]->complete) elements[i]->observe(list[i], names, depth + 1);
          }
          return;  // lists can grow in later messages
        }
        default:
          complete = true;
          return;
      }
    }
  };
  std::map<uint16_t, std::unique_ptr<Node>> roots;
  std::set<std::string> names;
};

CerealDiscovery::CerealDiscovery() : impl_(std::make_unique<Impl>()) {}
CerealDiscovery::~CerealDiscovery() = default;

void CerealDiscovery::observe(cereal::Event::Reader event) {
  auto reader = capnp::toDynamic(event);
  KJ_IF_MAYBE(field, reader.which()) {
    auto &node = impl_->roots[field->getIndex()];
    const std::string name = field->getProto().getName().cStr();
    if (!node) {
      node = std::make_unique<Impl::Node>(name);
      impl_->names.insert(name + "/__valid");
      impl_->names.insert(name + "/__logMonoTimeSeconds");
    }
    node->observe(reader.get(*field), impl_->names, 0);
  }
}

std::vector<std::string> CerealDiscovery::names() const {
  return {impl_->names.begin(), impl_->names.end()};
}

std::vector<std::string> cerealSignalPaths(cereal::Event::Reader event) {
  CerealDiscovery discovery;
  discovery.observe(event);
  return discovery.names();
}

namespace {
std::optional<capnp::DynamicValue::Reader> resolveCerealValue(cereal::Event::Reader event, const std::string &path) {
  if (path.empty() || path.front() == '/' || path.back() == '/') return std::nullopt;
  capnp::DynamicValue::Reader value = capnp::toDynamic(event);
  size_t begin = 0;
  while (begin < path.size()) {
    size_t end = path.find('/', begin);
    if (end == std::string::npos) end = path.size();
    std::string_view part(path.data() + begin, end - begin);
    if (part.empty()) return std::nullopt;
    if (value.getType() == capnp::DynamicValue::STRUCT) {
      auto reader = value.as<capnp::DynamicStruct>();
      const std::string name(part);
      auto field = reader.getSchema().findFieldByName(name.c_str());
      KJ_IF_MAYBE(found, field) {
        if (!activeField(reader, *found)) return std::nullopt;
        auto type = found->getType();
        if ((type.isStruct() || type.isList() || type.isText()) && !reader.has(*found)) return std::nullopt;
        value = reader.get(*found);
      } else {
        return std::nullopt;
      }
    } else if (value.getType() == capnp::DynamicValue::LIST) {
      auto list = value.as<capnp::DynamicList>();
      uint index = 0;
      auto [ptr, error] = std::from_chars(part.data(), part.data() + part.size(), index);
      if (error != std::errc() || ptr != part.data() + part.size() || index >= list.size()) return std::nullopt;
      value = list[index];
    } else {
      return std::nullopt;
    }
    begin = end + 1;
  }
  return value;
}
}  // namespace

std::optional<double> cerealSignalValue(cereal::Event::Reader event, const std::string &path) {
  if (path.empty() || path.front() == '/' || path.back() == '/') return std::nullopt;
  const auto slash = path.find('/');
  if (slash != std::string::npos && path.substr(slash + 1, 2) == "__") {
    auto reader = capnp::toDynamic(event);
    KJ_IF_MAYBE(field, reader.which()) {
      if (path.substr(0, slash) != field->getProto().getName().cStr()) return std::nullopt;
      auto name = path.substr(slash + 1);
      if (name == "__valid") return event.getValid() ? 1.0 : 0.0;
      if (name == "__logMonoTimeSeconds") return event.getLogMonoTime() * 1e-9;
    }
    return std::nullopt;
  }
  auto value = resolveCerealValue(event, path);
  return value ? numericValue(*value) : std::nullopt;
}

std::optional<std::string> cerealSignalText(cereal::Event::Reader event, const std::string &path) {
  auto value = resolveCerealValue(event, path);
  if (!value || value->getType() != capnp::DynamicValue::TEXT) return std::nullopt;
  auto text = value->as<capnp::Text>();
  return std::string(text.begin(), text.size());
}

std::map<uint16_t, std::string> cerealEnumLabels(const std::string &path) {
  capnp::Type type = capnp::Schema::from<cereal::Event>().asStruct();
  size_t start = 0;
  while (start < path.size()) {
    size_t end = path.find('/', start);
    if (end == std::string::npos) end = path.size();
    auto part = path.substr(start, end - start);
    if (type.isStruct()) {
      auto field = type.asStruct().findFieldByName(part.c_str());
      KJ_IF_MAYBE(found, field) { type = found->getType(); }
      else { return {}; }
    } else if (type.isList()) {
      uint index;
      auto [ptr, error] = std::from_chars(part.data(), part.data() + part.size(), index);
      if (error != std::errc() || ptr != part.data() + part.size()) return {};
      type = type.asList().getElementType();
    } else { return {}; }
    start = end + 1;
  }
  std::map<uint16_t, std::string> result;
  if (type.isEnum()) {
    for (auto enumerant : type.asEnum().getEnumerants()) {
      result[enumerant.getOrdinal()] = enumerant.getProto().getName().cStr();
    }
  }
  return result;
}

std::optional<double> heldValue(const CerealPoints &points, uint64_t time) {
  auto it = std::upper_bound(points.begin(), points.end(), time,
                             [](uint64_t requested, const auto &p) { return requested < p.mono_time; });
  if (it == points.begin()) return std::nullopt;
  return std::prev(it)->value;
}

CerealPoints transformSeries(const CerealPoints &points, const SeriesTransform &transform) {
  if (!std::isfinite(transform.scale) || !std::isfinite(transform.offset) ||
      !std::isfinite(transform.derivative_dt) || !std::isfinite(transform.time_offset) || transform.derivative_dt < 0) {
    throw std::invalid_argument("Transforms require finite scale/offset and a nonnegative derivative divisor");
  }
  CerealPoints output;
  output.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    if (i && points[i].mono_time < points[i - 1].mono_time) throw std::invalid_argument("Series timestamps are not sorted");
    if (points[i].text) throw std::invalid_argument("Text states cannot be used in numerical transforms");
    double value = points[i].value;
    if (transform.derivative) {
      if (!i) continue;
      // Same transform convention as JotPluggler: an explicit positive divisor
      // overrides elapsed time; zero uses adjacent sample timestamps.
      const double dt = transform.derivative_dt > 0 ? transform.derivative_dt :
                         (points[i].mono_time - points[i - 1].mono_time) * 1e-9;
      if (dt <= 0) continue;
      value = (value - points[i - 1].value) / dt;
    }
    value = value * transform.scale + transform.offset;
    const long double shifted = points[i].mono_time + static_cast<long double>(transform.time_offset) * 1e9L;
    if (shifted < 0 || shifted > static_cast<long double>(UINT64_MAX)) throw std::invalid_argument("Time offset exceeds the monotonic clock range");
    output.push_back({static_cast<uint64_t>(std::round(shifted)), std::isfinite(value) ? value : std::numeric_limits<double>::quiet_NaN()});
  }
  return output;
}

}  // namespace cabana
