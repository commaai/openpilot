#include "tools/cabana/streams/cereal_extractor.h"

#include <capnp/dynamic.h>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

static void walkValue(const std::string &path, capnp::DynamicValue::Reader value,
                      double t, CerealSeriesMap &out, int depth);

static void walkStruct(const std::string &path, capnp::DynamicStruct::Reader reader,
                       double t, CerealSeriesMap &out, int depth) {
  if (depth > 6) return;
  for (auto field : reader.getSchema().getNonUnionFields()) {
    walkValue(path + "/" + field.getProto().getName().cStr(), reader.get(field), t, out, depth + 1);
  }
  KJ_IF_MAYBE(which, reader.which()) {
    walkValue(path + "/" + which->getProto().getName().cStr(), reader.get(*which), t, out, depth + 1);
  }
}

static void walkValue(const std::string &path, capnp::DynamicValue::Reader value,
                      double t, CerealSeriesMap &out, int depth) {
  switch (value.getType()) {
    case capnp::DynamicValue::BOOL:
      out[path].push_back({t, value.as<bool>() ? 1.0 : 0.0});
      break;
    case capnp::DynamicValue::INT:
      out[path].push_back({t, (double)value.as<int64_t>()});
      break;
    case capnp::DynamicValue::UINT:
      out[path].push_back({t, (double)value.as<uint64_t>()});
      break;
    case capnp::DynamicValue::FLOAT:
      out[path].push_back({t, value.as<double>()});
      break;
    case capnp::DynamicValue::ENUM:
      out[path].push_back({t, (double)value.as<capnp::DynamicEnum>().getRaw()});
      break;
    case capnp::DynamicValue::STRUCT:
      walkStruct(path, value.as<capnp::DynamicStruct>(), t, out, depth);
      break;
    case capnp::DynamicValue::LIST: {
      auto list = value.as<capnp::DynamicList>();
      for (size_t i = 0; i < list.size() && i < 5; i++) {
        walkValue(path + "/" + std::to_string(i), list[i], t, out, depth + 1);
      }
      break;
    }
    default:
      break;
  }
}

void extractCerealEvent(kj::ArrayPtr<const capnp::word> data, double t, CerealSeriesMap &out) {
  capnp::FlatArrayMessageReader reader(data);
  auto event = reader.getRoot<cereal::Event>();
  capnp::DynamicStruct::Reader dyn = capnp::toDynamic(event);
  KJ_IF_MAYBE(which, dyn.which()) {
    auto value = dyn.get(*which);
    if (value.getType() == capnp::DynamicValue::STRUCT) {
      walkStruct("/" + std::string(which->getProto().getName().cStr()),
                 value.as<capnp::DynamicStruct>(), t, out, 0);
    }
  }
}

void extractCerealEvent(const Event &e, double t, CerealSeriesMap &out) {
  capnp::FlatArrayMessageReader reader(e.data);
  auto event = reader.getRoot<cereal::Event>();
  capnp::DynamicStruct::Reader dyn = capnp::toDynamic(event);

  KJ_IF_MAYBE(which, dyn.which()) {
    auto value = dyn.get(*which);
    if (value.getType() == capnp::DynamicValue::STRUCT) {
      walkStruct("/" + std::string(which->getProto().getName().cStr()),
                 value.as<capnp::DynamicStruct>(), t, out, 0);
    }
  }
}
