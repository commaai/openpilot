#include "tools/cabana/streams/cereal_extractor.h"

#include <algorithm>
#include <utility>

#include <capnp/dynamic.h>

namespace cabana {
namespace {

void extractValue(capnp::DynamicValue::Reader value, const std::string &path,
                  uint64_t mono_time, CerealSeriesMap &series) {
  switch (value.getType()) {
    case capnp::DynamicValue::STRUCT: {
      auto object = value.as<capnp::DynamicStruct>();
      for (auto field : object.getSchema().getFields()) {
        // has() alone omits default-valued scalars. Check only the union tag,
        // so zero/false samples are kept and inactive branches are never read.
        if (field.getProto().getDiscriminantValue() != 65535) {
          bool active = false;
          KJ_IF_MAYBE(selected, object.which()) { active = (*selected == field); }
          if (!active) continue;
        }
        extractValue(object.get(field), path + "/" + field.getProto().getName().cStr(), mono_time, series);
      }
      break;
    }
    case capnp::DynamicValue::LIST: {
      auto list = value.as<capnp::DynamicList>();
      for (unsigned int i = 0; i < list.size(); ++i) {
        extractValue(list[i], path + "/" + std::to_string(i), mono_time, series);
      }
      break;
    }
    case capnp::DynamicValue::BOOL:
      series[path].samples.push_back({mono_time, value.as<bool>() ? 1.0 : 0.0});
      break;
    case capnp::DynamicValue::INT:
      series[path].samples.push_back({mono_time, static_cast<double>(value.as<int64_t>())});
      break;
    case capnp::DynamicValue::UINT:
      series[path].samples.push_back({mono_time, static_cast<double>(value.as<uint64_t>())});
      break;
    case capnp::DynamicValue::FLOAT:
      series[path].samples.push_back({mono_time, value.as<double>()});
      break;
    case capnp::DynamicValue::ENUM: {
      auto enumeration = value.as<capnp::DynamicEnum>();
      auto &output = series[path];
      if (output.enum_names.empty()) {
        for (auto enumerant : enumeration.getSchema().getEnumerants()) {
          output.enum_names.emplace(enumerant.getOrdinal(), enumerant.getProto().getName().cStr());
        }
      }
      output.samples.push_back({mono_time, static_cast<double>(enumeration.getRaw())});
      break;
    }
    default:
      break;
  }
}

}  // namespace

void extractCerealEvent(cereal::Event::Reader event, CerealSeriesMap &series) {
  auto root = capnp::toDynamic(event);
  KJ_IF_MAYBE(field, root.which()) {
    std::string service = field->getProto().getName().cStr();
    if (service == "can" || service == "sendcan") return;
    extractValue(root.get(*field), "/" + service, event.getLogMonoTime(), series);
  }
}

void sortCerealSeries(CerealSeriesMap &series) {
  for (auto &[path, values] : series) {
    std::stable_sort(values.samples.begin(), values.samples.end(), [](const auto &a, const auto &b) {
      return a.mono_time < b.mono_time;
    });
  }
}

void CerealSeriesStore::replaceSegment(int number, CerealSeriesMap series) {
  sortCerealSeries(series);
  auto segment = std::make_shared<const CerealSeriesMap>(std::move(series));
  std::lock_guard lock(mutex_);
  segments_.insert_or_assign(number, std::move(segment));
  ++revision_;
}

void CerealSeriesStore::retainSegments(const std::vector<int> &numbers) {
  std::lock_guard lock(mutex_);
  bool changed = false;
  for (auto it = segments_.begin(); it != segments_.end();) {
    if (std::find(numbers.begin(), numbers.end(), it->first) == numbers.end()) {
      it = segments_.erase(it);
      changed = true;
    } else {
      ++it;
    }
  }
  if (changed) ++revision_;
}

CerealSeriesStore::Snapshot CerealSeriesStore::snapshot() const {
  std::lock_guard lock(mutex_);
  return {revision_, segments_};
}

void CerealSeriesStore::clear() {
  retainSegments({});
}

}  // namespace cabana
