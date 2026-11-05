#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <capnp/dynamic.h>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

namespace cabana {

struct CerealPoint {
  uint64_t mono_time;
  double value;
};

using CerealPoints = std::vector<CerealPoint>;
using CerealCancel = std::function<bool()>;
using CerealReader = std::function<CerealPoints(const std::string &, const CerealCancel &)>;

class CerealDiscovery {
public:
  CerealDiscovery();
  ~CerealDiscovery();
  void observe(cereal::Event::Reader event);
  std::vector<std::string> names() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::map<uint16_t, std::string> cerealEnumLabels(const std::string &path);

struct SeriesTransform {
  double scale = 1.0;
  double offset = 0.0;
  bool derivative = false;
  double derivative_dt = 0.0;
};

// Causal sample-and-hold: never read a future value.
std::optional<double> heldValue(const CerealPoints &points, uint64_t mono_time);
CerealPoints transformSeries(const CerealPoints &points, const SeriesTransform &transform);

// Paths use service/field/index, matching PlotJuggler's cereal signal names.
std::vector<std::string> cerealSignalPaths(cereal::Event::Reader event);
std::optional<double> cerealSignalValue(cereal::Event::Reader event, const std::string &path);

}  // namespace cabana
