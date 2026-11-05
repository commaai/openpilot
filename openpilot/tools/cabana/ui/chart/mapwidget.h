#pragma once
#include <memory>
#include <map>
#include <optional>
#include <string>
#include "tools/cabana/streams/cerealseries.h"

namespace cabana {
class MapWidget {
public:
  MapWidget();
  ~MapWidget();
  void setSeries(const std::map<std::string, CerealPoints> &series, uint64_t origin);
  std::optional<double> draw(double seconds, bool live);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace cabana
