#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct GpsTrace;
struct RouteBasemap;
struct MapCacheStats {
  uint64_t bytes = 0;
  size_t files = 0;
};

class MapDataManager {
public:
  MapDataManager();
  ~MapDataManager();

  MapDataManager(const MapDataManager &) = delete;
  MapDataManager &operator=(const MapDataManager &) = delete;

  void pump();
  void ensureTrace(const GpsTrace &trace);
  void clearCache();
  bool loading() const;
  const RouteBasemap *current() const;
  MapCacheStats cacheStats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
