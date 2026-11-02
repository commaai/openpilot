#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct GpsTrace;
struct GeoBounds {
  double south = 0.0;
  double west = 0.0;
  double north = 0.0;
  double east = 0.0;

  bool valid() const {
    return south < north && west < east;
  }
};

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
  struct Request {
    std::string key;
    GeoBounds bounds;
    std::string query;
  };

  void run();

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
  std::unique_ptr<Request> pending_;
  std::unique_ptr<Request> active_;
  std::unique_ptr<RouteBasemap> completed_;
  std::unique_ptr<RouteBasemap> current_;
  std::thread worker_;
};
