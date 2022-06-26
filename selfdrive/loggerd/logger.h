#pragma once

#include <cassert>
#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include <capnp/serialize.h>
#include <kj/array.h>

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "common/swaglog.h"
#include "system/hardware/hw.h"

const std::string LOG_ROOT = Path::log_root();

#define LOGGER_MAX_HANDLES 16

class RawFile {
 public:
  RawFile(const std::string &path) {
    file = util::safe_fopen(path.c_str(), "wb");
    assert(file != nullptr);
  }
  ~RawFile() {
    util::safe_fflush(file);
    int err = fclose(file);
    assert(err == 0);
  }
  inline void write(void* data, size_t size) {
    int written = util::safe_fwrite(data, 1, size, file);
    assert(written == size);
  }
  inline void write(kj::ArrayPtr<capnp::byte> array) { write(array.begin(), array.size()); }

 private:
  FILE* file = nullptr;
};

typedef cereal::Sentinel::SentinelType SentinelType;

class LoggerState {
 public:
  LoggerState(const std::string& path);
  ~LoggerState();
  void write(uint8_t* data, size_t size, bool in_qlog);

 protected:
  std::string lock_file;
  std::unique_ptr<RawFile> log, qlog;
};

class Logger {
 public:
  Logger(const std::string& log_root = LOG_ROOT);
  bool next();
  void close(int signal);
  inline const std::string& routeName() const { return route_name; }
  inline int segment() const { return part; }
  inline const std::string& segmentPath() const { return segment_path; }
  inline void write(uint8_t* data, size_t size, bool in_qlog) { logger->write(data, size, in_qlog); }

 protected:
  int part = -1;
  std::string route_path, route_name, segment_path;
  kj::Array<capnp::word> init_data;
  std::unique_ptr<LoggerState> logger;
};

kj::Array<capnp::word> logger_build_init_data();
std::string logger_get_route_name();
