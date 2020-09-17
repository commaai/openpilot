#pragma once
#include <assert.h>
#include <bzlib.h>

#include <memory>
#include <mutex>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/utilpp.h"
#include "messaging.hpp"

class LoggerHandle {
public:
  LoggerHandle() = default;
  ~LoggerHandle() { close(); }
  void write(uint8_t* data, size_t data_size, bool in_qlog = false);
  void write(MessageBuilder& msg, bool in_qlog = false);

private:
  bool open(const std::string& segment_path, const std::string& log_name, bool has_qlog);
  void close();
  std::mutex mutex;
  std::string lock_path;
  FILE* log_file = nullptr;
  BZFILE* bz_file = nullptr;
  BZFILE* bz_qlog = nullptr;
  FILE* qlog_file = nullptr;
  friend class Logger;
};

class Logger {
public:
  Logger(const char* root_path, const char* log_name, bool has_qlog);
  ~Logger(){};
  std::shared_ptr<LoggerHandle> openNext();
  inline std::shared_ptr<LoggerHandle> getHandle() const { return cur_handle; }
  inline int getPart() const { return part; }
  inline const char* getSegmentPath() const { return segment_path.c_str(); }

 private:
  std::string segment_path, log_name, root_path;
  char route_name[64] = {};
  int part = 0;
  bool has_qlog = false;
  kj::Array<capnp::word> init_data;
  std::shared_ptr<LoggerHandle> cur_handle;
};
