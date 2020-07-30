#pragma once
#include <bzlib.h>

#include <memory>
#include <mutex>
// #include <vector>
#include "capnp/serialize.h"
#include "kj/array.h"

class LoggerHandle {
 public:
  LoggerHandle() = default;
  ~LoggerHandle() { close();}
  void log(uint8_t* data, size_t data_size, bool in_qlog = false);

 private:
  bool open(const char* segment_path, const char* log_name, int part, bool has_qlog);
  void close();
  std::mutex mutex;
  char lock_path[4096] = {};
  FILE* log_file = nullptr;
  BZFILE* bz_file = nullptr;
  BZFILE* bz_qlog = nullptr;
  FILE* qlog_file = nullptr;
  friend class Logger;
};

class Logger {
 public:
  Logger() = default;
  ~Logger() {}
  void init(const char* log_name, bool has_qlog);
  bool open(const char* root_path);
  void close();
  std::shared_ptr<LoggerHandle> getHandle() { return cur_handle; }
  void log(uint8_t* data, size_t data_size, bool in_qlog = false) {
    if (cur_handle) {
      cur_handle->log(data, data_size, in_qlog);
    }
  }
  inline int getPart() { return part; }
  inline const char* getSegmentPath() const { return segment_path; }

 private:
  char segment_path[4096] = {};
  char route_name[64] = {};
  kj::Array<capnp::word> init_data_;
  int part = 0;

  std::string log_name_;
  bool has_qlog = false;
  std::shared_ptr<LoggerHandle> cur_handle;
};
