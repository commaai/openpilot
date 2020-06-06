#pragma once
#include <bzlib.h>
#include <mutex>
#include <vector>
#include "kj/array.h"
#include "capnp/serialize.h"

#define LOGGER_MAX_HANDLES 16

class LoggerHandle {
 public:
  LoggerHandle() {
  }
  ~LoggerHandle() {
  }
  bool open(const char* segment_path, const char* log_name, int part, bool has_qlog);
  void log(uint8_t* data, size_t data_size, bool in_qlog);
  void close();
  std::mutex mutex;
  int refcnt = 0;
  char lock_path[4096] = {};
  FILE* log_file = nullptr;
  BZFILE* bz_file = nullptr;
  BZFILE* bz_qlog = nullptr;
  FILE* qlog_file = nullptr;
};

class Logger {
 public:
  Logger() {
  }
  ~Logger() {
  }

  void init(const char* log_name, bool has_qlog);
  LoggerHandle* open(const char* root_path);
  void close();
  int next(const char* root_path);
  LoggerHandle* getHandle();
  void log(uint8_t* data, size_t data_size, bool in_qlog);
  inline int getPart() { return part; }
  inline const char* getSegmentPath() const { return segment_path; }

 private:
  char segment_path[4096] = {};
  char route_name[64] = {};
  std::mutex mutex;
  kj::Array<capnp::word> init_data_;
  int part = 0;
  
  std::string log_name_;
  bool has_qlog = false;

  LoggerHandle handles[LOGGER_MAX_HANDLES] = {};
  LoggerHandle* cur_handle = nullptr;
};
