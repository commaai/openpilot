#pragma once
#include <bzlib.h>

#include <memory>
#include <mutex>

#include "capnp/serialize.h"
#include "kj/array.h"

class LoggerHandle {
 public:
  LoggerHandle() = default;
  ~LoggerHandle();
  void log(uint8_t* data, size_t data_size, bool in_qlog = false);

 private:
  bool open(const char* segment_path, const char* log_name, bool has_qlog);
  std::mutex mutex_;
  std::string lock_path_;
  FILE* log_file_ = nullptr;
  FILE* qlog_file_ = nullptr;
  BZFILE* bz_file_ = nullptr;
  BZFILE* bz_qlog_ = nullptr;
  friend class Logger;
};

class Logger {
 public:
  Logger() = default;
  ~Logger() { close(); }
  void init(const char* log_name, bool has_qlog);
  bool next(const char* root_path);
  void close();
  std::shared_ptr<LoggerHandle> getHandle() { return cur_handle_; }
  inline int getSegment() const { return part_; }
  inline const char* getSegmentPath() const { return segment_path_; }
  inline void log(uint8_t* data, size_t data_size, bool in_qlog = false) {
    if (cur_handle_) cur_handle_->log(data, data_size, in_qlog);
  }
  inline void log(const kj::ArrayPtr<capnp::byte>& bytes, bool in_qlog = false) { log((uint8_t*)bytes.begin(), bytes.size(), in_qlog); }

 private:
  char segment_path_[4096] = {};
  char route_name_[64] = {};
  std::string log_name_;
  kj::Array<capnp::word> init_data_;
  int part_ = -1;
  bool has_qlog_ = false;
  std::shared_ptr<LoggerHandle> cur_handle_;
};
