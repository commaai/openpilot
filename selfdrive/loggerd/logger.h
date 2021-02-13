#pragma once

#include <memory>
#include <mutex>
#include <bzlib.h>
#include "messaging.hpp"
#include "common/util.h"

#if defined(QCOM) || defined(QCOM2)
const std::string LOG_ROOT = "/data/media/0/realdata";
#else
const std::string LOG_ROOT = util::getenv_default("HOME", "/.comma/media/0/realdata", "/data/media/0/realdata");
#endif

class BZFile {
 public:
  BZFile(const std::string &path) {
    file = fopen(path.c_str(), "wb");
    assert(file != nullptr);
    int bzerror;
    bz_file = BZ2_bzWriteOpen(&bzerror, file, 9, 0, 30);
    assert(bzerror == BZ_OK);
  }
  ~BZFile() {
    int bzerror;
    BZ2_bzWriteClose(&bzerror, bz_file, 0, nullptr, nullptr);
    if (bzerror != BZ_OK) {
      LOGE("BZ2_bzWriteClose error, bzerror=%d", bzerror);
    }
    int err = fclose(file);
    assert(err == 0);
  }
  inline void write(void* data, size_t size) {
    int bzerror;
    BZ2_bzWrite(&bzerror, bz_file, data, size);
    if (bzerror != BZ_OK && !error_logged) {
      LOGE("BZ2_bzWrite error, bzerror=%d", bzerror);
      error_logged = true;
    }
  }
  inline void write(kj::ArrayPtr<capnp::byte> array) { write(array.begin(), array.size()); }

 private:
  bool error_logged = false;
  FILE* file = nullptr;
  BZFILE* bz_file = nullptr;
};

typedef cereal::Sentinel::SentinelType SentinelType;
class LoggerHandle {
 public:
  LoggerHandle(const std::string& route_path, int part, SentinelType type, kj::ArrayPtr<kj::byte> init_data);
  ~LoggerHandle();
  void write(uint8_t* data, size_t data_size, bool in_qlog);
  inline void write(kj::ArrayPtr<capnp::byte> array, bool in_qlog) { write(array.begin(), array.size(), in_qlog); }
  inline int get_segment() const { return part; }
  inline const std::string& get_segment_path() const { return segment_path; }
  inline void end_of_route() { end_sentinel_type = SentinelType::END_OF_ROUTE; }

 private:
  std::mutex lock;
  const int part;
  std::string segment_path, lock_path;
  std::unique_ptr<BZFile> log, qlog;
  SentinelType end_sentinel_type = SentinelType::END_OF_SEGMENT;
};

class LoggerState {
 public:
  LoggerState(const std::string& log_root);
  ~LoggerState();
  std::shared_ptr<LoggerHandle> next();
  inline std::shared_ptr<LoggerHandle> get_handle() {
    assert(cur_handle);
    return cur_handle;
  }

 private:
  int part = -1;
  std::string route_path;
  kj::Array<capnp::word> init_data;
  std::shared_ptr<LoggerHandle> cur_handle;
};

int logger_mkpath(char* file_path);
kj::Array<capnp::word> logger_build_init_data();
std::string logger_get_route_name();
