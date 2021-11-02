#pragma once

#include <cassert>
#include <memory>
#include <mutex>
#include <bzlib.h>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/hardware/hw.h"

const std::string LOG_ROOT = Path::log_root();

#define LOGGER_MAX_HANDLES 16

class BZFile {
 public:
  BZFile(const std::string &path) {
    file = util::safe_fopen(path.c_str(), "wb");
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
    util::safe_fflush(file);
    int err = fclose(file);
    assert(err == 0);
  }
  inline void write(void* data, size_t size) {
    int bzerror;
    do {
      BZ2_bzWrite(&bzerror, bz_file, data, size);
    } while (bzerror == BZ_IO_ERROR && errno == EINTR);

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

class Logger {
public:
  Logger(const std::string& route_path, int part, kj::ArrayPtr<kj::byte> init_data);
  ~Logger();
  void end_of_route(int signal);
  void write(uint8_t* data, size_t data_size, bool in_qlog);
  inline void write(kj::ArrayPtr<capnp::byte> array, bool in_qlog) { write(array.begin(), array.size(), in_qlog); }
  inline int segment() const { return part_; }
  inline const std::string& segmentPath() const { return segment_path_; }

protected:
  std::mutex lock_;
  int part_ = -1, signal_ = 0;
  std::string segment_path_, lock_path_;
  std::unique_ptr<BZFile> log_, qlog_;
  SentinelType end_sentinel_type_ = SentinelType::END_OF_SEGMENT;
};

class LoggerManager {
public:
  LoggerManager(const std::string& log_root);
  std::shared_ptr<Logger> next();
  inline const std::string& routePath() const { return route_path_; }
  inline const std::string& routeName() const { return route_name_; }

protected:
  int part_ = -1;
  std::string route_path_, route_name_;
  kj::Array<capnp::word> init_data_;
};

kj::Array<capnp::word> logger_build_init_data();
std::string logger_get_route_name();
