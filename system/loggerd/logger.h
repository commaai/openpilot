#pragma once

#include <cassert>
#include <memory>
#include <string>

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "system/hardware/hw.h"
#include "system/loggerd/zstd_writer.h"


// 2 is ideal for real-time logging to Zstd files, providing a good balance between speed and compression ratio.
constexpr int LOG_COMPRESSION_LEVEL = 2;

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
  LoggerState(const std::string& log_root = Path::log_root());
  ~LoggerState();
  bool next();
  void write(uint8_t* data, size_t size, bool in_qlog);
  inline int segment() const { return part; }
  inline const std::string& segmentPath() const { return segment_path; }
  inline const std::string& routeName() const { return route_name; }
  inline void write(kj::ArrayPtr<kj::byte> bytes, bool in_qlog) { write(bytes.begin(), bytes.size(), in_qlog); }
  inline void setExitSignal(int signal) { exit_signal = signal; }

protected:
  int part = -1, exit_signal = 0;
  std::string route_path, route_name, segment_path, lock_file;
  kj::Array<capnp::word> init_data;
  std::unique_ptr<ZstdFileWriter> rlog, qlog;
};

kj::Array<capnp::word> logger_build_init_data();
std::string logger_get_identifier(std::string key);
std::string zstd_decompress(const std::string &in);

