#pragma once

#include <cassert>
#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include <bzlib.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/hardware/hw.h"

const std::string LOG_ROOT = Path::log_root();

#define LOGGER_MAX_HANDLES 16

class BZFile {
 public:
  BZFile(const char* path) {
    file = util::safe_fopen(path, "wb");
    assert(file != nullptr);
  }
  ~BZFile() {
    util::safe_fflush(file);
    int err = fclose(file);
    assert(err == 0);
  }
  inline void write(void* data, size_t size) {
    util::safe_fwrite(data, 1, size, file);
  }
  inline void write(kj::ArrayPtr<capnp::byte> array) { write(array.begin(), array.size()); }

 private:
  FILE* file = nullptr;
};

typedef cereal::Sentinel::SentinelType SentinelType;

typedef struct LoggerHandle {
  pthread_mutex_t lock;
  SentinelType end_sentinel_type;
  int exit_signal;
  int refcnt;
  char segment_path[4096];
  char log_path[4096];
  char qlog_path[4096];
  char lock_path[4096];
  std::unique_ptr<BZFile> log, q_log;
} LoggerHandle;

typedef struct LoggerState {
  pthread_mutex_t lock;
  int part;
  kj::Array<capnp::word> init_data;
  std::string route_name;
  char log_name[64];
  bool has_qlog;

  LoggerHandle handles[LOGGER_MAX_HANDLES];
  LoggerHandle* cur_handle;
} LoggerState;

kj::Array<capnp::word> logger_build_init_data();
std::string logger_get_route_name();
void logger_init(LoggerState *s, const char* log_name, bool has_qlog);
int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part);
LoggerHandle* logger_get_handle(LoggerState *s);
void logger_close(LoggerState *s, ExitHandler *exit_handler=nullptr);
void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog);

void lh_log(LoggerHandle* h, uint8_t* data, size_t data_size, bool in_qlog);
void lh_close(LoggerHandle* h);
void clear_locks(const std::string log_root);
