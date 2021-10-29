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
    file = fopen(path, "wb");
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
