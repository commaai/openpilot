#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <errno.h>

#include <unistd.h>
#include <sys/stat.h>

#include <pthread.h>
#include <bzlib.h>
#include "messaging.hpp"

#include "common/swaglog.h"
#include "common/util.h"


#include "logger.h"

static void log_sentinel(LoggerState *s, cereal::Sentinel::SentinelType type) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  auto bytes = msg.toBytes();

  logger_log(s, bytes.begin(), bytes.size(), true);
}

static int mkpath(char* file_path) {
  assert(file_path && *file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p = '\0';
    if (mkdir(file_path, 0777)==-1) {
      if (errno != EEXIST) {
        *p = '/';
        return -1;
      }
    }
    *p = '/';
  }
  return 0;
}

void logger_init(LoggerState *s, const char* log_name, const uint8_t* init_data, size_t init_data_len, bool has_qlog) {
  if (init_data) {
    s->init_data = (uint8_t*)malloc(init_data_len);
    assert(s->init_data);
    memcpy(s->init_data, init_data, init_data_len);
    s->init_data_len = init_data_len;
  }

  umask(0);

  s->part = -1;
  s->has_qlog = has_qlog;

  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);

  strftime(s->route_name, sizeof(s->route_name),
           "%Y-%m-%d--%H-%M-%S", &timeinfo);
  snprintf(s->log_name, sizeof(s->log_name), "%s", log_name);
}

static LoggerHandle* logger_open(LoggerState *s, const char* root_path) {
  LoggerHandle *h = NULL;
  for (int i=0; i<LOGGER_MAX_HANDLES; i++) {
    if (s->handles[i].refcnt == 0) {
      h = &s->handles[i];
      break;
    }
  }
  assert(h);
  return h->open(s, root_path) ? h : nullptr;
}

int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part) {
  bool is_start_of_route = !s->cur_handle;
  if (!is_start_of_route) log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_SEGMENT);

  std::unique_lock lk(s->lock);
  s->part++;

  LoggerHandle* next_h = logger_open(s, root_path);
  if (!next_h) {
    return -1;
  }

  if (s->cur_handle) {
    s->cur_handle->close();
  }
  s->cur_handle = next_h;

  if (out_segment_path) {
    snprintf(out_segment_path, out_segment_path_len, "%s", next_h->segment_path);
  }
  if (out_part) {
    *out_part = s->part;
  }

  lk.unlock();

  log_sentinel(s, is_start_of_route ? cereal::Sentinel::SentinelType::START_OF_ROUTE : cereal::Sentinel::SentinelType::START_OF_SEGMENT);
  return 0;
}

LoggerHandle* logger_get_handle(LoggerState *s) {
  std::unique_lock logger_lk(s->lock);
  LoggerHandle* h = s->cur_handle;
  if (h) {
    std::unique_lock lk(h->lock);
    h->refcnt++;
  }
  return h;
}

void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog) {
  std::unique_lock lk(s->lock);
  if (s->cur_handle) {
    s->cur_handle->write(data, data_size, in_qlog);
  }
}

void logger_close(LoggerState *s) {
  log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_ROUTE);

  std::unique_lock lk(s->lock);
  free(s->init_data);
  if (s->cur_handle) {
    s->cur_handle->close();
  }
}

bool LoggerHandle::open(LoggerState *s, const char *root_path) {
  snprintf(segment_path, sizeof(segment_path), "%s/%s--%d", root_path, s->route_name, s->part);

  std::string log_path = util::string_format("%s/%s.bz2", segment_path, s->log_name);
  std::string qlog_path = util::string_format("%s/qlog.bz2", segment_path);
  snprintf(lock_path, sizeof(lock_path), "%s.lock", log_path.c_str());

  if (0 != mkpath((char*)log_path.c_str())) return false;

  FILE *lock_file = fopen(lock_path, "wb");
  if (lock_file == nullptr) return false;
  fclose(lock_file);

  ++refcnt;

  auto open_files = [](const std::string &f_path, FILE*& f, BZFILE*& bz_f) {
    f = fopen(f_path.c_str(), "wb");
    if (f != nullptr) {
      int bzerror;
      bz_f = BZ2_bzWriteOpen(&bzerror, f, 9, 0, 30);
      return bzerror == BZ_OK;
    }
    return false;
  };

  if (!open_files(log_path, log_file, bz_file) ||
      (s->has_qlog && !open_files(qlog_path, qlog_file, bz_qlog))) {
    close();
    return false;
  }
  return true;
}

void LoggerHandle::write(uint8_t* data, size_t data_size, bool in_qlog) {
  std::unique_lock lk(lock);
  assert(refcnt > 0);
  int bzerror;
  BZ2_bzWrite(&bzerror, bz_file, data, data_size);
  if (in_qlog && bz_qlog != NULL) {
    BZ2_bzWrite(&bzerror, bz_qlog, data, data_size);
  }
}

void LoggerHandle::close() {
  std::unique_lock lk(lock);
  assert(refcnt > 0);
  if (--refcnt == 0) {
    auto close_files = [](FILE*& f, BZFILE*& bz_f) {
      if (bz_f) {
        int bzerror;
        BZ2_bzWriteClose(&bzerror, bz_f, 0, NULL, NULL);
        bz_f = nullptr;
      }
      if (f) {
        fclose(f);
        f = nullptr;
      }
    };
    close_files(qlog_file, bz_qlog);
    close_files(log_file, bz_file);
    unlink(lock_path);
  }
}
