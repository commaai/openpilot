#include "logger.hpp"

#include <assert.h>
#include <capnp/serialize.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "cereal/gen/cpp/log.capnp.h"
#include "common/swaglog.h"

static void log_sentinel(Logger* s, cereal::Sentinel::SentinelType type) {
  capnp::MallocMessageBuilder msg;
  auto event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto sen = event.initSentinel();
  sen.setType(type);
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();

  s->log(bytes.begin(), bytes.size(), true);
}

static int mkpath(char* file_path) {
  assert(file_path && *file_path);
  char* p;
  for (p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/')) {
    *p = '\0';
    if (mkdir(file_path, 0777) == -1) {
      if (errno != EEXIST) {
        *p = '/';
        return -1;
      }
    }
    *p = '/';
  }
  return 0;
}

void Logger::init(const char* log_name, const uint8_t* init_data, size_t init_data_len, bool has_qlog) {
  if (init_data) {
    init_data_.resize(init_data_len);
    memcpy(init_data_.data(), init_data, init_data_len);
  }

  umask(0);

  part = -1;
  has_qlog = has_qlog;

  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  log_name_ = log_name;
}

LoggerHandle* Logger::open(const char* root_path) {
  LoggerHandle* h = NULL;
  for (int i = 0; i < LOGGER_MAX_HANDLES; i++) {
    if (handles[i].refcnt == 0) {
      h = &handles[i];
      break;
    }
  }
  assert(h);
  snprintf(segment_path, sizeof(segment_path), "%s/%s--%d", root_path, route_name, part);
  h->open(segment_path, log_name_.c_str(), part, has_qlog);
  if (init_data_.size() > 0) {
    h->log(init_data_.data(), init_data_.size(), has_qlog);
  }
  return h;
}

void Logger::close() {
  log_sentinel(this, cereal::Sentinel::SentinelType::END_OF_ROUTE);

  std::lock_guard<std::mutex> guard(mutex);
  if (cur_handle) {
    cur_handle->close();
  }
}

int Logger::next(const char* root_path) {
  bool is_start_of_route = !cur_handle;
  if (!is_start_of_route) log_sentinel(this, cereal::Sentinel::SentinelType::END_OF_SEGMENT);

  {
    std::lock_guard<std::mutex> guard(mutex);
    part++;
    LoggerHandle* next_h = open(root_path);
    if (!next_h) return -1;

    if (cur_handle) {
      cur_handle->close();
    }
    cur_handle = next_h;
  }
  log_sentinel(this, is_start_of_route ? cereal::Sentinel::SentinelType::START_OF_ROUTE : cereal::Sentinel::SentinelType::START_OF_SEGMENT);
  return 0;
}

LoggerHandle* Logger::getHandle() {
  const std::lock_guard<std::mutex> lock(mutex);
  if (cur_handle) {
    cur_handle->mutex.lock();
    cur_handle->refcnt++;
    cur_handle->mutex.unlock();
  }
  return cur_handle;
}

void Logger::log(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex);
  if (cur_handle) {
    cur_handle->log(data, data_size, in_qlog);
  }
}

bool LoggerHandle::open(const char* segment_path, const char* log_name, int part, bool has_qlog) {
  char log_path[4096] = {};

  char qlog_path[4096] = {};

  snprintf(log_path, sizeof(log_path), "%s/%s.bz2", segment_path, log_name);
  snprintf(qlog_path, sizeof(qlog_path), "%s/qlog.bz2", segment_path);
  snprintf(lock_path, sizeof(lock_path), "%s.lock", log_path);

  int err = mkpath(log_path);
  if (err) return false;

  FILE* lock_file = fopen(lock_path, "wb");
  if (lock_file == NULL) return false;
  fclose(lock_file);

  int bzerror;

  log_file = fopen(log_path, "wb");
  if (log_file == NULL) goto fail;
  bz_file = BZ2_bzWriteOpen(&bzerror, log_file, 9, 0, 30);
  if (bzerror != BZ_OK) goto fail;

  if (has_qlog) {
    qlog_file = fopen(qlog_path, "wb");
    if (qlog_file == NULL) goto fail;
    bz_qlog = BZ2_bzWriteOpen(&bzerror, qlog_file, 9, 0, 30);
    if (bzerror != BZ_OK) goto fail;
  }
  refcnt++;
  return true;
fail:
  LOGE("logger failed to open files");
  close();
  return false;
}

void LoggerHandle::log(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex);
  assert(refcnt > 0);
  int bzerror;
  BZ2_bzWrite(&bzerror, bz_file, data, data_size);

  if (in_qlog && bz_qlog != NULL) {
    BZ2_bzWrite(&bzerror, bz_qlog, data, data_size);
  }
}

void LoggerHandle::close() {
  const std::lock_guard<std::mutex> lock(mutex);
  assert(refcnt > 0);
  refcnt--;
  int bzerror;
  if (refcnt == 0) {
    if (bz_file) {
      BZ2_bzWriteClose(&bzerror, bz_file, 0, NULL, NULL);
      bz_file = NULL;
    }
    if (bz_qlog) {
      BZ2_bzWriteClose(&bzerror, bz_qlog, 0, NULL, NULL);
      bz_qlog = NULL;
    }
    if (qlog_file) {
      fclose(qlog_file);
      qlog_file = NULL;
    }
    fclose(log_file);
    log_file = NULL;
    unlink(lock_path);
  }
}
