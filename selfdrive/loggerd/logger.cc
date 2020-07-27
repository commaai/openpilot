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
  memset(s, 0, sizeof(*s));
  if (init_data) {
    s->init_data = (uint8_t*)malloc(init_data_len);
    assert(s->init_data);
    memcpy(s->init_data, init_data, init_data_len);
    s->init_data_len = init_data_len;
  }

  umask(0);

  pthread_mutex_init(&s->lock, NULL);

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
  int err;

  LoggerHandle *h = NULL;
  for (int i=0; i<LOGGER_MAX_HANDLES; i++) {
    if (s->handles[i].refcnt == 0) {
      h = &s->handles[i];
      break;
    }
  }
  assert(h);

  snprintf(h->segment_path, sizeof(h->segment_path),
          "%s/%s--%d", root_path, s->route_name, s->part);

  snprintf(h->log_path, sizeof(h->log_path), "%s/%s.bz2", h->segment_path, s->log_name);
  snprintf(h->qlog_path, sizeof(h->qlog_path), "%s/qlog.bz2", h->segment_path);
  snprintf(h->lock_path, sizeof(h->lock_path), "%s.lock", h->log_path);

  err = mkpath(h->log_path);
  if (err) return NULL;

  FILE* lock_file = fopen(h->lock_path, "wb");
  if (lock_file == NULL) return NULL;
  fclose(lock_file);

  h->log_file = fopen(h->log_path, "wb");
  if (h->log_file == NULL) goto fail;

  if (s->has_qlog) {
    h->qlog_file = fopen(h->qlog_path, "wb");
    if (h->qlog_file == NULL) goto fail;
  }

  int bzerror;
  h->bz_file = BZ2_bzWriteOpen(&bzerror, h->log_file, 9, 0, 30);
  if (bzerror != BZ_OK) goto fail;

  if (s->has_qlog) {
    h->bz_qlog = BZ2_bzWriteOpen(&bzerror, h->qlog_file, 9, 0, 30);
    if (bzerror != BZ_OK) goto fail;
  }

  if (s->init_data) {
    BZ2_bzWrite(&bzerror, h->bz_file, s->init_data, s->init_data_len);
    if (bzerror != BZ_OK) goto fail;

    if (s->has_qlog) {
      // init data goes in the qlog too
      BZ2_bzWrite(&bzerror, h->bz_qlog, s->init_data, s->init_data_len);
      if (bzerror != BZ_OK) goto fail;
    }
  }

  pthread_mutex_init(&h->lock, NULL);
  h->refcnt++;
  return h;
fail:
  LOGE("logger failed to open files");
  if (h->bz_file) {
    BZ2_bzWriteClose(&bzerror, h->bz_file, 0, NULL, NULL);
    h->bz_file = NULL;
  }
  if (h->bz_qlog) {
    BZ2_bzWriteClose(&bzerror, h->bz_qlog, 0, NULL, NULL);
    h->bz_qlog = NULL;
  }
  if (h->qlog_file) {
    fclose(h->qlog_file);
    h->qlog_file = NULL;
  }
  if (h->log_file) {
    fclose(h->log_file);
    h->log_file = NULL;
  }
  return NULL;
}

int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part) {
  bool is_start_of_route = !s->cur_handle;
  if (!is_start_of_route) log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_SEGMENT);

  pthread_mutex_lock(&s->lock);
  s->part++;

  LoggerHandle* next_h = logger_open(s, root_path);
  if (!next_h) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  if (s->cur_handle) {
    lh_close(s->cur_handle);
  }
  s->cur_handle = next_h;

  if (out_segment_path) {
    snprintf(out_segment_path, out_segment_path_len, "%s", next_h->segment_path);
  }
  if (out_part) {
    *out_part = s->part;
  }

  pthread_mutex_unlock(&s->lock);

  log_sentinel(s, is_start_of_route ? cereal::Sentinel::SentinelType::START_OF_ROUTE : cereal::Sentinel::SentinelType::START_OF_SEGMENT);
  return 0;
}

LoggerHandle* logger_get_handle(LoggerState *s) {
  pthread_mutex_lock(&s->lock);
  LoggerHandle* h = s->cur_handle;
  if (h) {
    pthread_mutex_lock(&h->lock);
    h->refcnt++;
    pthread_mutex_unlock(&h->lock);
  }
  pthread_mutex_unlock(&s->lock);
  return h;
}

void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&s->lock);
  if (s->cur_handle) {
    lh_log(s->cur_handle, data, data_size, in_qlog);
  }
  pthread_mutex_unlock(&s->lock);
}

void logger_close(LoggerState *s) {
  log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_ROUTE);

  pthread_mutex_lock(&s->lock);
  free(s->init_data);
  if (s->cur_handle) {
    lh_close(s->cur_handle);
  }
  pthread_mutex_unlock(&s->lock);
}

void lh_log(LoggerHandle* h, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  int bzerror;
  BZ2_bzWrite(&bzerror, h->bz_file, data, data_size);

  if (in_qlog && h->bz_qlog != NULL) {
    BZ2_bzWrite(&bzerror, h->bz_qlog, data, data_size);
  }
  pthread_mutex_unlock(&h->lock);
}

void lh_close(LoggerHandle* h) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  h->refcnt--;
  if (h->refcnt == 0) {
    if (h->bz_file){
      int bzerror;
      BZ2_bzWriteClose(&bzerror, h->bz_file, 0, NULL, NULL);
      h->bz_file = NULL;
    }
    if (h->bz_qlog){
      int bzerror;
      BZ2_bzWriteClose(&bzerror, h->bz_qlog, 0, NULL, NULL);
      h->bz_qlog = NULL;
    }
    if (h->qlog_file) {
      fclose(h->qlog_file);
      h->qlog_file = NULL;
    }
    fclose(h->log_file);
    h->log_file = NULL;
    unlink(h->lock_path);
    pthread_mutex_unlock(&h->lock);
    pthread_mutex_destroy(&h->lock);
    return;
  }
  pthread_mutex_unlock(&h->lock);
}
