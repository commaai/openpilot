#pragma once

#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <bzlib.h>
#include <mutex>

#define LOGGER_MAX_HANDLES 16

class LoggerState;
class LoggerHandle {
public:
  LoggerHandle() = default;
  void write(uint8_t* data, size_t data_size, bool in_qlog = false);
  bool open(LoggerState *s, const char *root_path);
  void close();
  std::mutex lock;
  int refcnt;
  char segment_path[4096];
  char lock_path[4096];
  FILE* log_file, *qlog_file;
  BZFILE* bz_file, *bz_qlog;
};

class LoggerState {
public:
  LoggerState() = default;
  std::mutex lock;

  uint8_t* init_data;
  size_t init_data_len;

  int part;
  char route_name[64];
  char log_name[64];
  bool has_qlog;

  LoggerHandle handles[LOGGER_MAX_HANDLES];
  LoggerHandle* cur_handle;
};

void logger_init(LoggerState *s, const char* log_name, const uint8_t* init_data, size_t init_data_len, bool has_qlog);
int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part);
LoggerHandle* logger_get_handle(LoggerState *s);
void logger_close(LoggerState *s);
void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog);
