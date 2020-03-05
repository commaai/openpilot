#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <bzlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOGGER_MAX_HANDLES 16

typedef struct LoggerHandle {
  pthread_mutex_t lock;
  int refcnt;
  char segment_path[4096];
  char log_path[4096];
  char lock_path[4096];
  FILE* log_file;
  BZFILE* bz_file;

  FILE* qlog_file;
  char qlog_path[4096];
  BZFILE* bz_qlog;
} LoggerHandle;

typedef struct LoggerState {
  pthread_mutex_t lock;

  uint8_t* init_data;
  size_t init_data_len;

  int part;
  char route_name[64];
  char log_name[64];
  bool has_qlog;

  LoggerHandle handles[LOGGER_MAX_HANDLES];
  LoggerHandle* cur_handle;
} LoggerState;

void logger_init(LoggerState *s, const char* log_name, const uint8_t* init_data, size_t init_data_len, bool has_qlog);
int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part);
LoggerHandle* logger_get_handle(LoggerState *s);
void logger_close(LoggerState *s);
void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog);

void lh_log(LoggerHandle* h, uint8_t* data, size_t data_size, bool in_qlog);
void lh_close(LoggerHandle* h);

#ifdef __cplusplus
}
#endif

#endif
