#pragma once

#include "selfdrive/common/timing.h"

#define CLOUDLOG_TIMESTAMP 10
#define CLOUDLOG_DEBUG 10
#define CLOUDLOG_INFO 20
#define CLOUDLOG_WARNING 30
#define CLOUDLOG_ERROR 40
#define CLOUDLOG_CRITICAL 50



void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) /*__attribute__ ((format (printf, 6, 7)))*/;
void set_frame_id(uint64_t frame_id);

#define cloudlog(lvl, fmt, ...) cloudlog_e(lvl, __FILE__, __LINE__, \
                                           __func__, \
                                           fmt, ## __VA_ARGS__)

#define cloudlog_t_translate(lvl, event_name, translate,...)                                 \
{                                                                 \
  uint64_t ns = nanos_since_boot();                               \
  std::string json_msg = std::string("{'timestamp': {'event': '") + std::string(event_name) + std::string("','time':") + std::to_string(ns)+ std::string(",'translate':") + std::to_string(translate)+std::string("}}");   \
  cloudlog(lvl, json_msg.c_str(), ## __VA_ARGS__);                             \
}

#define cloudlog_t(lvl, event_name,...)                                 \
{                                                                 \
  uint64_t ns = nanos_since_boot();                               \
  std::string json_msg = std::string("{'timestamp': {'event': '") + std::string(event_name)+ std::string("','time':") + std::to_string(ns)+ std::string(",'translate':") + std::string("false")+std::string("}}");   \
  cloudlog(lvl, json_msg.c_str(), ## __VA_ARGS__);                             \
}

#define cloudlog_rl(burst, millis, lvl, fmt, ...)   \
{                                                   \
  static uint64_t __begin = 0;                      \
  static int __printed = 0;                         \
  static int __missed = 0;                          \
                                                    \
  int __burst = (burst);                            \
  int __millis = (millis);                          \
  uint64_t __ts = nanos_since_boot();               \
                                                    \
  if (!__begin) __begin = __ts;                     \
                                                    \
  if (__begin + __millis*1000000ULL < __ts) {       \
    if (__missed) {                                 \
      cloudlog(CLOUDLOG_WARNING, "cloudlog: %d messages suppressed", __missed); \
    }                                               \
    __begin = 0;                                    \
    __printed = 0;                                  \
    __missed = 0;                                   \
  }                                                 \
                                                    \
  if (__printed < __burst) {                        \
    cloudlog(lvl, fmt, ## __VA_ARGS__);             \
    __printed++;                                    \
  } else {                                          \
    __missed++;                                     \
  }                                                 \
}

#define LOGTT(event_name , translate, ...) cloudlog_t_translate(CLOUDLOG_TIMESTAMP, event_name, translate,## __VA_ARGS__)
#define LOGT(event_name , ...) cloudlog_t(CLOUDLOG_TIMESTAMP, event_name, ## __VA_ARGS__)
#define LOGD(fmt, ...) cloudlog(CLOUDLOG_DEBUG, fmt, ## __VA_ARGS__)
#define LOG(fmt, ...) cloudlog(CLOUDLOG_INFO, fmt, ## __VA_ARGS__)
#define LOGW(fmt, ...) cloudlog(CLOUDLOG_WARNING, fmt, ## __VA_ARGS__)
#define LOGE(fmt, ...) cloudlog(CLOUDLOG_ERROR, fmt, ## __VA_ARGS__)

#define LOGD_100(fmt, ...) cloudlog_rl(2, 100, CLOUDLOG_DEBUG, fmt, ## __VA_ARGS__)
#define LOG_100(fmt, ...) cloudlog_rl(2, 100, CLOUDLOG_INFO, fmt, ## __VA_ARGS__)
#define LOGW_100(fmt, ...) cloudlog_rl(2, 100, CLOUDLOG_WARNING, fmt, ## __VA_ARGS__)
#define LOGE_100(fmt, ...) cloudlog_rl(2, 100, CLOUDLOG_ERROR, fmt, ## __VA_ARGS__)
