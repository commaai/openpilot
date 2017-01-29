#ifndef SWAGLOG_H
#define SWAGLOG_H

#define CLOUDLOG_DEBUG 10
#define CLOUDLOG_INFO 20
#define CLOUDLOG_WARNING 30
#define CLOUDLOG_ERROR 40
#define CLOUDLOG_CRITICAL 50

#ifdef __cplusplus
extern "C" {
#endif

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func, const char* srctime, 
                const char* fmt, ...) /*__attribute__ ((format (printf, 6, 7)))*/;

void cloudlog_bind(const char* k, const char* v);

#ifdef __cplusplus
}
#endif

#define cloudlog(lvl, fmt, ...) cloudlog_e(lvl, __FILE__, __LINE__, \
                                           __func__, __DATE__ " " __TIME__, \
                                           fmt, ## __VA_ARGS__)

#define LOGD(fmt, ...) cloudlog(CLOUDLOG_DEBUG, fmt, ## __VA_ARGS__)
#define LOG(fmt, ...) cloudlog(CLOUDLOG_INFO, fmt, ## __VA_ARGS__)
#define LOGW(fmt, ...) cloudlog(CLOUDLOG_WARNING, fmt, ## __VA_ARGS__)
#define LOGE(fmt, ...) cloudlog(CLOUDLOG_ERROR, fmt, ## __VA_ARGS__)

#endif
