#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "selfdrive/common/statlog.h"
#include "selfdrive/common/util.h"

#include <stdio.h>
#include <mutex>
#include <zmq.h>

class StatlogState : public LogState {
  public:
    StatlogState() : LogState("ipc:///tmp/stats") {}
};

static StatlogState s = {};

static void log(const char* metric_type, const char* metric, const char* fmt, ...) {
  char* value_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  int ret = vasprintf(&value_buf, fmt, args);
  va_end(args);

  if (ret > 0 && value_buf) {
    char* line_buf = nullptr;
    ret = asprintf(&line_buf, "%s:%s|%s", metric, value_buf, metric_type);
    if (ret > 0 && line_buf) {
      zmq_send(s.sock, line_buf, ret, ZMQ_NOBLOCK);
      free(line_buf);
    }
    free(value_buf);
  }
}

void statlog_log(const char* metric_type, const char* metric, int value) {
  log(metric_type, metric, "%d", value);
}

void statlog_log(const char* metric_type, const char* metric, float value) {
  log(metric_type, metric, "%f", value);
}
