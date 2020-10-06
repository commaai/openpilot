#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <string>
#include <string.h>
#include <assert.h>

#include <pthread.h>
#include <zmq.h>

#include "json11.hpp"

#include "common/timing.h"
#include "common/version.h"

#include "swaglog.h"

typedef struct LogState {
  pthread_mutex_t lock;
  bool inited;
  json11::Json::object ctx_j;
  void *zctx;
  void *sock;
  int print_level;
} LogState;

static LogState s = {
  .lock = PTHREAD_MUTEX_INITIALIZER,
};

static void cloudlog_bind_locked(const char* k, const char* v) {
  s.ctx_j[k] = v;
}

static void cloudlog_init() {
  if (s.inited) return;
  s.ctx_j = json11::Json::object {};
  s.zctx = zmq_ctx_new();
  s.sock = zmq_socket(s.zctx, ZMQ_PUSH);
  zmq_connect(s.sock, "ipc:///tmp/logmessage");

  s.print_level = CLOUDLOG_WARNING;
  const char* print_level = getenv("LOGPRINT");
  if (print_level) {
    if (strcmp(print_level, "debug") == 0) {
      s.print_level = CLOUDLOG_DEBUG;
    } else if (strcmp(print_level, "info") == 0) {
      s.print_level = CLOUDLOG_INFO;
    } else if (strcmp(print_level, "warning") == 0) {
      s.print_level = CLOUDLOG_WARNING;
    }
  }

  // openpilot bindings
  char* dongle_id = getenv("DONGLE_ID");
  if (dongle_id) {
    cloudlog_bind_locked("dongle_id", dongle_id);
  }
  cloudlog_bind_locked("version", COMMA_VERSION);
  s.ctx_j["dirty"] = !getenv("CLEAN");

  s.inited = true;
}

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  pthread_mutex_lock(&s.lock);
  cloudlog_init();

  char* msg_buf = NULL;
  va_list args;
  va_start(args, fmt);
  vasprintf(&msg_buf, fmt, args);
  va_end(args);

  if (!msg_buf) {
    pthread_mutex_unlock(&s.lock);
    return;
  }

  if (levelnum >= s.print_level) {
    printf("%s: %s\n", filename, msg_buf);
  }

  json11::Json log_j = json11::Json::object {
    {"msg", msg_buf},
    {"ctx", s.ctx_j},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()}
  };

  std::string log_s = log_j.dump();

  free(msg_buf);

  char levelnum_c = levelnum;
  zmq_send(s.sock, &levelnum_c, 1, ZMQ_NOBLOCK | ZMQ_SNDMORE);
  zmq_send(s.sock, log_s.c_str(), log_s.length(), ZMQ_NOBLOCK);

  pthread_mutex_unlock(&s.lock);
}

void cloudlog_bind(const char* k, const char* v) {
  pthread_mutex_lock(&s.lock);
  cloudlog_init();
  cloudlog_bind_locked(k, v);
  pthread_mutex_unlock(&s.lock);
}
