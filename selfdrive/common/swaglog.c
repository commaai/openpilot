#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <pthread.h>
#include <zmq.h>
#include <json.h>

#include "common/timing.h"
#include "common/version.h"

#include "swaglog.h"

typedef struct LogState {
  pthread_mutex_t lock;
  bool inited;
  JsonNode *ctx_j;
  void *zctx;
  void *sock;
  int print_level;
} LogState;

static LogState s = {
  .lock = PTHREAD_MUTEX_INITIALIZER,
};

static void cloudlog_bind_locked(const char* k, const char* v) {
  json_append_member(s.ctx_j, k, json_mkstring(v));
}

static void cloudlog_init() {
  if (s.inited) return;
  s.ctx_j = json_mkobject();
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
  bool dirty = !getenv("CLEAN");
  json_append_member(s.ctx_j, "dirty", json_mkbool(dirty));

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

  JsonNode *log_j = json_mkobject();
  assert(log_j);

  json_append_member(log_j, "msg", json_mkstring(msg_buf));
  json_append_member(log_j, "ctx", s.ctx_j);
  json_append_member(log_j, "levelnum", json_mknumber(levelnum));
  json_append_member(log_j, "filename", json_mkstring(filename));
  json_append_member(log_j, "lineno", json_mknumber(lineno));
  json_append_member(log_j, "funcname", json_mkstring(func));
  json_append_member(log_j, "created", json_mknumber(seconds_since_epoch()));

  char* log_s = json_encode(log_j);
  assert(log_s);

  json_remove_from_parent(s.ctx_j);  

  json_delete(log_j);
  free(msg_buf);

  char levelnum_c = levelnum;
  zmq_send(s.sock, &levelnum_c, 1, ZMQ_NOBLOCK | ZMQ_SNDMORE);
  zmq_send(s.sock, log_s, strlen(log_s), ZMQ_NOBLOCK);
  free(log_s);

  pthread_mutex_unlock(&s.lock);
}

void cloudlog_bind(const char* k, const char* v) {
  pthread_mutex_lock(&s.lock);
  cloudlog_init();
  cloudlog_bind_locked(k, v);
  pthread_mutex_unlock(&s.lock);
}
