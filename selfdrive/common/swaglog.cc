#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <string>
#include <string.h>
#include <assert.h>
#include <mutex>
#include <zmq.h>

#include "json11.hpp"
#include "common/version.h"
#include "swaglog.h"

class LogState {
public:
  LogState() = default;
  ~LogState();
  void init();
  void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string &log_s);
  std::mutex lock;
  bool inited;
  json11::Json::object ctx_j;
  void *zctx;
  void *sock;
  int print_level;
};

void LogState::init() {
  zctx = zmq_ctx_new();
  sock = zmq_socket(zctx, ZMQ_PUSH);
  
  print_level = CLOUDLOG_WARNING;
  const char* log_print = getenv("LOGPRINT");
  if (log_print) {
    if (strcmp(log_print, "debug") == 0) {
      print_level = CLOUDLOG_DEBUG;
    } else if (strcmp(log_print, "info") == 0) {
      print_level = CLOUDLOG_INFO;
    } else if (strcmp(log_print, "warning") == 0) {
      print_level = CLOUDLOG_WARNING;
    }
  }

  // openpilot bindings
  ctx_j = json11::Json::object{};
  char* dongle_id = getenv("DONGLE_ID");
  if (dongle_id) {
    ctx_j["dongle_id"] = dongle_id;
  }
  ctx_j["version"] = COMMA_VERSION;
  ctx_j["dirty"] = !getenv("CLEAN");
  inited = true;
}

LogState::~LogState() {
  zmq_close(sock);
  zmq_ctx_destroy(zctx);
}

  // device type
  if (util::file_exists("/EON")) {
    cloudlog_bind_locked("device", "eon");
  } else if (util::file_exists("/TICI")) {
    cloudlog_bind_locked("device", "tici");
  } else {
    cloudlog_bind_locked("device", "pc");
  }

  s.inited = true;
void LogState::log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string &log_s) {
  std::lock_guard lk(lock);
  if (!inited) {
    init();
  }
  if (levelnum >= print_level) {
    printf("%s: %s\n", filename, msg);
  }
  char levelnum_c = levelnum;
  zmq_send(sock, &levelnum_c, 1, ZMQ_NOBLOCK | ZMQ_SNDMORE);
  zmq_send(sock, log_s.c_str(), log_s.length(), ZMQ_NOBLOCK);
}

static LogState log_state;

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  char* msg_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  vasprintf(&msg_buf, fmt, args);
  va_end(args);

  if (!msg_buf) return;

  json11::Json log_j = json11::Json::object {
    {"msg", msg_buf},
    {"ctx", log_state.ctx_j},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()}
  };
  std::string log_s = log_j.dump();
  log_state.log(levelnum, filename, lineno, func, msg_buf, log_s);
  free(msg_buf);
}
