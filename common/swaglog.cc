#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common/swaglog.h"

#include <cassert>
#include <mutex>
#include <string>

#include <zmq.h>
#include <stdarg.h>
#include "third_party/json11/json11.hpp"
#include "common/version.h"
#include "system/hardware/hw.h"

class SwaglogState {
public:
  SwaglogState() {
    zctx = zmq_ctx_new();
    sock = zmq_socket(zctx, ZMQ_PUSH);

    // Timeout on shutdown for messages to be received by the logging process
    int timeout = 100;
    zmq_setsockopt(sock, ZMQ_LINGER, &timeout, sizeof(timeout));
    zmq_connect(sock, Path::swaglog_ipc().c_str());

    print_level = CLOUDLOG_WARNING;
    if (const char* print_lvl = getenv("LOGPRINT")) {
      if (strcmp(print_lvl, "debug") == 0) {
        print_level = CLOUDLOG_DEBUG;
      } else if (strcmp(print_lvl, "info") == 0) {
        print_level = CLOUDLOG_INFO;
      } else if (strcmp(print_lvl, "warning") == 0) {
        print_level = CLOUDLOG_WARNING;
      }
    }

    ctx_j = json11::Json::object{};
    if (char* dongle_id = getenv("DONGLE_ID")) {
      ctx_j["dongle_id"] = dongle_id;
    }
    if (char* git_origin = getenv("GIT_ORIGIN")) {
      ctx_j["origin"] = git_origin;
    }
    if (char* git_branch = getenv("GIT_BRANCH")) {
      ctx_j["branch"] = git_branch;
    }
    if (char* git_commit = getenv("GIT_COMMIT")) {
      ctx_j["commit"] = git_commit;
    }
    if (char* daemon_name = getenv("MANAGER_DAEMON")) {
      ctx_j["daemon"] = daemon_name;
    }
    ctx_j["version"] = COMMA_VERSION;
    ctx_j["dirty"] = !getenv("CLEAN");
    ctx_j["device"] = Hardware::get_name();
  }

  ~SwaglogState() {
    zmq_close(sock);
    zmq_ctx_destroy(zctx);
  }

  void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string& log_s) {
    std::lock_guard lk(lock);
    if (levelnum >= print_level) {
      printf("%s: %s\n", filename, msg);
    }
    zmq_send(sock, log_s.data(), log_s.length(), ZMQ_NOBLOCK);
  }

  std::mutex lock;
  void* zctx = nullptr;
  void* sock = nullptr;
  int print_level;
  json11::Json::object ctx_j;
};

static bool LOG_TIMESTAMPS = getenv("LOG_TIMESTAMPS");

void cloudlog_e(int levelnum, bool timestamp, const char* filename, int lineno, const char* func, const char* fmt, ...) {
  static SwaglogState s;

  if (timestamp && !LOG_TIMESTAMPS) return;

  va_list args;
  va_start(args, fmt);
  char* msg_buf = nullptr;
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);

  if (ret <= 0 || !msg_buf) return;

  json11::Json::object log_j = json11::Json::object {
    {"ctx", s.ctx_j},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()}
  };

  if (timestamp) {
    log_j["msg"] = json11::Json::object{
        {"timestamp", json11::Json::object{
                          {"event", msg_buf},
                          {"time", std::to_string(nanos_since_boot())}}}};
  } else {
    log_j["msg"] = msg_buf;
  }

  std::string log_s(1, (char)levelnum);  // Prepend levelnum
  ((json11::Json)log_j).dump(log_s);
  s.log(levelnum, filename, lineno, func, msg_buf, log_s);

  free(msg_buf);
}
