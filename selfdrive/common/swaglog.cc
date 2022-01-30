#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "selfdrive/common/swaglog.h"

#include <cassert>
#include <cstring>
#include <mutex>
#include <string>

#include <zmq.h>
#include "json11.hpp"

#include "selfdrive/common/util.h"
#include "selfdrive/common/version.h"
#include "selfdrive/hardware/hw.h"

class SwaglogState : public LogState {
 public:
  SwaglogState() : LogState("ipc:///tmp/logmessage") {}

  bool initialized = false;
  json11::Json::object ctx_j;

  inline void initialize() {
    ctx_j = json11::Json::object {};
    print_level = CLOUDLOG_WARNING;
    const char* print_lvl = getenv("LOGPRINT");
    if (print_lvl) {
      if (strcmp(print_lvl, "debug") == 0) {
        print_level = CLOUDLOG_DEBUG;
      } else if (strcmp(print_lvl, "info") == 0) {
        print_level = CLOUDLOG_INFO;
      } else if (strcmp(print_lvl, "warning") == 0) {
        print_level = CLOUDLOG_WARNING;
      }
    }

    // openpilot bindings
    char* dongle_id = getenv("DONGLE_ID");
    if (dongle_id) {
      ctx_j["dongle_id"] = dongle_id;
    }
    char* daemon_name = getenv("MANAGER_DAEMON");
    if (daemon_name) {
      ctx_j["daemon"] = daemon_name;
    }
    ctx_j["version"] = COMMA_VERSION;
    ctx_j["dirty"] = !getenv("CLEAN");

    // device type
    if (Hardware::EON()) {
      ctx_j["device"] =  "eon";
    } else if (Hardware::TICI()) {
      ctx_j["device"] =  "tici";
    } else {
      ctx_j["device"] =  "pc";
    }

    initialized = true;
  }
};

static SwaglogState s = {};

static void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string& log_s) {
  if (levelnum >= s.print_level) {
    printf("%s: %s\n", filename, msg);
  }
  char levelnum_c = levelnum;
  zmq_send(s.sock, (levelnum_c + log_s).c_str(), log_s.length() + 1, ZMQ_NOBLOCK);
}

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  char* msg_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);

  if (ret <= 0 || !msg_buf) return;

  std::lock_guard lk(s.lock);

  if (!s.initialized) s.initialize();

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
  log(levelnum, filename, lineno, func, msg_buf, log_s);
  free(msg_buf);
}
