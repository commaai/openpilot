#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common/swaglog.h"

#include <cassert>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#include <zmq.h>
#include "json11.hpp"

#include "common/util.h"
#include "common/version.h"
#include "selfdrive/hardware/hw.h"

class SwaglogState : public LogState {
 public:
  SwaglogState() : LogState("ipc:///tmp/logmessage") {}

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
    ctx_j["device"] = Hardware::get_name();
    LogState::initialize();
  }
};

static SwaglogState s = {};
bool LOG_TIMESTAMPS = getenv("LOG_TIMESTAMPS");
uint32_t NO_FRAME_ID = std::numeric_limits<uint32_t>::max();

static void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string& log_s) {
  if (levelnum >= s.print_level) {
    printf("%s: %s\n", filename, msg);
  }
  char levelnum_c = levelnum;
  zmq_send(s.sock, (levelnum_c + log_s).c_str(), log_s.length() + 1, ZMQ_NOBLOCK);
}
static void cloudlog_common(int levelnum, const char* filename, int lineno, const char* func,
                            char* msg_buf, json11::Json::object msg_j={}) {
  std::lock_guard lk(s.lock);
  if (!s.initialized) s.initialize();

  json11::Json::object log_j = json11::Json::object {
    {"ctx", s.ctx_j},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()}
  };
  if (msg_j.empty()) {
    log_j["msg"] = msg_buf;
  } else {
    log_j["msg"] = msg_j;
  }

  std::string log_s = ((json11::Json)log_j).dump();
  log(levelnum, filename, lineno, func, msg_buf, log_s);
  free(msg_buf);
}

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char* msg_buf = nullptr;
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);
  if (ret <= 0 || !msg_buf) return;
  cloudlog_common(levelnum, filename, lineno, func, msg_buf);
}

void cloudlog_t_common(int levelnum, const char* filename, int lineno, const char* func,
                       uint32_t frame_id, const char* fmt, va_list args) {
  if (!LOG_TIMESTAMPS) return;
  char* msg_buf = nullptr;
  int ret = vasprintf(&msg_buf, fmt, args);
  if (ret <= 0 || !msg_buf) return;
  json11::Json::object tspt_j = json11::Json::object{
    {"event", msg_buf},
    {"time", std::to_string(nanos_since_boot())}
  };
  if (frame_id < NO_FRAME_ID) {
    tspt_j["frame_id"] = std::to_string(frame_id);
  }
  tspt_j = json11::Json::object{{"timestamp", tspt_j}};
  cloudlog_common(levelnum, filename, lineno, func, msg_buf, tspt_j);
}


void cloudlog_te(int levelnum, const char* filename, int lineno, const char* func,
                 const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  cloudlog_t_common(levelnum, filename, lineno, func, NO_FRAME_ID, fmt, args);
  va_end(args);
}
void cloudlog_te(int levelnum, const char* filename, int lineno, const char* func,
                 uint32_t frame_id, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  cloudlog_t_common(levelnum, filename, lineno, func, frame_id, fmt, args);
  va_end(args);
}

