#include "selfdrive/common/swaglog.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

#include "json11.hpp"
#include <zmq.h>

#include "selfdrive/common/util.h"
#include "selfdrive/common/version.h"
#include "selfdrive/hardware/hw.h"

class LogState {
public:
  LogState();
  ~LogState();
  void bind(const char* k, const char* v);
  void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg);

protected:
  std::mutex lock_;
  json11::Json::object ctx_j_;
  void *zctx_, *sock_;
  int print_level_ = CLOUDLOG_WARNING;
};

LogState::~LogState() {
  zmq_close(sock_);
  zmq_ctx_destroy(zctx_);
}

void LogState::bind(const char* k, const char* v) {
  ctx_j_[k] = v;
}

LogState::LogState() {
  zctx_ = zmq_ctx_new();

  sock_ = zmq_socket(zctx_, ZMQ_PUSH);
  int timeout = 100;  // 100 ms timeout on shutdown for messages to be received by logmessaged
  zmq_setsockopt(sock_, ZMQ_LINGER, &timeout, sizeof(timeout));
  zmq_connect(sock_, "ipc:///tmp/logmessage");

  std::string level = util::getenv("LOGPRINT", "warning");
  if (level == "debug") {
    print_level_ = CLOUDLOG_DEBUG;
  } else if (level == "info") {
    print_level_ = CLOUDLOG_INFO;
  } else if (level == "warning") {
    print_level_ = CLOUDLOG_WARNING;
  }

  // openpilot bindings
  ctx_j_["dongle_id"] = util::getenv("DONGLE_ID");
  ctx_j_["version"] = COMMA_VERSION;
  ctx_j_["dirty"] = !getenv("CLEAN");
  if (Hardware::EON()) {
    ctx_j_["device"] = "eon";
  } else if (Hardware::TICI()) {
    ctx_j_["device"] = "tici";
  } else {
    ctx_j_["device"] = "pc";
  }
}

void LogState::log(int levelnum, const char* filename, int lineno, const char* func, const char* msg) {
  std::lock_guard lk(lock_);
  if (levelnum >= print_level_) {
    std::cout << filename << ": " << msg << std::endl;
  }
  json11::Json log_j = json11::Json::object{
    {"msg", msg},
    {"ctx", ctx_j_},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()}};
  std::string log_s = (char)levelnum + log_j.dump();
  zmq_send(sock_, log_s.c_str(), log_s.length(), ZMQ_NOBLOCK);
}

LogState* logInstance() {
  // local static variable initialization is thread-safe after c++11
  static LogState s;
  return &s;
}

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func, const char* fmt, ...) {
  char* msg_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  vasprintf(&msg_buf, fmt, args);
  va_end(args);

  if (msg_buf) {
    logInstance()->log(levelnum, filename, lineno, func, msg_buf);
    free(msg_buf);
  }
}

void cloudlog_bind(const char* k, const char* v) {
  logInstance()->bind(k, v);
}
