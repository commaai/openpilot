#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common/swaglog.h"

#include <cassert>
#include <limits>
#include <mutex>
#include <string>

#include <clocale>
#include <cstring>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "third_party/json11/json11.hpp"
#include "common/version.h"
#include "system/hardware/hw.h"

class SwaglogState {
public:
  SwaglogState() {
    // Create Unix datagram socket
    sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (sock_fd >= 0) {
      // Set non-blocking
      int flags = fcntl(sock_fd, F_GETFL, 0);
      fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);

      // Set up destination address
      memset(&sock_addr, 0, sizeof(sock_addr));
      sock_addr.sun_family = AF_UNIX;
      std::string ipc_path = Path::swaglog_ipc();
      // Convert "ipc:///path" to "/path"
      if (ipc_path.rfind("ipc://", 0) == 0) {
        ipc_path = ipc_path.substr(6);
      }
      strncpy(sock_addr.sun_path, ipc_path.c_str(), sizeof(sock_addr.sun_path) - 1);
    }

    // workaround for https://github.com/dropbox/json11/issues/38
    setlocale(LC_NUMERIC, "C");

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
    if (sock_fd >= 0) {
      close(sock_fd);
    }
  }

  void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string& log_s) {
    std::lock_guard lk(lock);
    if (levelnum >= print_level) {
      printf("%s: %s\n", filename, msg);
    }
    if (sock_fd >= 0) {
      // Non-blocking sendto, silently drops on failure (matches ZMQ NOBLOCK behavior)
      sendto(sock_fd, log_s.data(), log_s.length(), 0,
             (struct sockaddr*)&sock_addr, sizeof(sock_addr));
    }
  }

  std::mutex lock;
  int sock_fd = -1;
  struct sockaddr_un sock_addr;
  int print_level;
  json11::Json::object ctx_j;
};

bool LOG_TIMESTAMPS = getenv("LOG_TIMESTAMPS");
uint32_t NO_FRAME_ID = std::numeric_limits<uint32_t>::max();

static void cloudlog_common(int levelnum, const char* filename, int lineno, const char* func,
                            char* msg_buf, const json11::Json::object &msg_j={}) {
  static SwaglogState s;

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

  std::string log_s;
  log_s += (char)levelnum;
  ((json11::Json)log_j).dump(log_s);
  s.log(levelnum, filename, lineno, func, msg_buf, log_s);

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
