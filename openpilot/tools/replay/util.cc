#include "tools/replay/util.h"

#include <cassert>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <mutex>
#include "common/timing.h"
#include "common/util.h"

ReplayMessageHandler message_handler = nullptr;
void installMessageHandler(ReplayMessageHandler handler) { message_handler = handler; }

void logMessage(ReplyMsgType type, const char *fmt, ...) {
  static std::mutex lock;
  std::lock_guard lk(lock);

  char *msg_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);
  if (ret <= 0 || !msg_buf) return;

  if (message_handler) {
    message_handler(type, msg_buf);
  } else {
    if (type == ReplyMsgType::Debug) {
      std::cout << "\033[38;5;248m" << msg_buf << "\033[00m" << std::endl;
    } else if (type == ReplyMsgType::Warning) {
      std::cout << "\033[38;5;227m" << msg_buf << "\033[00m" << std::endl;
    } else if (type == ReplyMsgType::Critical) {
      std::cout << "\033[38;5;196m" << msg_buf << "\033[00m" << std::endl;
    } else {
      std::cout << msg_buf << std::endl;
    }
  }

  free(msg_buf);
}

std::string formattedDataSize(size_t size) {
  if (size < 1024) {
    return std::to_string(size) + " B";
  } else if (size < 1024 * 1024) {
    return util::string_format("%.2f KB", (float)size / 1024);
  } else {
    return util::string_format("%.2f MB", (float)size / (1024 * 1024));
  }
}

std::string getUrlWithoutQuery(const std::string &url) {
  size_t idx = url.find("?");
  return (idx == std::string::npos ? url : url.substr(0, idx));
}

void precise_nano_sleep(int64_t nanoseconds, std::atomic<bool> &interrupt_requested) {
  struct timespec req, rem;
  req.tv_sec = nanoseconds / 1000000000;
  req.tv_nsec = nanoseconds % 1000000000;
  while (!interrupt_requested) {
#ifdef __APPLE__
    int ret = nanosleep(&req, &rem);
    if (ret == 0 || errno != EINTR)
      break;
#else
    int ret = clock_nanosleep(CLOCK_MONOTONIC, 0, &req, &rem);
    if (ret == 0 || ret != EINTR)
      break;
#endif
    // Retry sleep if interrupted by a signal
    req = rem;
  }
}

std::vector<std::string> split(std::string_view source, char delimiter) {
  std::vector<std::string> fields;
  size_t last = 0;
  for (size_t i = 0; i < source.length(); ++i) {
    if (source[i] == delimiter) {
      fields.emplace_back(source.substr(last, i - last));
      last = i + 1;
    }
  }
  fields.emplace_back(source.substr(last));
  return fields;
}

std::string extractFileName(const std::string &file) {
  size_t queryPos = file.find_first_of("?");
  std::string path = (queryPos != std::string::npos) ? file.substr(0, queryPos) : file;
  size_t lastSlash = path.find_last_of("/\\");
  return (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
}

// MonotonicBuffer

void *MonotonicBuffer::allocate(size_t bytes, size_t alignment) {
  assert(bytes > 0);
  void *p = std::align(alignment, bytes, current_buf, available);
  if (p == nullptr) {
    available = next_buffer_size = std::max(next_buffer_size, bytes);
    current_buf = buffers.emplace_back(std::aligned_alloc(alignment, next_buffer_size));
    next_buffer_size *= growth_factor;
    p = current_buf;
  }

  current_buf = (char *)current_buf + bytes;
  available -= bytes;
  return p;
}

MonotonicBuffer::~MonotonicBuffer() {
  for (auto buf : buffers) {
    free(buf);
  }
}
