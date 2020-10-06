#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include "common/visionipc.h"
// #include "logger.h"
#include "messaging.hpp"
class FrameLogger {
 public:
  FrameLogger(const std::string &afilename, int awidth, int aheight, int afps)
      : filename(afilename), width(awidth), height(aheight), fps(afps) {}

  virtual ~FrameLogger() {
    CloseFile();
  }

  int LogFrame(uint64_t cnt, void *addr, int in_width, int in_height, const VIPCBufExtra &extra) {
    int ret = -1;
    if (!is_open) return ret;

    uint8_t *y = (uint8_t *)addr;
    uint8_t *u = y + in_width * in_height;
    uint8_t *v = u + (in_width / 2) * (in_height / 2);

    if (ProcessFrame(cnt, y, u, v, in_width, in_height, extra)) {
      counter += 1;
      ret = counter;
    }
    return ret;
  }

  void Rotate(const std::string &new_path) {
    CloseFile();
    // create camera lock file
    lock_path = util::string_format("%s/%s.lock", new_path.c_str(), filename.c_str());
    int lock_fd = open(lock_path.c_str(), O_RDWR | O_CREAT, 0777);
    assert(lock_fd >= 0);
    close(lock_fd);

    Open(new_path);
  }

 protected:
  void CloseFile() {
    if (is_open) {
      Close();
      is_open = false;

      // delete camera lock file
      unlink(lock_path.c_str());
    }
  }

  virtual void Open(const std::string &path) = 0;
  virtual void Close() = 0;
  virtual bool ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                            int in_width, int in_height, const VIPCBufExtra &extra) = 0;

  int width, height, fps;

 private:
  int counter = 0;
  bool is_open = false;
  std::string filename, lock_path;
};
