#ifndef FRAMELOGGER_H
#define FRAMELOGGER_H

#include <cstdint>

#include <string>
#include <mutex>

class FrameLogger {
public:
  virtual ~FrameLogger() {}

  virtual void Open(const std::string &path) = 0;
  virtual void Close() = 0;

  int LogFrame(uint64_t ts, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, int *frame_segment) {
    std::lock_guard<std::recursive_mutex> guard(lock);
    
    if (opening) {
      Open(next_path);
      opening = false;
    }

    if (!is_open) return -1;

    if (rotating) {
      Close();
      Open(next_path);
      segment = next_segment;
      rotating = false;
    }

    int ret = ProcessFrame(ts, y_ptr, u_ptr, v_ptr);

    if (ret >= 0 && frame_segment) {
      *frame_segment = segment;
    }

    if (closing) {
      Close();
      closing = false;
    }

    return ret;
  }

  void Rotate(const std::string &new_path, int new_segment) {
    std::lock_guard<std::recursive_mutex> guard(lock);

    next_path = new_path;
    next_segment = new_segment;
    if (is_open) {
      if (next_segment == -1) {
        closing = true;
      } else {
        rotating = true;
      }
    } else {
      segment = next_segment;
      opening = true;
    }
  }

protected:

  virtual int ProcessFrame(uint64_t ts, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr) = 0;

  std::recursive_mutex lock;

  bool is_open = false;
  int segment = -1;

  std::string vid_path, lock_path;

private:
  int next_segment = -1;
  bool opening = false, closing = false, rotating = false;
  std::string next_path;
};

#endif
