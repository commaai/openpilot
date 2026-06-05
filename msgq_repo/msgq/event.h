#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#define CEREAL_EVENTS_PREFIX std::string("cereal_events")

constexpr size_t EVENT_PATH_MAX = 128;

void event_state_shm_mmap(std::string endpoint, std::string identifier, char **shm_mem, std::string *shm_name);

enum EventPurpose {
  RECV_CALLED,
  RECV_READY
};

struct EventState {
  char paths[2][EVENT_PATH_MAX];
  bool enabled;
};

class Event {
private:
  int event_fd = -1;

  inline void throw_if_invalid() const {
    if (!this->is_valid()) {
      throw std::runtime_error("Event does not have valid file descriptor.");
    }
  }
public:
  Event(int fd = -1);

  void set() const;
  int clear() const;
  void wait(int timeout_sec = -1) const;
  bool peek() const;
  bool is_valid() const;
  int fd() const;

  static int wait_for_one(const std::vector<Event>& events, int timeout_sec = -1);
};

class SocketEventHandle {
private:
  std::string shm_name;
  EventState* state = nullptr;
  bool owns_fifos = false;
  int fds[2] = {-1, -1};
public:
  SocketEventHandle(std::string endpoint, std::string identifier = "", bool override = true);
  ~SocketEventHandle();

  bool is_enabled();
  void set_enabled(bool enabled);
  Event recv_called();
  Event recv_ready();

  static void toggle_fake_events(bool enabled);
  static void set_fake_prefix(std::string prefix);
  static std::string fake_prefix();
};
