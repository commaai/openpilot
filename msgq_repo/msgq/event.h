#pragma once

#include <string>
#include <vector>

#define CEREAL_EVENTS_PREFIX std::string("cereal_events")

void event_state_shm_mmap(std::string endpoint, std::string identifier, char **shm_mem, std::string *shm_path);

enum EventPurpose {
  RECV_CALLED,
  RECV_READY
};

struct EventState {
  int fds[2];
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
  std::string shm_path;
  EventState* state;
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
