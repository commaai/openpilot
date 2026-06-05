#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "msgq/ipc.h"
#include "msgq/event.h"

template<typename TSubSocket>
class FakeSubSocket: public TSubSocket {
private:
  EventState *state = nullptr;
  int fds[2] = {-1, -1};

  void ensure_fifos_open() {
    for (size_t i = 0; i < 2; i++) {
      if (fds[i] < 0 && state->paths[i][0] != '\0') {
        fds[i] = open(state->paths[i], O_RDWR | O_NONBLOCK);
      }
    }
  }

public:
  FakeSubSocket(): TSubSocket() {}
  ~FakeSubSocket() {
    for (int fd : fds) {
      if (fd >= 0) close(fd);
    }
    if (state != nullptr) {
      munmap(state, sizeof(EventState));
    }
  }

  int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true, size_t segment_size=0) override {
    const char* cereal_prefix = std::getenv("CEREAL_FAKE_PREFIX");

    char* mem;
    std::string identifier = cereal_prefix != nullptr ?  std::string(cereal_prefix) : "";
    event_state_shm_mmap(endpoint, identifier, &mem, nullptr);

    this->state = (EventState*)mem;
    ensure_fifos_open();

    return TSubSocket::connect(context, endpoint, address, conflate, check_endpoint, segment_size);
  }

  Message *receive(bool non_blocking=false) override {
    if (this->state->enabled) {
      ensure_fifos_open();
      Event(fds[EventPurpose::RECV_CALLED]).set();
      Event(fds[EventPurpose::RECV_READY]).wait();
      Event(fds[EventPurpose::RECV_READY]).clear();
    }

    return TSubSocket::receive(non_blocking);
  }
};

class FakePoller: public Poller {
private:
  std::vector<SubSocket*> sockets;

public:
  void registerSocket(SubSocket *socket) override;
  std::vector<SubSocket*> poll(int timeout) override;
  ~FakePoller() {}
};
