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

#include "cereal/messaging/messaging.h"
#include "cereal/messaging/event.h"

template<typename TSubSocket>
class FakeSubSocket: public TSubSocket {
private:
  Event *recv_called = nullptr;
  Event *recv_ready = nullptr;
  EventState *state = nullptr;

public:
  FakeSubSocket(): TSubSocket() {}
  ~FakeSubSocket() {
    delete recv_called;
    delete recv_ready;
    if (state != nullptr) {
      munmap(state, sizeof(EventState));
    }
  }

  int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true) override {
    const char* cereal_prefix = std::getenv("CEREAL_FAKE_PREFIX");

    char* mem;
    std::string identifier = cereal_prefix != nullptr ?  std::string(cereal_prefix) : "";
    event_state_shm_mmap(endpoint, identifier, &mem, nullptr);

    this->state = (EventState*)mem;
    this->recv_called = new Event(state->fds[EventPurpose::RECV_CALLED]);
    this->recv_ready = new Event(state->fds[EventPurpose::RECV_READY]);

    return TSubSocket::connect(context, endpoint, address, conflate, check_endpoint);
  }

  Message *receive(bool non_blocking=false) override {
    if (this->state->enabled) {
      this->recv_called->set();
      this->recv_ready->wait();
      this->recv_ready->clear();
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
  ~FakePoller() {};
};
