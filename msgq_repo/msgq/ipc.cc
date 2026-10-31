#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "msgq/ipc.h"
#include "msgq/event.h"

static bool messaging_use_fake(){
  return std::getenv("CEREAL_FAKE") != NULL;
}

void Message::init(size_t sz) {
  size = sz;
  data = new char[size];
}

void Message::init(char * d, size_t sz) {
  size = sz;
  data = new char[size];
  memcpy(data, d, size);
}

void Message::takeOwnership(char * d, size_t sz) {
  size = sz;
  data = d;
}

void Message::close() {
  if (size > 0){
    delete[] data;
  }
  size = 0;
}

Message::~Message() {
  this->close();
}

void SubSocket::open_fake_fifos() {
  for (size_t i = 0; i < 2; i++) {
    if (fake_fds[i] < 0 && fake_state->paths[i][0] != '\0') {
      fake_fds[i] = open(fake_state->paths[i], O_RDWR | O_NONBLOCK);
    }
  }
}

int SubSocket::connect(std::string endpoint, bool conflate, size_t segment_size){
  if (messaging_use_fake()) {
    const char* cereal_prefix = std::getenv("CEREAL_FAKE_PREFIX");
    std::string identifier = cereal_prefix != nullptr ? std::string(cereal_prefix) : "";

    char* mem;
    event_state_shm_mmap(endpoint, identifier, &mem, nullptr);
    fake_state = (EventState*)mem;
    open_fake_fifos();
  }

  q = new msgq_queue_t;
  size_t size = segment_size > 0 ? segment_size : DEFAULT_SEGMENT_SIZE;
  int r = msgq_new_queue(q, endpoint.c_str(), size);
  if (r != 0){
    return r;
  }

  msgq_init_subscriber(q);

  if (conflate){
    q->read_conflate = true;
  }

  return 0;
}


Message * SubSocket::receive(bool non_blocking){
  if (fake_state != nullptr && fake_state->enabled) {
    open_fake_fifos();
    Event(fake_fds[EventPurpose::RECV_CALLED]).set();
    Event(fake_fds[EventPurpose::RECV_READY]).wait();
    Event(fake_fds[EventPurpose::RECV_READY]).clear();
  }

  msgq_msg_t msg;

  Message *r = NULL;

  int rc = msgq_msg_recv(&msg, q);

  // Hack to implement blocking read with a poller. Don't use this
  while (!non_blocking && rc == 0){
    msgq_pollitem_t items[1];
    items[0].q = q;

    int t = (timeout != -1) ? timeout : 100;

    int n = msgq_poll(items, 1, t);
    rc = msgq_msg_recv(&msg, q);

    // The poll indicated a message was ready, but the receive failed. Try again
    if (n == 1 && rc == 0){
      continue;
    }

    if (timeout != -1){
      break;
    }
  }

  if (rc > 0){
    r = new Message;
    r->takeOwnership(msg.data, msg.size);
  }

  return r;
}

void SubSocket::setTimeout(int t){
  timeout = t;
}

SubSocket * SubSocket::create(std::string endpoint, bool conflate, size_t segment_size){
  SubSocket *s = new SubSocket();
  int r = s->connect(endpoint, conflate, segment_size);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to connect SubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

SubSocket::~SubSocket(){
  for (int fd : fake_fds) {
    if (fd >= 0) ::close(fd);
  }
  if (fake_state != nullptr) {
    munmap(fake_state, sizeof(EventState));
  }

  if (q != NULL){
    msgq_close_queue(q);
    delete q;
  }
}

int PubSocket::connect(std::string endpoint, size_t segment_size){
  q = new msgq_queue_t;
  size_t size = segment_size > 0 ? segment_size : DEFAULT_SEGMENT_SIZE;
  int r = msgq_new_queue(q, endpoint.c_str(), size);
  if (r != 0){
    return r;
  }

  msgq_init_publisher(q);

  return 0;
}

int PubSocket::sendMessage(Message *message){
  msgq_msg_t msg;
  msg.data = message->getData();
  msg.size = message->getSize();

  return msgq_msg_send(&msg, q);
}

int PubSocket::send(char *data, size_t size){
  msgq_msg_t msg;
  msg.data = data;
  msg.size = size;

  return msgq_msg_send(&msg, q);
}

bool PubSocket::all_readers_updated() {
  return msgq_all_readers_updated(q);
}

PubSocket * PubSocket::create(std::string endpoint, size_t segment_size){
  PubSocket *s = new PubSocket();
  int r = s->connect(endpoint, segment_size);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to bind PubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

PubSocket::~PubSocket(){
  if (q != NULL){
    msgq_close_queue(q);
    delete q;
  }
}

Poller::Poller() : fake(messaging_use_fake()) {}

Poller::Poller(std::vector<SubSocket*> sockets) : Poller() {
  for (auto s : sockets){
    registerSocket(s);
  }
}

void Poller::registerSocket(SubSocket * socket){
  sockets.push_back(socket);
  polls.push_back({socket->getQueue(), 0});
}

std::vector<SubSocket*> Poller::poll(int timeout){
  // in fake mode, the sockets block in receive() until the test releases them
  if (fake) {
    return sockets;
  }

  std::vector<SubSocket*> r;

  msgq_poll(polls.data(), polls.size(), timeout);
  for (size_t i = 0; i < polls.size(); i++){
    if (polls[i].revents){
      r.push_back(sockets[i]);
    }
  }

  return r;
}
