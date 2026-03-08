#include <cassert>
#include <iostream>
#include <string>

#include "msgq/ipc.h"
#include "msgq/impl_msgq.h"
#include "msgq/impl_fake.h"

bool messaging_use_fake(){
  char* fake_enabled = std::getenv("CEREAL_FAKE");
  return fake_enabled != NULL;
}

Context * Context::create(){
  return new Context();
}

SubSocket * SubSocket::create(){
  if (messaging_use_fake()) {
    return new FakeSubSocket<MSGQSubSocket>();
  }
  return new MSGQSubSocket();
}

SubSocket * SubSocket::create(Context * context, std::string endpoint, std::string address, bool conflate, bool check_endpoint, size_t segment_size){
  SubSocket *s = SubSocket::create();
  int r = s->connect(context, endpoint, address, conflate, check_endpoint, segment_size);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to connect SubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

PubSocket * PubSocket::create(){
  return new PubSocket();
}

PubSocket * PubSocket::create(Context * context, std::string endpoint, bool check_endpoint, size_t segment_size){
  PubSocket *s = PubSocket::create();
  int r = s->connect(context, endpoint, check_endpoint, segment_size);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to bind PubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

Poller * Poller::create(){
  if (messaging_use_fake()) {
    return new FakePoller();
  }
  return new MSGQPoller();
}

Poller * Poller::create(std::vector<SubSocket*> sockets){
  Poller * p = Poller::create();

  for (auto s : sockets){
    p->registerSocket(s);
  }
  return p;
}
