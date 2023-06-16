#include <cassert>
#include <iostream>

#include "cereal/messaging/messaging.h"
#include "cereal/messaging/impl_zmq.h"
#include "cereal/messaging/impl_msgq.h"
#include "cereal/messaging/impl_fake.h"

#ifdef __APPLE__
const bool MUST_USE_ZMQ = true;
#else
const bool MUST_USE_ZMQ = false;
#endif

bool messaging_use_zmq(){
  if (std::getenv("ZMQ") || MUST_USE_ZMQ) {
    if (std::getenv("OPENPILOT_PREFIX")) {
      std::cerr << "OPENPILOT_PREFIX not supported with ZMQ backend\n";
      assert(false);
    }
    return true;
  }
  return false;
}

bool messaging_use_fake(){
  char* fake_enabled = std::getenv("CEREAL_FAKE");
  return fake_enabled != NULL;
}

Context * Context::create(){
  Context * c;
  if (messaging_use_zmq()){
    c = new ZMQContext();
  } else {
    c = new MSGQContext();
  }
  return c;
}

SubSocket * SubSocket::create(){
  SubSocket * s;
  if (messaging_use_fake()) {
    if (messaging_use_zmq()) {
      s = new FakeSubSocket<ZMQSubSocket>();
    } else {
      s = new FakeSubSocket<MSGQSubSocket>();
    }
  } else {
    if (messaging_use_zmq()){
      s = new ZMQSubSocket();
    } else {
      s = new MSGQSubSocket();
    }
  }

  return s;
}

SubSocket * SubSocket::create(Context * context, std::string endpoint, std::string address, bool conflate, bool check_endpoint){
  SubSocket *s = SubSocket::create();
  int r = s->connect(context, endpoint, address, conflate, check_endpoint);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to connect SubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

PubSocket * PubSocket::create(){
  PubSocket * s;
  if (messaging_use_zmq()){
    s = new ZMQPubSocket();
  } else {
    s = new MSGQPubSocket();
  }

  return s;
}

PubSocket * PubSocket::create(Context * context, std::string endpoint, bool check_endpoint){
  PubSocket *s = PubSocket::create();
  int r = s->connect(context, endpoint, check_endpoint);

  if (r == 0) {
    return s;
  } else {
    std::cerr << "Error, failed to bind PubSocket to " << endpoint << ": " << strerror(errno) << std::endl;

    delete s;
    return nullptr;
  }
}

Poller * Poller::create(){
  Poller * p;
  if (messaging_use_fake()) {
    p = new FakePoller();
  } else {
    if (messaging_use_zmq()){
      p = new ZMQPoller();
    } else {
      p = new MSGQPoller();
    }
  }
  return p;
}

Poller * Poller::create(std::vector<SubSocket*> sockets){
  Poller * p = Poller::create();

  for (auto s : sockets){
    p->registerSocket(s);
  }
  return p;
}
