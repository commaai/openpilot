#include "messaging.hpp"
#include "impl_zmq.hpp"
#include "impl_msgq.hpp"

Context * Context::create(){
  Context * c;
  if (std::getenv("ZMQ")){
    c = new ZMQContext();
  } else {
    c = new MSGQContext();
  }
  return c;
}

SubSocket * SubSocket::create(){
  SubSocket * s;
  if (std::getenv("ZMQ")){
    s = new ZMQSubSocket();
  } else {
    s = new MSGQSubSocket();
  }
  return s;
}

SubSocket * SubSocket::create(Context * context, std::string endpoint){
  SubSocket *s = SubSocket::create();
  int r = s->connect(context, endpoint, "127.0.0.1");

  if (r == 0) {
    return s;
  } else {
    delete s;
    return NULL;
  }
}

SubSocket * SubSocket::create(Context * context, std::string endpoint, std::string address){
  SubSocket *s = SubSocket::create();
  int r = s->connect(context, endpoint, address);

  if (r == 0) {
    return s;
  } else {
    delete s;
    return NULL;
  }
}

SubSocket * SubSocket::create(Context * context, std::string endpoint, std::string address, bool conflate){
  SubSocket *s = SubSocket::create();
  int r = s->connect(context, endpoint, address, conflate);

  if (r == 0) {
    return s;
  } else {
    delete s;
    return NULL;
  }
}

PubSocket * PubSocket::create(){
  PubSocket * s;
  if (std::getenv("ZMQ")){
    s = new ZMQPubSocket();
  } else {
    s = new MSGQPubSocket();
  }
  return s;
}

PubSocket * PubSocket::create(Context * context, std::string endpoint){
  PubSocket *s = PubSocket::create();
  int r = s->connect(context, endpoint);

  if (r == 0) {
    return s;
  } else {
    delete s;
    return NULL;
  }
}

Poller * Poller::create(){
  Poller * p;
  if (std::getenv("ZMQ")){
    p = new ZMQPoller();
  } else {
    p = new MSGQPoller();
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

extern "C" Context * messaging_context_create() {
  return Context::create();
}

extern "C" SubSocket * messaging_subsocket_create(Context* context, const char* endpoint) {
  return SubSocket::create(context, std::string(endpoint));
}

extern "C" PubSocket * messaging_pubsocket_create(Context* context, const char* endpoint) {
  return PubSocket::create(context, std::string(endpoint));
}

extern "C" Poller * messaging_poller_create(SubSocket** sockets, int size) {
  std::vector<SubSocket*> socketsVec(sockets, sockets + size);
  return Poller::create(socketsVec);
}
