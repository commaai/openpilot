#include "messaging.hpp"
#include "impl_zmq.hpp"

Context * Context::create(){
  Context * c = new ZMQContext();
  return c;
}

SubSocket * SubSocket::create(){
  SubSocket * s = new ZMQSubSocket();
  return s;
}

SubSocket * SubSocket::create(Context * context, std::string endpoint){
  SubSocket *s = SubSocket::create();
  s->connect(context, endpoint, "127.0.0.1");

  return s;
}

SubSocket * SubSocket::create(Context * context, std::string endpoint, std::string address){
  SubSocket *s = SubSocket::create();
  s->connect(context, endpoint, address);

  return s;
}

PubSocket * PubSocket::create(){
  PubSocket * s = new ZMQPubSocket();
  return s;
}

PubSocket * PubSocket::create(Context * context, std::string endpoint){
  PubSocket *s = PubSocket::create();
  s->connect(context, endpoint);
  return s;
}

Poller * Poller::create(){
  Poller * p = new ZMQPoller();
  return p;
}

Poller * Poller::create(std::vector<SubSocket*> sockets){
  Poller * p = Poller::create();

  for (auto s : sockets){
    p->registerSocket(s);
  }
  return p;
}
