#include <assert.h>
#include <map>
#include "messaging.hpp"
class SocketMaster {
 public:
  SocketMaster() : poller(NULL) { context = Context::create(); }

  ~SocketMaster() {
    if (poller) { delete poller; }
    for (auto s : subSockets) { delete s; }
    for (auto s : pubSockets) { delete s; }
    if (context) { delete context; }
  }

  PubSocket *createPubSocket(const char *name) {
    PubSocket *socket = PubSocket::create(context, name);
    assert(socket != NULL);
    pubSockets.push_back(socket);
    return socket;
  }

  SubSocket *createSubSocket(const char *name, bool needPoller, const char *address = NULL, bool conflate = false) {
    SubSocket *socket = SubSocket::create(context, name, address ? address : "127.0.0.1", conflate);
    assert(socket != NULL);
    if (needPoller) {
      if (!poller) { poller = Poller::create(); }
      poller->registerSocket(socket);
    }
    subSockets.push_back(socket);
    return socket;
  }

  void createSubSockets(std::vector<const char *> services) {
    for (auto name : services) { createSubSocket(name, true); }
  }

  std::vector<Message *> pollMessages(int timeout) {
    assert(poller != NULL);
    std::vector<Message *> messages;
    while (true) {
      auto polls = poller->poll(timeout);
      if (polls.size() == 0) {
        break;
      }
      for (auto sock : polls) {
        Message *msg = sock->receive();
        if (msg) {
          messages.push_back(msg);
        }
      }
    }
    return messages;
  }

  inline Poller *getPoller() {assert(poller != NULL);  return poller;}
  inline std::vector<SubSocket *> poll(int timeout) { assert(poller != NULL); return poller->poll(timeout); }

 private:
  Context *context;
  Poller *poller;
  std::vector<SubSocket *> subSockets;
  std::vector<PubSocket *> pubSockets;
};
