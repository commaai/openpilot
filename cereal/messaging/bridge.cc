#include <iostream>
#include <string>
#include <cassert>
#include <csignal>
#include <map>

typedef void (*sighandler_t)(int sig);

#include "services.h"

#include "impl_msgq.hpp"
#include "impl_zmq.hpp"

void sigpipe_handler(int sig) {
  assert(sig == SIGPIPE);
  std::cout << "SIGPIPE received" << std::endl;
}

static std::vector<std::string> get_services() {
  std::vector<std::string> name_list;

  for (const auto& it : services) {
    std::string name = it.name;
    if (name == "plusFrame" || name == "uiLayoutState") continue;
    name_list.push_back(name);
  }

  return name_list;
}


int main(void){
  signal(SIGPIPE, (sighandler_t)sigpipe_handler);

  auto endpoints = get_services();

  std::map<SubSocket*, PubSocket*> sub2pub;

  Context *zmq_context = new ZMQContext();
  Context *msgq_context = new MSGQContext();
  Poller *poller = new MSGQPoller();

  for (auto endpoint: endpoints){
    SubSocket * msgq_sock = new MSGQSubSocket();
    msgq_sock->connect(msgq_context, endpoint, "127.0.0.1", false);
    poller->registerSocket(msgq_sock);

    PubSocket * zmq_sock = new ZMQPubSocket();
    zmq_sock->connect(zmq_context, endpoint);

    sub2pub[msgq_sock] = zmq_sock;
  }


  while (true){
    for (auto sub_sock : poller->poll(100)){
      Message * msg = sub_sock->receive();
      if (msg == NULL) continue;
      sub2pub[sub_sock]->sendMessage(msg);
      delete msg;
    }
  }
  return 0;
}
