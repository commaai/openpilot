#include <boost/interprocess/ipc/message_queue.hpp>
#include <iostream>
#include <vector>
#include <thread>

using namespace boost::interprocess;
#define N 1024

message_queue *sub_queue(const char *name){
  while (true){
    try {
      message_queue *mq = new message_queue(open_only, name);
      return mq;
    }
    catch(interprocess_exception &ex){
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

  }
}

message_queue *pub_queue(const char *name){
  message_queue::remove(name);
  message_queue *mq = new message_queue(create_only, name, 100, N);
  return mq;
}



int main ()
{

  message_queue::remove("queue_1");
  message_queue::remove("queue_2");

  message_queue *pq = pub_queue("queue_2");
  message_queue *sq = sub_queue("queue_1");
  std::cout << "Ready" << std::endl;

  unsigned int priority;
  std::size_t recvd_size;

  char * rcv_msg = new char[N];

  while (true){

    sq->receive(rcv_msg, N, recvd_size, priority);
    assert(N == recvd_size);

    pq->send(rcv_msg, N, 0);
  }

  return 0;
}
