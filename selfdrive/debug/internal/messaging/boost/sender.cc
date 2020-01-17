#include <boost/interprocess/ipc/message_queue.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cassert>

#define N 1024
#define MSGS 1e5

using namespace boost::interprocess;

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
  message_queue *pq = pub_queue("queue_1");
  message_queue *sq = sub_queue("queue_2");
  std::cout << "Ready" << std::endl;

  auto start = std::chrono::steady_clock::now();
  char * rcv_msg = new char[N];
  char * snd_msg = new char[N];

  unsigned int priority;
  std::size_t recvd_size;

  for (int i = 0; i < MSGS; i++){
    sprintf(snd_msg, "%d", i);

    pq->send(snd_msg, N, 0);
    sq->receive(rcv_msg, N, recvd_size, priority);
  }

  auto end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
  double throughput = ((double) MSGS / (double) elapsed);

  std::cout << "Elapsed: " << elapsed << " s" << std::endl;
  std::cout << "Throughput: " << throughput << " msg/s" << std::endl;

  return 0;
}
