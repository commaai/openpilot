#ifndef CHANNEL_HPP
#define CHANNEL_HPP

#include <mutex>
#include <list>
#include <condition_variable>

template<class item>
class channel {
private:
  std::list<item> queue;
  std::mutex m;
  std::condition_variable cv;
public:
  void put(const item &i) {
    std::unique_lock<std::mutex> lock(m);
    queue.push_back(i);
    cv.notify_one();
  }
  void put_front(const item &i) {
    std::unique_lock<std::mutex> lock(m);
    queue.push_front(i);
    cv.notify_one();
  }
  item get() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&](){ return !queue.empty(); });
    item result = queue.front();
    queue.pop_front();
    return result;
  }
};

#endif

