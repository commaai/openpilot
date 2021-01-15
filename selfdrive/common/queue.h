#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class SafeQueue {
public:
  SafeQueue() = default;
  ~SafeQueue() {}
  void push(T v) {
    {
      std::unique_lock lk(m);
      q.push(v);
    }
    cv.notify_one();
  }
  T pop() {
    std::unique_lock lk(m);
    cv.wait(lk, [this] { return !q.empty(); });
    T v = q.front();
    q.pop();
    return v;
  }
  bool try_pop(T& v) {
    std::unique_lock lk(m);
    if (q.empty()) return false;

    v = q.front();
    q.pop();
    return true;
  }

  bool empty() const {
    std::scoped_lock lk(m);
    return q.empty();
  }

  size_t size() const {
    std::scoped_lock lk(m);
    return q.size();
  }

private:
  mutable std::mutex m;
  std::condition_variable cv;
  std::queue<T> q;
};
