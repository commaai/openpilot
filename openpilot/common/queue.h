#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template <class T>
class SafeQueue {
public:
  SafeQueue() = default;

  void push(const T& v) {
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

  bool try_pop(T& v, int timeout_ms = 0) {
    std::unique_lock lk(m);
    if (!cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] { return !q.empty(); })) {
      return false;
    }
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
