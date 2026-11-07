#pragma once

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// Reusing the same threads matters: a std::async thread per update lands each allocation in a different
// glibc malloc arena and the process RSS grows without bound.
class ThreadPool {
public:
  static ThreadPool &instance() {
    static ThreadPool pool(std::clamp(std::thread::hardware_concurrency(), 2u, 4u));
    return pool;
  }

  static ThreadPool &media() { static ThreadPool pool(4); return pool; }
  static ThreadPool &history() { static ThreadPool pool(2); return pool; }
  static ThreadPool &analysis() { static ThreadPool pool(2); return pool; }

  template <class Function>
  auto run(Function &&fn) -> std::future<std::invoke_result_t<Function>> {
    using Result = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(fn));
    auto future = task->get_future();
    {
      std::lock_guard lk(mutex_);
      tasks_.push([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

  ~ThreadPool() {
    {
      std::lock_guard lk(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &thread : threads_) thread.join();
  }

private:
  explicit ThreadPool(unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
      threads_.emplace_back([this]() {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock lk(mutex_);
            cv_.wait(lk, [this]() { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  std::vector<std::thread> threads_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
};

// fn(begin, end) over [0, n) split into one chunk per pool thread plus one for the caller, which also
// waits for the others. Not for use from a pool thread.
inline void parallelFor(size_t n, const std::function<void(size_t begin, size_t end)> &fn) {
  const size_t chunks = std::clamp<size_t>(std::thread::hardware_concurrency(), 2, 4) + 1;
  const size_t chunk = (n + chunks - 1) / chunks;
  if (chunk == 0) return;
  std::vector<std::future<void>> futures;
  size_t begin = chunk;
  for (; begin < n; begin += chunk) {
    futures.push_back(ThreadPool::instance().run([&fn, begin, end = std::min(begin + chunk, n)]() { fn(begin, end); }));
  }
  // All tasks must finish before captures can go out of scope, even if one fails.
  std::exception_ptr error;
  try { fn(0, std::min(chunk, n)); } catch (...) { error = std::current_exception(); }
  for (auto &f : futures) {
    try { f.get(); } catch (...) { if (!error) error = std::current_exception(); }
  }
  if (error) std::rethrow_exception(error);
}
