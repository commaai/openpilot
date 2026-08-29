#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// QtConcurrent::run equivalent. Reusing the same threads matters: a std::async thread per update lands each
// allocation in a different glibc malloc arena and the process RSS grows without bound.
class ThreadPool {
public:
  static ThreadPool &instance() {
    // a few reused threads: every worker thread owns a glibc malloc arena and each arena fragments on its own
    static ThreadPool pool(std::clamp(std::thread::hardware_concurrency(), 2u, 4u));
    return pool;
  }

  std::future<void> run(std::function<void()> fn) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
    std::future<void> future = task->get_future();
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
    for (auto &t : threads_) t.join();
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
