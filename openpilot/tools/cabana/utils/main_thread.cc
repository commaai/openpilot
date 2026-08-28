#include "tools/cabana/utils/main_thread.h"

#include <mutex>
#include <thread>
#include <utility>
#include <vector>

static const std::thread::id main_thread_id = std::this_thread::get_id();
static std::mutex main_thread_queue_mutex;
static std::vector<std::function<void()>> main_thread_queue;

bool utils::isMainThread() { return std::this_thread::get_id() == main_thread_id; }

void utils::runOnMainThread(std::function<void()> fn) {
  if (isMainThread()) {
    fn();
  } else {
    std::lock_guard lk(main_thread_queue_mutex);
    main_thread_queue.push_back(std::move(fn));
  }
}

void utils::drainMainThreadQueue() {
  std::vector<std::function<void()>> fns;
  {
    std::lock_guard lk(main_thread_queue_mutex);
    fns.swap(main_thread_queue);
  }
  for (auto &fn : fns) fn();
}
