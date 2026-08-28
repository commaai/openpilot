#pragma once

#include <functional>

namespace utils {

bool isMainThread();
// inline on the main thread, queued until drainMainThreadQueue() otherwise
void runOnMainThread(std::function<void()> fn);
void drainMainThreadQueue();

}  // namespace utils
