#pragma once

#include <functional>
#include <utility>
#include <vector>

// Minimal observer list replacing Qt signals in the de-Qt'd core.
// Not thread-safe: connect and emit only from the UI thread. Producer threads
// stage data under a mutex and events are emitted from AbstractStream::update()
// or direct API calls on the UI thread.
//
// Namespaced rather than global because
// tools/replay/seg_mgr.h declares an unrelated, non-template `class Event`
// (a replayed log event) that streams/ needs alongside this one in the same
// translation units.
namespace loggy {

template <typename... Args>
class Event {
public:
  void connect(std::function<void(Args...)> fn) { callbacks_.push_back(std::move(fn)); }
  void operator()(Args... args) const {
    for (const auto &fn : callbacks_) fn(args...);
  }

private:
  std::vector<std::function<void(Args...)>> callbacks_;
};

}  // namespace loggy
