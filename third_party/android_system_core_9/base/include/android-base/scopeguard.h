/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_BASE_SCOPEGUARD_H
#define ANDROID_BASE_SCOPEGUARD_H

#include <utility>  // for std::move, std::forward

namespace android {
namespace base {

// ScopeGuard ensures that the specified functor is executed no matter how the
// current scope exits.
template <typename F>
class ScopeGuard {
 public:
  ScopeGuard(F&& f) : f_(std::forward<F>(f)), active_(true) {}

  ScopeGuard(ScopeGuard&& that) : f_(std::move(that.f_)), active_(that.active_) {
    that.active_ = false;
  }

  template <typename Functor>
  ScopeGuard(ScopeGuard<Functor>&& that) : f_(std::move(that.f_)), active_(that.active_) {
    that.active_ = false;
  }

  ~ScopeGuard() {
    if (active_) f_();
  }

  ScopeGuard() = delete;
  ScopeGuard(const ScopeGuard&) = delete;
  void operator=(const ScopeGuard&) = delete;
  void operator=(ScopeGuard&& that) = delete;

  void Disable() { active_ = false; }

  bool active() const { return active_; }

 private:
  template <typename Functor>
  friend class ScopeGuard;

  F f_;
  bool active_;
};

template <typename F>
ScopeGuard<F> make_scope_guard(F&& f) {
  return ScopeGuard<F>(std::forward<F>(f));
}

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_SCOPEGUARD_H
