// Copyright (c) 2014, Jason Choy <jjwchoy@gmail.com>
// Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef KJ_THREADLOCAL_H_
#define KJ_THREADLOCAL_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif
// This file declares a macro `KJ_THREADLOCAL_PTR` for declaring thread-local pointer-typed
// variables.  Use like:
//     KJ_THREADLOCAL_PTR(MyType) foo = nullptr;
// This is equivalent to:
//     thread_local MyType* foo = nullptr;
// This can only be used at the global scope.
//
// AVOID USING THIS.  Use of thread-locals is discouraged because they often have many of the same
// properties as singletons: http://www.object-oriented-security.org/lets-argue/singletons
//
// Also, thread-locals tend to be hostile to event-driven code, which can be particularly
// surprising when using fibers (all fibers in the same thread will share the same threadlocals,
// even though they do not share a stack).
//
// That said, thread-locals are sometimes needed for runtime logistics in the KJ framework.  For
// example, the current exception callback and current EventLoop are stored as thread-local
// pointers.  Since KJ only ever needs to store pointers, not values, we avoid the question of
// whether these values' destructors need to be run, and we avoid the need for heap allocation.

#include "common.h"

#if !defined(KJ_USE_PTHREAD_THREADLOCAL) && defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
// iOS apparently does not support __thread (nor C++11 thread_local).
#define KJ_USE_PTHREAD_TLS 1
#endif
#endif

#if KJ_USE_PTHREAD_TLS
#include <pthread.h>
#endif

namespace kj {

#if KJ_USE_PTHREAD_TLS
// If __thread is unavailable, we'll fall back to pthreads.

#define KJ_THREADLOCAL_PTR(type) \
  namespace { struct KJ_UNIQUE_NAME(_kj_TlpTag); } \
  static ::kj::_::ThreadLocalPtr< type, KJ_UNIQUE_NAME(_kj_TlpTag)>
// Hack:  In order to ensure each thread-local results in a unique template instance, we declare
//   a one-off dummy type to use as the second type parameter.

namespace _ {  // private

template <typename T, typename>
class ThreadLocalPtr {
  // Hacky type to emulate __thread T*.  We need a separate instance of the ThreadLocalPtr template
  // for every thread-local variable, because we don't want to require a global constructor, and in
  // order to initialize the TLS on first use we need to use a local static variable (in getKey()).
  // Each template instance will get a separate such local static variable, fulfilling our need.

public:
  ThreadLocalPtr() = default;
  constexpr ThreadLocalPtr(decltype(nullptr)) {}
  // Allow initialization to nullptr without a global constructor.

  inline ThreadLocalPtr& operator=(T* val) {
    pthread_setspecific(getKey(), val);
    return *this;
  }

  inline operator T*() const {
    return get();
  }

  inline T& operator*() const {
    return *get();
  }

  inline T* operator->() const {
    return get();
  }

private:
  inline T* get() const {
    return reinterpret_cast<T*>(pthread_getspecific(getKey()));
  }

  inline static pthread_key_t getKey() {
    static pthread_key_t key = createKey();
    return key;
  }

  static pthread_key_t createKey() {
    pthread_key_t key;
    pthread_key_create(&key, 0);
    return key;
  }
};

}  // namespace _ (private)

#elif __GNUC__

#define KJ_THREADLOCAL_PTR(type) static __thread type*
// GCC's __thread is lighter-weight than thread_local and is good enough for our purposes.

#else

#define KJ_THREADLOCAL_PTR(type) static thread_local type*

#endif // KJ_USE_PTHREAD_TLS

}  // namespace kj

#endif  // KJ_THREADLOCAL_H_
