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

#ifndef KJ_THREAD_H_
#define KJ_THREAD_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "common.h"
#include "function.h"
#include "exception.h"

namespace kj {

class Thread {
  // A thread!  Pass a lambda to the constructor, and it runs in the thread.  The destructor joins
  // the thread.  If the function throws an exception, it is rethrown from the thread's destructor
  // (if not unwinding from another exception).

public:
  explicit Thread(Function<void()> func);
  KJ_DISALLOW_COPY(Thread);

  ~Thread() noexcept(false);

#if !_WIN32
  void sendSignal(int signo);
  // Send a Unix signal to the given thread, using pthread_kill or an equivalent.
#endif

  void detach();
  // Don't join the thread in ~Thread().

private:
  struct ThreadState {
    Function<void()> func;
    kj::Maybe<kj::Exception> exception;

    unsigned int refcount;
    // Owned by the parent thread and the child thread.

    void unref();
  };
  ThreadState* state;

#if _WIN32
  void* threadHandle;
#else
  unsigned long long threadId;  // actually pthread_t
#endif
  bool detached = false;

#if _WIN32
  static unsigned long __stdcall runThread(void* ptr);
#else
  static void* runThread(void* ptr);
#endif
};

}  // namespace kj

#endif  // KJ_THREAD_H_
