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

#pragma once

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

KJ_BEGIN_HEADER

namespace kj {

#if __GNUC__

#define KJ_THREADLOCAL_PTR(type) static __thread type*
// GCC's __thread is lighter-weight than thread_local and is good enough for our purposes.
//
// TODO(cleanup): The above comment was written many years ago. Is it still true? Shouldn't the
//   compiler be smart enough to optimize a thread_local of POD type?

#else

#define KJ_THREADLOCAL_PTR(type) static thread_local type*

#endif // KJ_USE_PTHREAD_TLS

}  // namespace kj

KJ_END_HEADER
