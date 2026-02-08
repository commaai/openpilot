// Copyright (c) 2021 Cloudflare, Inc. and contributors
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

#include "string.h"

KJ_BEGIN_HEADER

// GCC does not implement __builtin_COLUMN() as that's non-standard but MSVC & clang do.
// MSVC does as of version https://github.com/microsoft/STL/issues/54) but there's currently not any
// pressing need for this for MSVC & writing the write compiler version check is annoying.
// Checking for clang version is problematic due to the way that XCode lies about __clang_major__.
// Instead we use __has_builtin as the feature check to check clang.
// Context: https://github.com/capnproto/capnproto/issues/1305
#ifdef __has_builtin
#if __has_builtin(__builtin_COLUMN)
#define KJ_CALLER_COLUMN() __builtin_COLUMN()
#else
#define KJ_CALLER_COLUMN() 0
#endif
#else
#define KJ_CALLER_COLUMN() 0
#endif

#if __cplusplus > 201703L
#define KJ_COMPILER_SUPPORTS_SOURCE_LOCATION 1
#elif defined(__has_builtin)
// Clang 9 added these builtins: https://releases.llvm.org/9.0.0/tools/clang/docs/LanguageExtensions.html
// Use __has_builtin as the way to detect this because __clang_major__ is unreliable (see above
// about issue with Xcode-provided clang).
#define KJ_COMPILER_SUPPORTS_SOURCE_LOCATION (   \
  __has_builtin(__builtin_FILE) &&               \
  __has_builtin(__builtin_LINE) &&               \
  __has_builtin(__builtin_FUNCTION)              \
  )
#elif __GNUC__ >= 5
// GCC 5 supports the required builtins: https://gcc.gnu.org/onlinedocs/gcc-5.1.0/gcc/Other-Builtins.html
#define KJ_COMPILER_SUPPORTS_SOURCE_LOCATION 1
#endif

namespace kj {
class SourceLocation {
  // libc++ doesn't seem to implement <source_location> (or even <experimental/source_location>), so
  // this is a non-STL wrapper over the compiler primitives (these are the same across MSVC/clang/
  // gcc). Additionally this uses kj::StringPtr for holding the strings instead of const char* which
  // makes it integrate a little more nicely into KJ.

  struct Badge { explicit constexpr Badge() = default; };
  // Neat little trick to make sure we can never call SourceLocation with explicit arguments.
public:
#if !KJ_COMPILER_SUPPORTS_SOURCE_LOCATION
  constexpr SourceLocation() : fileName("??"), function("??"), lineNumber(0), columnNumber(0) {}
  // Constructs a dummy source location that's not pointing at anything.
#else
  constexpr SourceLocation(Badge = Badge{}, const char* file = __builtin_FILE(),
      const char* func = __builtin_FUNCTION(), uint line = __builtin_LINE(),
      uint column = KJ_CALLER_COLUMN())
    : fileName(file), function(func), lineNumber(line), columnNumber(column)
  {}
#endif

#if KJ_COMPILER_SUPPORTS_SOURCE_LOCATION
  // This can only be exposed if we actually generate valid SourceLocation objects as otherwise all
  // SourceLocation objects would confusingly (and likely problematically) be equated equal.
  constexpr bool operator==(const SourceLocation& o) const {
    // Pointer equality is fine here based on how SourceLocation operates & how compilers will
    // intern all duplicate string constants.
    return fileName == o.fileName && function == o.function && lineNumber == o.lineNumber &&
        columnNumber == o.columnNumber;
  }
#endif

  const char* fileName;
  const char* function;
  uint lineNumber;
  uint columnNumber;
};

kj::String KJ_STRINGIFY(const SourceLocation& l);

class NoopSourceLocation {
  // This is used in places where we want to conditionally compile out tracking the source location.
  // As such it intentionally lacks all the features but the default constructor so that the API
  // isn't accidentally used in the wrong compilation context.
};

KJ_UNUSED static kj::String KJ_STRINGIFY(const NoopSourceLocation& l) {
  return kj::String();
}
}  // namespace kj

KJ_END_HEADER
