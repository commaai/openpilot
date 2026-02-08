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

#include "debug.h"
#include "vector.h"
#include "function.h"
#include "windows-sanity.h"  // work-around macro conflict with `ERROR`

KJ_BEGIN_HEADER

namespace kj {

class TestRunner;

class TestCase {
public:
  TestCase(const char* file, uint line, const char* description);
  ~TestCase();

  virtual void run() = 0;

protected:
  template <typename Func>
  void doBenchmark(Func&& func) {
    // Perform a benchmark with configurable iterations. func() will be called N times, where N
    // is set by the --benchmark CLI flag. This defaults to 1, so that when --benchmark is not
    // specified, we only test that the benchmark works.
    //
    // In the future, this could adaptively choose iteration count by running a few iterations to
    // find out how fast the benchmark is, then scaling.

    for (size_t i = iterCount(); i-- > 0;) {
      func();
    }
  }

private:
  const char* file;
  uint line;
  const char* description;
  TestCase* next;
  TestCase** prev;
  bool matchedFilter;

  static size_t iterCount();

  friend class TestRunner;
};

#define KJ_TEST(description) \
  /* Make sure the linker fails if tests are not in anonymous namespaces. */ \
  extern int KJ_CONCAT(YouMustWrapTestsInAnonymousNamespace, __COUNTER__) KJ_UNUSED; \
  class KJ_UNIQUE_NAME(TestCase): public ::kj::TestCase { \
  public: \
    KJ_UNIQUE_NAME(TestCase)(): ::kj::TestCase(__FILE__, __LINE__, description) {} \
    void run() override; \
  } KJ_UNIQUE_NAME(testCase); \
  void KJ_UNIQUE_NAME(TestCase)::run()

#if KJ_MSVC_TRADITIONAL_CPP
#define KJ_INDIRECT_EXPAND(m, vargs) m vargs
#define KJ_FAIL_EXPECT(...) \
  KJ_INDIRECT_EXPAND(KJ_LOG, (ERROR , __VA_ARGS__));
#define KJ_EXPECT(cond, ...) \
  if (auto _kjCondition = ::kj::_::MAGIC_ASSERT << cond); \
  else KJ_INDIRECT_EXPAND(KJ_FAIL_EXPECT, ("failed: expected " #cond , _kjCondition, __VA_ARGS__))
#else
#define KJ_FAIL_EXPECT(...) \
  KJ_LOG(ERROR, ##__VA_ARGS__);
#define KJ_EXPECT(cond, ...) \
  if (auto _kjCondition = ::kj::_::MAGIC_ASSERT << cond); \
  else KJ_FAIL_EXPECT("failed: expected " #cond, _kjCondition, ##__VA_ARGS__)
#endif

#if _MSC_VER && !defined(__clang__)
#define KJ_EXPECT_THROW_RECOVERABLE(type, code, ...) \
  do { \
    KJ_IF_MAYBE(e, ::kj::runCatchingExceptions([&]() { code; })) { \
      KJ_INDIRECT_EXPAND(KJ_EXPECT, (e->getType() == ::kj::Exception::Type::type, \
          "code threw wrong exception type: " #code, *e, __VA_ARGS__)); \
    } else { \
      KJ_INDIRECT_EXPAND(KJ_FAIL_EXPECT, ("code did not throw: " #code, __VA_ARGS__)); \
    } \
  } while (false)

#define KJ_EXPECT_THROW_RECOVERABLE_MESSAGE(message, code, ...) \
  do { \
    KJ_IF_MAYBE(e, ::kj::runCatchingExceptions([&]() { code; })) { \
      KJ_INDIRECT_EXPAND(KJ_EXPECT, (::kj::_::hasSubstring(e->getDescription(), message), \
          "exception description didn't contain expected substring", *e, __VA_ARGS__)); \
    } else { \
      KJ_INDIRECT_EXPAND(KJ_FAIL_EXPECT, ("code did not throw: " #code, __VA_ARGS__)); \
    } \
  } while (false)
#else
#define KJ_EXPECT_THROW_RECOVERABLE(type, code, ...) \
  do { \
    KJ_IF_MAYBE(e, ::kj::runCatchingExceptions([&]() { code; })) { \
      KJ_EXPECT(e->getType() == ::kj::Exception::Type::type, \
          "code threw wrong exception type: " #code, *e, ##__VA_ARGS__); \
    } else { \
      KJ_FAIL_EXPECT("code did not throw: " #code, ##__VA_ARGS__); \
    } \
  } while (false)

#define KJ_EXPECT_THROW_RECOVERABLE_MESSAGE(message, code, ...) \
  do { \
    KJ_IF_MAYBE(e, ::kj::runCatchingExceptions([&]() { code; })) { \
      KJ_EXPECT(::kj::_::hasSubstring(e->getDescription(), message), \
          "exception description didn't contain expected substring", *e, ##__VA_ARGS__); \
    } else { \
      KJ_FAIL_EXPECT("code did not throw: " #code, ##__VA_ARGS__); \
    } \
  } while (false)
#endif

#if KJ_NO_EXCEPTIONS
#define KJ_EXPECT_THROW(type, code, ...) \
  do { \
    KJ_EXPECT(::kj::_::expectFatalThrow(::kj::Exception::Type::type, nullptr, [&]() { code; })); \
  } while (false)
#define KJ_EXPECT_THROW_MESSAGE(message, code, ...) \
  do { \
    KJ_EXPECT(::kj::_::expectFatalThrow(nullptr, kj::StringPtr(message), [&]() { code; })); \
  } while (false)
#else
#define KJ_EXPECT_THROW KJ_EXPECT_THROW_RECOVERABLE
#define KJ_EXPECT_THROW_MESSAGE KJ_EXPECT_THROW_RECOVERABLE_MESSAGE
#endif

#define KJ_EXPECT_EXIT(statusCode, code) \
  do { \
    KJ_EXPECT(::kj::_::expectExit(statusCode, [&]() { code; })); \
  } while (false)
// Forks the code and expects it to exit with a given code.

#define KJ_EXPECT_SIGNAL(signal, code) \
  do { \
    KJ_EXPECT(::kj::_::expectSignal(signal, [&]() { code; })); \
  } while (false)
// Forks the code and expects it to trigger a signal.
// In the child resets all signal handlers as printStackTraceOnCrash sets.

#define KJ_EXPECT_LOG(level, substring) \
  ::kj::_::LogExpectation KJ_UNIQUE_NAME(_kjLogExpectation)(::kj::LogSeverity::level, substring)
// Expects that a log message with the given level and substring text will be printed within
// the current scope. This message will not cause the test to fail, even if it is an error.

// =======================================================================================

namespace _ {  // private

bool hasSubstring(kj::StringPtr haystack, kj::StringPtr needle);

#if KJ_NO_EXCEPTIONS
bool expectFatalThrow(Maybe<Exception::Type> type, Maybe<StringPtr> message,
                      Function<void()> code);
// Expects that the given code will throw a fatal exception matching the given type and/or message.
// Since exceptions are disabled, the test will fork() and run in a subprocess. On Windows, where
// fork() is not available, this always returns true.
#endif

bool expectExit(Maybe<int> statusCode, FunctionParam<void()> code) noexcept;
// Expects that the given code will exit with a given statusCode.
// The test will fork() and run in a subprocess. On Windows, where fork() is not available,
// this always returns true.

bool expectSignal(Maybe<int> signal, FunctionParam<void()> code) noexcept;
// Expects that the given code will trigger a signal.
// The test will fork() and run in a subprocess. On Windows, where fork() is not available,
// this always returns true.
// Resets signal handlers to default prior to running the code in the child process.

class LogExpectation: public ExceptionCallback {
public:
  LogExpectation(LogSeverity severity, StringPtr substring);
  ~LogExpectation();

  void logMessage(LogSeverity severity, const char* file, int line, int contextDepth,
                  String&& text) override;

private:
  LogSeverity severity;
  StringPtr substring;
  bool seen;
  UnwindDetector unwindDetector;
};

class GlobFilter {
  // Implements glob filters for the --filter flag.
  //
  // Exposed in header only for testing.

public:
  explicit GlobFilter(const char* pattern);
  explicit GlobFilter(ArrayPtr<const char> pattern);

  bool matches(StringPtr name);

private:
  String pattern;
  Vector<uint> states;

  void applyState(char c, int state);
};

}  // namespace _ (private)
}  // namespace kj

KJ_END_HEADER
