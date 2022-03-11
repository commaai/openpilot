/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_BASE_LOGGING_H
#define ANDROID_BASE_LOGGING_H

//
// Google-style C++ logging.
//

// This header provides a C++ stream interface to logging.
//
// To log:
//
//   LOG(INFO) << "Some text; " << some_value;
//
// Replace `INFO` with any severity from `enum LogSeverity`.
//
// To log the result of a failed function and include the string
// representation of `errno` at the end:
//
//   PLOG(ERROR) << "Write failed";
//
// The output will be something like `Write failed: I/O error`.
// Remember this as 'P' as in perror(3).
//
// To output your own types, simply implement operator<< as normal.
//
// By default, output goes to logcat on Android and stderr on the host.
// A process can use `SetLogger` to decide where all logging goes.
// Implementations are provided for logcat, stderr, and dmesg.
//
// By default, the process' name is used as the log tag.
// Code can choose a specific log tag by defining LOG_TAG
// before including this header.

// This header also provides assertions:
//
//   CHECK(must_be_true);
//   CHECK_EQ(a, b) << z_is_interesting_too;

// NOTE: For Windows, you must include logging.h after windows.h to allow the
// following code to suppress the evil ERROR macro:
#ifdef _WIN32
// windows.h includes wingdi.h which defines an evil macro ERROR.
#ifdef ERROR
#undef ERROR
#endif
#endif

#include <functional>
#include <memory>
#include <ostream>

#include "android-base/macros.h"

// Note: DO NOT USE DIRECTLY. Use LOG_TAG instead.
#ifdef _LOG_TAG_INTERNAL
#error "_LOG_TAG_INTERNAL must not be defined"
#endif
#ifdef LOG_TAG
#define _LOG_TAG_INTERNAL LOG_TAG
#else
#define _LOG_TAG_INTERNAL nullptr
#endif

namespace android {
namespace base {

enum LogSeverity {
  VERBOSE,
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  FATAL_WITHOUT_ABORT,
  FATAL,
};

enum LogId {
  DEFAULT,
  MAIN,
  SYSTEM,
};

using LogFunction = std::function<void(LogId, LogSeverity, const char*, const char*,
                                       unsigned int, const char*)>;
using AbortFunction = std::function<void(const char*)>;

void KernelLogger(LogId, LogSeverity, const char*, const char*, unsigned int, const char*);
void StderrLogger(LogId, LogSeverity, const char*, const char*, unsigned int, const char*);

void DefaultAborter(const char* abort_message);

std::string GetDefaultTag();
void SetDefaultTag(const std::string& tag);

#ifdef __ANDROID__
// We expose this even though it is the default because a user that wants to
// override the default log buffer will have to construct this themselves.
class LogdLogger {
 public:
  explicit LogdLogger(LogId default_log_id = android::base::MAIN);

  void operator()(LogId, LogSeverity, const char* tag, const char* file,
                  unsigned int line, const char* message);

 private:
  LogId default_log_id_;
};
#endif

// Configure logging based on ANDROID_LOG_TAGS environment variable.
// We need to parse a string that looks like
//
//      *:v jdwp:d dalvikvm:d dalvikvm-gc:i dalvikvmi:i
//
// The tag (or '*' for the global level) comes first, followed by a colon and a
// letter indicating the minimum priority level we're expected to log.  This can
// be used to reveal or conceal logs with specific tags.
#ifdef __ANDROID__
#define INIT_LOGGING_DEFAULT_LOGGER LogdLogger()
#else
#define INIT_LOGGING_DEFAULT_LOGGER StderrLogger
#endif
void InitLogging(char* argv[],
                 LogFunction&& logger = INIT_LOGGING_DEFAULT_LOGGER,
                 AbortFunction&& aborter = DefaultAborter);
#undef INIT_LOGGING_DEFAULT_LOGGER

// Replace the current logger.
void SetLogger(LogFunction&& logger);

// Replace the current aborter.
void SetAborter(AbortFunction&& aborter);

class ErrnoRestorer {
 public:
  ErrnoRestorer()
      : saved_errno_(errno) {
  }

  ~ErrnoRestorer() {
    errno = saved_errno_;
  }

  // Allow this object to be used as part of && operation.
  operator bool() const {
    return true;
  }

 private:
  const int saved_errno_;

  DISALLOW_COPY_AND_ASSIGN(ErrnoRestorer);
};

// A helper macro that produces an expression that accepts both a qualified name and an
// unqualified name for a LogSeverity, and returns a LogSeverity value.
// Note: DO NOT USE DIRECTLY. This is an implementation detail.
#define SEVERITY_LAMBDA(severity) ([&]() {    \
  using ::android::base::VERBOSE;             \
  using ::android::base::DEBUG;               \
  using ::android::base::INFO;                \
  using ::android::base::WARNING;             \
  using ::android::base::ERROR;               \
  using ::android::base::FATAL_WITHOUT_ABORT; \
  using ::android::base::FATAL;               \
  return (severity); }())

#ifdef __clang_analyzer__
// Clang's static analyzer does not see the conditional statement inside
// LogMessage's destructor that will abort on FATAL severity.
#define ABORT_AFTER_LOG_FATAL for (;; abort())

struct LogAbortAfterFullExpr {
  ~LogAbortAfterFullExpr() __attribute__((noreturn)) { abort(); }
  explicit operator bool() const { return false; }
};
// Provides an expression that evaluates to the truthiness of `x`, automatically
// aborting if `c` is true.
#define ABORT_AFTER_LOG_EXPR_IF(c, x) (((c) && ::android::base::LogAbortAfterFullExpr()) || (x))
// Note to the static analyzer that we always execute FATAL logs in practice.
#define MUST_LOG_MESSAGE(severity) (SEVERITY_LAMBDA(severity) == ::android::base::FATAL)
#else
#define ABORT_AFTER_LOG_FATAL
#define ABORT_AFTER_LOG_EXPR_IF(c, x) (x)
#define MUST_LOG_MESSAGE(severity) false
#endif
#define ABORT_AFTER_LOG_FATAL_EXPR(x) ABORT_AFTER_LOG_EXPR_IF(true, x)

// Defines whether the given severity will be logged or silently swallowed.
#define WOULD_LOG(severity) \
  (UNLIKELY((SEVERITY_LAMBDA(severity)) >= ::android::base::GetMinimumLogSeverity()) || \
   MUST_LOG_MESSAGE(severity))

// Get an ostream that can be used for logging at the given severity and to the default
// destination.
//
// Notes:
// 1) This will not check whether the severity is high enough. One should use WOULD_LOG to filter
//    usage manually.
// 2) This does not save and restore errno.
#define LOG_STREAM(severity) LOG_STREAM_TO(DEFAULT, severity)

// Get an ostream that can be used for logging at the given severity and to the
// given destination. The same notes as for LOG_STREAM apply.
#define LOG_STREAM_TO(dest, severity)                                           \
  ::android::base::LogMessage(__FILE__, __LINE__, ::android::base::dest,        \
                              SEVERITY_LAMBDA(severity), _LOG_TAG_INTERNAL, -1) \
      .stream()

// Logs a message to logcat on Android otherwise to stderr. If the severity is
// FATAL it also causes an abort. For example:
//
//     LOG(FATAL) << "We didn't expect to reach here";
#define LOG(severity) LOG_TO(DEFAULT, severity)

// Checks if we want to log something, and sets up appropriate RAII objects if
// so.
// Note: DO NOT USE DIRECTLY. This is an implementation detail.
#define LOGGING_PREAMBLE(severity)                                                         \
  (WOULD_LOG(severity) &&                                                                  \
   ABORT_AFTER_LOG_EXPR_IF((SEVERITY_LAMBDA(severity)) == ::android::base::FATAL, true) && \
   ::android::base::ErrnoRestorer())

// Logs a message to logcat with the specified log ID on Android otherwise to
// stderr. If the severity is FATAL it also causes an abort.
// Use an expression here so we can support the << operator following the macro,
// like "LOG(DEBUG) << xxx;".
#define LOG_TO(dest, severity) LOGGING_PREAMBLE(severity) && LOG_STREAM_TO(dest, severity)

// A variant of LOG that also logs the current errno value. To be used when
// library calls fail.
#define PLOG(severity) PLOG_TO(DEFAULT, severity)

// Behaves like PLOG, but logs to the specified log ID.
#define PLOG_TO(dest, severity)                                                        \
  LOGGING_PREAMBLE(severity) &&                                                        \
      ::android::base::LogMessage(__FILE__, __LINE__, ::android::base::dest,           \
                                  SEVERITY_LAMBDA(severity), _LOG_TAG_INTERNAL, errno) \
          .stream()

// Marker that code is yet to be implemented.
#define UNIMPLEMENTED(level) \
  LOG(level) << __PRETTY_FUNCTION__ << " unimplemented "

// Check whether condition x holds and LOG(FATAL) if not. The value of the
// expression x is only evaluated once. Extra logging can be appended using <<
// after. For example:
//
//     CHECK(false == true) results in a log message of
//       "Check failed: false == true".
#define CHECK(x)                                                                 \
  LIKELY((x)) || ABORT_AFTER_LOG_FATAL_EXPR(false) ||                            \
      ::android::base::LogMessage(__FILE__, __LINE__, ::android::base::DEFAULT,  \
                                  ::android::base::FATAL, _LOG_TAG_INTERNAL, -1) \
              .stream()                                                          \
          << "Check failed: " #x << " "

// clang-format off
// Helper for CHECK_xx(x,y) macros.
#define CHECK_OP(LHS, RHS, OP)                                                                 \
  for (auto _values = ::android::base::MakeEagerEvaluator(LHS, RHS);                           \
       UNLIKELY(!(_values.lhs OP _values.rhs));                                                \
       /* empty */)                                                                            \
  ABORT_AFTER_LOG_FATAL                                                                        \
  ::android::base::LogMessage(__FILE__, __LINE__, ::android::base::DEFAULT,                    \
                              ::android::base::FATAL, _LOG_TAG_INTERNAL, -1)                   \
          .stream()                                                                            \
      << "Check failed: " << #LHS << " " << #OP << " " << #RHS << " (" #LHS "=" << _values.lhs \
      << ", " #RHS "=" << _values.rhs << ") "
// clang-format on

// Check whether a condition holds between x and y, LOG(FATAL) if not. The value
// of the expressions x and y is evaluated once. Extra logging can be appended
// using << after. For example:
//
//     CHECK_NE(0 == 1, false) results in
//       "Check failed: false != false (0==1=false, false=false) ".
#define CHECK_EQ(x, y) CHECK_OP(x, y, == )
#define CHECK_NE(x, y) CHECK_OP(x, y, != )
#define CHECK_LE(x, y) CHECK_OP(x, y, <= )
#define CHECK_LT(x, y) CHECK_OP(x, y, < )
#define CHECK_GE(x, y) CHECK_OP(x, y, >= )
#define CHECK_GT(x, y) CHECK_OP(x, y, > )

// clang-format off
// Helper for CHECK_STRxx(s1,s2) macros.
#define CHECK_STROP(s1, s2, sense)                                             \
  while (UNLIKELY((strcmp(s1, s2) == 0) != (sense)))                           \
    ABORT_AFTER_LOG_FATAL                                                      \
    ::android::base::LogMessage(__FILE__, __LINE__, ::android::base::DEFAULT,  \
                                ::android::base::FATAL, _LOG_TAG_INTERNAL, -1) \
        .stream()                                                              \
        << "Check failed: " << "\"" << (s1) << "\""                            \
        << ((sense) ? " == " : " != ") << "\"" << (s2) << "\""
// clang-format on

// Check for string (const char*) equality between s1 and s2, LOG(FATAL) if not.
#define CHECK_STREQ(s1, s2) CHECK_STROP(s1, s2, true)
#define CHECK_STRNE(s1, s2) CHECK_STROP(s1, s2, false)

// Perform the pthread function call(args), LOG(FATAL) on error.
#define CHECK_PTHREAD_CALL(call, args, what)                           \
  do {                                                                 \
    int rc = call args;                                                \
    if (rc != 0) {                                                     \
      errno = rc;                                                      \
      ABORT_AFTER_LOG_FATAL                                            \
      PLOG(FATAL) << #call << " failed for " << (what);                \
    }                                                                  \
  } while (false)

// CHECK that can be used in a constexpr function. For example:
//
//    constexpr int half(int n) {
//      return
//          DCHECK_CONSTEXPR(n >= 0, , 0)
//          CHECK_CONSTEXPR((n & 1) == 0),
//              << "Extra debugging output: n = " << n, 0)
//          n / 2;
//    }
#define CHECK_CONSTEXPR(x, out, dummy)                                     \
  (UNLIKELY(!(x)))                                                         \
      ? (LOG(FATAL) << "Check failed: " << #x out, dummy) \
      :

// DCHECKs are debug variants of CHECKs only enabled in debug builds. Generally
// CHECK should be used unless profiling identifies a CHECK as being in
// performance critical code.
#if defined(NDEBUG) && !defined(__clang_analyzer__)
static constexpr bool kEnableDChecks = false;
#else
static constexpr bool kEnableDChecks = true;
#endif

#define DCHECK(x) \
  if (::android::base::kEnableDChecks) CHECK(x)
#define DCHECK_EQ(x, y) \
  if (::android::base::kEnableDChecks) CHECK_EQ(x, y)
#define DCHECK_NE(x, y) \
  if (::android::base::kEnableDChecks) CHECK_NE(x, y)
#define DCHECK_LE(x, y) \
  if (::android::base::kEnableDChecks) CHECK_LE(x, y)
#define DCHECK_LT(x, y) \
  if (::android::base::kEnableDChecks) CHECK_LT(x, y)
#define DCHECK_GE(x, y) \
  if (::android::base::kEnableDChecks) CHECK_GE(x, y)
#define DCHECK_GT(x, y) \
  if (::android::base::kEnableDChecks) CHECK_GT(x, y)
#define DCHECK_STREQ(s1, s2) \
  if (::android::base::kEnableDChecks) CHECK_STREQ(s1, s2)
#define DCHECK_STRNE(s1, s2) \
  if (::android::base::kEnableDChecks) CHECK_STRNE(s1, s2)
#if defined(NDEBUG) && !defined(__clang_analyzer__)
#define DCHECK_CONSTEXPR(x, out, dummy)
#else
#define DCHECK_CONSTEXPR(x, out, dummy) CHECK_CONSTEXPR(x, out, dummy)
#endif

// Temporary class created to evaluate the LHS and RHS, used with
// MakeEagerEvaluator to infer the types of LHS and RHS.
template <typename LHS, typename RHS>
struct EagerEvaluator {
  constexpr EagerEvaluator(LHS l, RHS r) : lhs(l), rhs(r) {
  }
  LHS lhs;
  RHS rhs;
};

// Helper function for CHECK_xx.
template <typename LHS, typename RHS>
constexpr EagerEvaluator<LHS, RHS> MakeEagerEvaluator(LHS lhs, RHS rhs) {
  return EagerEvaluator<LHS, RHS>(lhs, rhs);
}

// Explicitly instantiate EagerEvalue for pointers so that char*s aren't treated
// as strings. To compare strings use CHECK_STREQ and CHECK_STRNE. We rely on
// signed/unsigned warnings to protect you against combinations not explicitly
// listed below.
#define EAGER_PTR_EVALUATOR(T1, T2)               \
  template <>                                     \
  struct EagerEvaluator<T1, T2> {                 \
    EagerEvaluator(T1 l, T2 r)                    \
        : lhs(reinterpret_cast<const void*>(l)),  \
          rhs(reinterpret_cast<const void*>(r)) { \
    }                                             \
    const void* lhs;                              \
    const void* rhs;                              \
  }
EAGER_PTR_EVALUATOR(const char*, const char*);
EAGER_PTR_EVALUATOR(const char*, char*);
EAGER_PTR_EVALUATOR(char*, const char*);
EAGER_PTR_EVALUATOR(char*, char*);
EAGER_PTR_EVALUATOR(const unsigned char*, const unsigned char*);
EAGER_PTR_EVALUATOR(const unsigned char*, unsigned char*);
EAGER_PTR_EVALUATOR(unsigned char*, const unsigned char*);
EAGER_PTR_EVALUATOR(unsigned char*, unsigned char*);
EAGER_PTR_EVALUATOR(const signed char*, const signed char*);
EAGER_PTR_EVALUATOR(const signed char*, signed char*);
EAGER_PTR_EVALUATOR(signed char*, const signed char*);
EAGER_PTR_EVALUATOR(signed char*, signed char*);

// Data for the log message, not stored in LogMessage to avoid increasing the
// stack size.
class LogMessageData;

// A LogMessage is a temporarily scoped object used by LOG and the unlikely part
// of a CHECK. The destructor will abort if the severity is FATAL.
class LogMessage {
 public:
  LogMessage(const char* file, unsigned int line, LogId id, LogSeverity severity, const char* tag,
             int error);

  ~LogMessage();

  // Returns the stream associated with the message, the LogMessage performs
  // output when it goes out of scope.
  std::ostream& stream();

  // The routine that performs the actual logging.
  static void LogLine(const char* file, unsigned int line, LogId id, LogSeverity severity,
                      const char* tag, const char* msg);

 private:
  const std::unique_ptr<LogMessageData> data_;

  // TODO(b/35361699): remove these symbols once all prebuilds stop using it.
  LogMessage(const char* file, unsigned int line, LogId id, LogSeverity severity, int error);
  static void LogLine(const char* file, unsigned int line, LogId id, LogSeverity severity,
                      const char* msg);

  DISALLOW_COPY_AND_ASSIGN(LogMessage);
};

// Get the minimum severity level for logging.
LogSeverity GetMinimumLogSeverity();

// Set the minimum severity level for logging, returning the old severity.
LogSeverity SetMinimumLogSeverity(LogSeverity new_severity);

// Allows to temporarily change the minimum severity level for logging.
class ScopedLogSeverity {
 public:
  explicit ScopedLogSeverity(LogSeverity level);
  ~ScopedLogSeverity();

 private:
  LogSeverity old_;
};

}  // namespace base
}  // namespace android

namespace std {

// Emit a warning of ostream<< with std::string*. The intention was most likely to print *string.
//
// Note: for this to work, we need to have this in a namespace.
// Note: lots of ifdef magic to make this work with Clang (platform) vs GCC (windows tools)
// Note: using diagnose_if(true) under Clang and nothing under GCC/mingw as there is no common
//       attribute support.
// Note: using a pragma because "-Wgcc-compat" (included in "-Weverything") complains about
//       diagnose_if.
// Note: to print the pointer, use "<< static_cast<const void*>(string_pointer)" instead.
// Note: a not-recommended alternative is to let Clang ignore the warning by adding
//       -Wno-user-defined-warnings to CPPFLAGS.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#define OSTREAM_STRING_POINTER_USAGE_WARNING \
    __attribute__((diagnose_if(true, "Unexpected logging of string pointer", "warning")))
#else
#define OSTREAM_STRING_POINTER_USAGE_WARNING /* empty */
#endif
inline std::ostream& operator<<(std::ostream& stream, const std::string* string_pointer)
    OSTREAM_STRING_POINTER_USAGE_WARNING {
  return stream << static_cast<const void*>(string_pointer);
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#undef OSTREAM_STRING_POINTER_USAGE_WARNING

}  // namespace std

#endif  // ANDROID_BASE_LOGGING_H
