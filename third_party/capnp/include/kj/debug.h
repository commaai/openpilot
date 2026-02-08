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

// This file declares convenient macros for debug logging and error handling.  The macros make
// it excessively easy to extract useful context information from code.  Example:
//
//     KJ_ASSERT(a == b, a, b, "a and b must be the same.");
//
// On failure, this will throw an exception whose description looks like:
//
//     myfile.c++:43: bug in code: expected a == b; a = 14; b = 72; a and b must be the same.
//
// As you can see, all arguments after the first provide additional context.
//
// The macros available are:
//
// * `KJ_LOG(severity, ...)`:  Just writes a log message, to stderr by default (but you can
//   intercept messages by implementing an ExceptionCallback).  `severity` is `INFO`, `WARNING`,
//   `ERROR`, or `FATAL`.  By default, `INFO` logs are not written, but for command-line apps the
//   user should be able to pass a flag like `--verbose` to enable them.  Other log levels are
//   enabled by default.  Log messages -- like exceptions -- can be intercepted by registering an
//   ExceptionCallback.
//
// * `KJ_DBG(...)`:  Like `KJ_LOG`, but intended specifically for temporary log lines added while
//   debugging a particular problem.  Calls to `KJ_DBG` should always be deleted before committing
//   code.  It is suggested that you set up a pre-commit hook that checks for this.
//
// * `KJ_ASSERT(condition, ...)`:  Throws an exception if `condition` is false, or aborts if
//   exceptions are disabled.  This macro should be used to check for bugs in the surrounding code
//   and its dependencies, but NOT to check for invalid input.  The macro may be followed by a
//   brace-delimited code block; if so, the block will be executed in the case where the assertion
//   fails, before throwing the exception.  If control jumps out of the block (e.g. with "break",
//   "return", or "goto"), then the error is considered "recoverable" -- in this case, if
//   exceptions are disabled, execution will continue normally rather than aborting (but if
//   exceptions are enabled, an exception will still be thrown on exiting the block). A "break"
//   statement in particular will jump to the code immediately after the block (it does not break
//   any surrounding loop or switch).  Example:
//
//       KJ_ASSERT(value >= 0, "Value cannot be negative.", value) {
//         // Assertion failed.  Set value to zero to "recover".
//         value = 0;
//         // Don't abort if exceptions are disabled.  Continue normally.
//         // (Still throw an exception if they are enabled, though.)
//         break;
//       }
//       // When exceptions are disabled, we'll get here even if the assertion fails.
//       // Otherwise, we get here only if the assertion passes.
//
// * `KJ_REQUIRE(condition, ...)`:  Like `KJ_ASSERT` but used to check preconditions -- e.g. to
//   validate parameters passed from a caller.  A failure indicates that the caller is buggy.
//
// * `KJ_ASSUME(condition, ...)`: Like `KJ_ASSERT`, but in release mode (if KJ_DEBUG is not
//   defined; see below) instead warrants to the compiler that the condition can be assumed to
//   hold, allowing it to optimize accordingly.  This can result in undefined behavior, so use
//   this macro *only* if you can prove to your satisfaction that the condition is guaranteed by
//   surrounding code, and if the condition failing to hold would in any case result in undefined
//   behavior in its dependencies.
//
// * `KJ_SYSCALL(code, ...)`:  Executes `code` assuming it makes a system call.  A negative result
//   is considered an error, with error code reported via `errno`.  EINTR is handled by retrying.
//   Other errors are handled by throwing an exception.  If you need to examine the return code,
//   assign it to a variable like so:
//
//       int fd;
//       KJ_SYSCALL(fd = open(filename, O_RDONLY), filename);
//
//   `KJ_SYSCALL` can be followed by a recovery block, just like `KJ_ASSERT`.
//
// * `KJ_NONBLOCKING_SYSCALL(code, ...)`:  Like KJ_SYSCALL, but will not throw an exception on
//   EAGAIN/EWOULDBLOCK.  The calling code should check the syscall's return value to see if it
//   indicates an error; in this case, it can assume the error was EAGAIN because any other error
//   would have caused an exception to be thrown.
//
// * `KJ_CONTEXT(...)`:  Notes additional contextual information relevant to any exceptions thrown
//   from within the current scope.  That is, until control exits the block in which KJ_CONTEXT()
//   is used, if any exception is generated, it will contain the given information in its context
//   chain.  This is helpful because it can otherwise be very difficult to come up with error
//   messages that make sense within low-level helper code.  Note that the parameters to
//   KJ_CONTEXT() are only evaluated if an exception is thrown.  This implies that any variables
//   used must remain valid until the end of the scope.
//
// Notes:
// * Do not write expressions with side-effects in the message content part of the macro, as the
//   message will not necessarily be evaluated.
// * For every macro `FOO` above except `LOG`, there is also a `FAIL_FOO` macro used to report
//   failures that already happened.  For the macros that check a boolean condition, `FAIL_FOO`
//   omits the first parameter and behaves like it was `false`.  `FAIL_SYSCALL` and
//   `FAIL_RECOVERABLE_SYSCALL` take a string and an OS error number as the first two parameters.
//   The string should be the name of the failed system call.
// * For every macro `FOO` above except `ASSUME`, there is a `DFOO` version (or
//   `RECOVERABLE_DFOO`) which is only executed in debug mode, i.e. when KJ_DEBUG is defined.
//   KJ_DEBUG is defined automatically by common.h when compiling without optimization (unless
//   NDEBUG is defined), but you can also define it explicitly (e.g. -DKJ_DEBUG).  Generally,
//   production builds should NOT use KJ_DEBUG as it may enable expensive checks that are unlikely
//   to fail.

#pragma once

#include "string.h"
#include "exception.h"
#include "windows-sanity.h"  // work-around macro conflict with `ERROR`

KJ_BEGIN_HEADER

namespace kj {

#if KJ_MSVC_TRADITIONAL_CPP
// MSVC does __VA_ARGS__ differently from GCC:
// - A trailing comma before an empty __VA_ARGS__ is removed automatically, whereas GCC wants
//   you to request this behavior with "##__VA_ARGS__".
// - If __VA_ARGS__ is passed directly as an argument to another macro, it will be treated as a
//   *single* argument rather than an argument list. This can be worked around by wrapping the
//   outer macro call in KJ_EXPAND(), which apparently forces __VA_ARGS__ to be expanded before
//   the macro is evaluated. I don't understand the C preprocessor.
// - Using "#__VA_ARGS__" to stringify __VA_ARGS__ expands to zero tokens when __VA_ARGS__ is
//   empty, rather than expanding to an empty string literal. We can work around by concatenating
//   with an empty string literal.

#define KJ_EXPAND(X) X

#define KJ_LOG(severity, ...) \
  for (bool _kj_shouldLog = ::kj::_::Debug::shouldLog(::kj::LogSeverity::severity); \
       _kj_shouldLog; _kj_shouldLog = false) \
    ::kj::_::Debug::log(__FILE__, __LINE__, ::kj::LogSeverity::severity, \
                        "" #__VA_ARGS__, __VA_ARGS__)

#define KJ_DBG(...) KJ_EXPAND(KJ_LOG(DBG, __VA_ARGS__))

#define KJ_REQUIRE(cond, ...) \
  if (auto _kjCondition = ::kj::_::MAGIC_ASSERT << cond) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
        #cond, "_kjCondition," #__VA_ARGS__, _kjCondition, __VA_ARGS__);; f.fatal())

#define KJ_FAIL_REQUIRE(...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                               nullptr, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_SYSCALL(call, ...) \
  if (auto _kjSyscallResult = ::kj::_::Debug::syscall([&](){return (call);}, false)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjSyscallResult.getErrorNumber(), #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_NONBLOCKING_SYSCALL(call, ...) \
  if (auto _kjSyscallResult = ::kj::_::Debug::syscall([&](){return (call);}, true)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjSyscallResult.getErrorNumber(), #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_FAIL_SYSCALL(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           errorNumber, code, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#if _WIN32 || __CYGWIN__

#define KJ_WIN32(call, ...) \
  if (auto _kjWin32Result = ::kj::_::Debug::win32Call(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjWin32Result, #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_WINSOCK(call, ...) \
  if (auto _kjWin32Result = ::kj::_::Debug::winsockCall(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjWin32Result, #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_FAIL_WIN32(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           ::kj::_::Debug::Win32Result(errorNumber), code, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#endif

#define KJ_UNIMPLEMENTED(...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::UNIMPLEMENTED, \
                               nullptr, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

// TODO(msvc):  MSVC mis-deduces `ContextImpl<decltype(func)>` as `ContextImpl<int>` in some edge
// cases, such as inside nested lambdas inside member functions. Wrapping the type in
// `decltype(instance<...>())` helps it deduce the context function's type correctly.
#define KJ_CONTEXT(...) \
  auto KJ_UNIQUE_NAME(_kjContextFunc) = [&]() -> ::kj::_::Debug::Context::Value { \
        return ::kj::_::Debug::Context::Value(__FILE__, __LINE__, \
            ::kj::_::Debug::makeDescription("" #__VA_ARGS__, __VA_ARGS__)); \
      }; \
  decltype(::kj::instance<::kj::_::Debug::ContextImpl<decltype(KJ_UNIQUE_NAME(_kjContextFunc))>>()) \
      KJ_UNIQUE_NAME(_kjContext)(KJ_UNIQUE_NAME(_kjContextFunc))

#define KJ_REQUIRE_NONNULL(value, ...) \
  (*[&] { \
    auto _kj_result = ::kj::_::readMaybe(value); \
    if (KJ_UNLIKELY(!_kj_result)) { \
      ::kj::_::Debug::Fault(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                            #value " != nullptr", "" #__VA_ARGS__, __VA_ARGS__).fatal(); \
    } \
    return _kj_result; \
  }())

#define KJ_EXCEPTION(type, ...) \
  ::kj::Exception(::kj::Exception::Type::type, __FILE__, __LINE__, \
      ::kj::_::Debug::makeDescription("" #__VA_ARGS__, __VA_ARGS__))

#else

#define KJ_LOG(severity, ...) \
  for (bool _kj_shouldLog = ::kj::_::Debug::shouldLog(::kj::LogSeverity::severity); \
       _kj_shouldLog; _kj_shouldLog = false) \
    ::kj::_::Debug::log(__FILE__, __LINE__, ::kj::LogSeverity::severity, \
                        #__VA_ARGS__, ##__VA_ARGS__)

#define KJ_DBG(...) KJ_LOG(DBG, ##__VA_ARGS__)

#define KJ_REQUIRE(cond, ...) \
  if (auto _kjCondition = ::kj::_::MAGIC_ASSERT << cond) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
        #cond, "_kjCondition," #__VA_ARGS__, _kjCondition, ##__VA_ARGS__);; f.fatal())

#define KJ_FAIL_REQUIRE(...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                               nullptr, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_SYSCALL(call, ...) \
  if (auto _kjSyscallResult = ::kj::_::Debug::syscall([&](){return (call);}, false)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjSyscallResult.getErrorNumber(), #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_NONBLOCKING_SYSCALL(call, ...) \
  if (auto _kjSyscallResult = ::kj::_::Debug::syscall([&](){return (call);}, true)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjSyscallResult.getErrorNumber(), #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_FAIL_SYSCALL(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           errorNumber, code, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#if _WIN32 || __CYGWIN__

#define KJ_WIN32(call, ...) \
  if (auto _kjWin32Result = ::kj::_::Debug::win32Call(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjWin32Result, #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())
// Invoke a Win32 syscall that returns either BOOL or HANDLE, and throw an exception if it fails.

#define KJ_WINSOCK(call, ...) \
  if (auto _kjWin32Result = ::kj::_::Debug::winsockCall(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             _kjWin32Result, #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())
// Like KJ_WIN32 but for winsock calls which return `int` with SOCKET_ERROR indicating failure.
//
// Unfortunately, it's impossible to distinguish these from BOOL-returning Win32 calls by type,
// since BOOL is in fact an alias for `int`. :(

#define KJ_FAIL_WIN32(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           ::kj::_::Debug::Win32Result(errorNumber), code, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#endif

#define KJ_UNIMPLEMENTED(...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::UNIMPLEMENTED, \
                               nullptr, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_CONTEXT(...) \
  auto KJ_UNIQUE_NAME(_kjContextFunc) = [&]() -> ::kj::_::Debug::Context::Value { \
        return ::kj::_::Debug::Context::Value(__FILE__, __LINE__, \
            ::kj::_::Debug::makeDescription(#__VA_ARGS__, ##__VA_ARGS__)); \
      }; \
  ::kj::_::Debug::ContextImpl<decltype(KJ_UNIQUE_NAME(_kjContextFunc))> \
      KJ_UNIQUE_NAME(_kjContext)(KJ_UNIQUE_NAME(_kjContextFunc))

#if _MSC_VER && !defined(__clang__)

#define KJ_REQUIRE_NONNULL(value, ...) \
  (*([&] { \
    auto _kj_result = ::kj::_::readMaybe(value); \
    if (KJ_UNLIKELY(!_kj_result)) { \
      ::kj::_::Debug::Fault(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                            #value " != nullptr", #__VA_ARGS__, ##__VA_ARGS__).fatal(); \
    } \
    return _kj_result; \
  }()))

#else

#define KJ_REQUIRE_NONNULL(value, ...) \
  (*({ \
    auto _kj_result = ::kj::_::readMaybe(value); \
    if (KJ_UNLIKELY(!_kj_result)) { \
      ::kj::_::Debug::Fault(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                            #value " != nullptr", #__VA_ARGS__, ##__VA_ARGS__).fatal(); \
    } \
    kj::mv(_kj_result); \
  }))

#endif

#define KJ_EXCEPTION(type, ...) \
  ::kj::Exception(::kj::Exception::Type::type, __FILE__, __LINE__, \
      ::kj::_::Debug::makeDescription(#__VA_ARGS__, ##__VA_ARGS__))

#endif

#define KJ_SYSCALL_HANDLE_ERRORS(call) \
  if (int _kjSyscallError = ::kj::_::Debug::syscallError([&](){return (call);}, false)) \
    switch (int error KJ_UNUSED = _kjSyscallError)
// Like KJ_SYSCALL, but doesn't throw. Instead, the block after the macro is a switch block on the
// error. Additionally, the int value `error` is defined within the block. So you can do:
//
//     KJ_SYSCALL_HANDLE_ERRORS(foo()) {
//       case ENOENT:
//         handleNoSuchFile();
//         break;
//       case EEXIST:
//         handleExists();
//         break;
//       default:
//         KJ_FAIL_SYSCALL("foo()", error);
//     } else {
//       handleSuccessCase();
//     }

#if _WIN32 || __CYGWIN__

#define KJ_WIN32_HANDLE_ERRORS(call) \
  if (uint _kjWin32Error = ::kj::_::Debug::win32Call(call).number) \
    switch (uint error KJ_UNUSED = _kjWin32Error)
// Like KJ_WIN32, but doesn't throw. Instead, the block after the macro is a switch block on the
// error. Additionally, the int value `error` is defined within the block. So you can do:
//
//     KJ_SYSCALL_HANDLE_ERRORS(foo()) {
//       case ERROR_FILE_NOT_FOUND:
//         handleNoSuchFile();
//         break;
//       case ERROR_FILE_EXISTS:
//         handleExists();
//         break;
//       default:
//         KJ_FAIL_WIN32("foo()", error);
//     } else {
//       handleSuccessCase();
//     }

#endif

#define KJ_ASSERT KJ_REQUIRE
#define KJ_FAIL_ASSERT KJ_FAIL_REQUIRE
#define KJ_ASSERT_NONNULL KJ_REQUIRE_NONNULL
// Use "ASSERT" in place of "REQUIRE" when the problem is local to the immediate surrounding code.
// That is, if the assert ever fails, it indicates that the immediate surrounding code is broken.

#ifdef KJ_DEBUG
#define KJ_DLOG KJ_LOG
#define KJ_DASSERT KJ_ASSERT
#define KJ_DREQUIRE KJ_REQUIRE
#define KJ_ASSUME KJ_ASSERT
#else
#define KJ_DLOG(...) do {} while (false)
#define KJ_DASSERT(...) do {} while (false)
#define KJ_DREQUIRE(...) do {} while (false)
#if defined(__GNUC__)
#define KJ_ASSUME(cond, ...) do { if (cond) {} else __builtin_unreachable(); } while (false)
#elif defined(__clang__)
#define KJ_ASSUME(cond, ...) __builtin_assume(cond)
#elif defined(_MSC_VER)
#define KJ_ASSUME(cond, ...) __assume(cond)
#else
#define KJ_ASSUME(...) do {} while (false)
#endif

#endif

namespace _ {  // private

class Debug {
public:
  Debug() = delete;

  typedef LogSeverity Severity;  // backwards-compatibility

#if _WIN32 || __CYGWIN__
  struct Win32Result {
    uint number;
    inline explicit Win32Result(uint number): number(number) {}
    operator bool() const { return number == 0; }
  };
#endif

  static inline bool shouldLog(LogSeverity severity) { return severity >= minSeverity; }
  // Returns whether messages of the given severity should be logged.

  static inline void setLogLevel(LogSeverity severity) { minSeverity = severity; }
  // Set the minimum message severity which will be logged.
  //
  // TODO(someday):  Expose publicly.

  template <typename... Params>
  static void log(const char* file, int line, LogSeverity severity, const char* macroArgs,
                  Params&&... params);

  class Fault {
  public:
    template <typename Code, typename... Params>
    Fault(const char* file, int line, Code code,
          const char* condition, const char* macroArgs, Params&&... params);
    Fault(const char* file, int line, Exception::Type type,
          const char* condition, const char* macroArgs);
    Fault(const char* file, int line, int osErrorNumber,
          const char* condition, const char* macroArgs);
#if _WIN32 || __CYGWIN__
    Fault(const char* file, int line, Win32Result osErrorNumber,
          const char* condition, const char* macroArgs);
#endif
    ~Fault() noexcept(false);

    KJ_NOINLINE KJ_NORETURN(void fatal());
    // Throw the exception.

  private:
    void init(const char* file, int line, Exception::Type type,
              const char* condition, const char* macroArgs, ArrayPtr<String> argValues);
    void init(const char* file, int line, int osErrorNumber,
              const char* condition, const char* macroArgs, ArrayPtr<String> argValues);
#if _WIN32 || __CYGWIN__
    void init(const char* file, int line, Win32Result osErrorNumber,
              const char* condition, const char* macroArgs, ArrayPtr<String> argValues);
#endif

    Exception* exception;
  };

  class SyscallResult {
  public:
    inline SyscallResult(int errorNumber): errorNumber(errorNumber) {}
    inline operator void*() { return errorNumber == 0 ? this : nullptr; }
    inline int getErrorNumber() { return errorNumber; }

  private:
    int errorNumber;
  };

  template <typename Call>
  static SyscallResult syscall(Call&& call, bool nonblocking);
  template <typename Call>
  static int syscallError(Call&& call, bool nonblocking);

#if _WIN32 || __CYGWIN__
  static Win32Result win32Call(int boolean);
  static Win32Result win32Call(void* handle);
  static Win32Result winsockCall(int result);
  static uint getWin32ErrorCode();
#endif

  class Context: public ExceptionCallback {
  public:
    Context();
    KJ_DISALLOW_COPY_AND_MOVE(Context);
    virtual ~Context() noexcept(false);

    struct Value {
      const char* file;
      int line;
      String description;

      inline Value(const char* file, int line, String&& description)
          : file(file), line(line), description(mv(description)) {}
    };

    virtual Value evaluate() = 0;

    virtual void onRecoverableException(Exception&& exception) override;
    virtual void onFatalException(Exception&& exception) override;
    virtual void logMessage(LogSeverity severity, const char* file, int line, int contextDepth,
                            String&& text) override;

  private:
    bool logged;
    Maybe<Value> value;

    Value ensureInitialized();
  };

  template <typename Func>
  class ContextImpl: public Context {
  public:
    inline ContextImpl(Func& func): func(func) {}
    KJ_DISALLOW_COPY_AND_MOVE(ContextImpl);

    Value evaluate() override {
      return func();
    }
  private:
    Func& func;
  };

  template <typename... Params>
  static String makeDescription(const char* macroArgs, Params&&... params);

private:
  static LogSeverity minSeverity;

  static void logInternal(const char* file, int line, LogSeverity severity, const char* macroArgs,
                          ArrayPtr<String> argValues);
  static String makeDescriptionInternal(const char* macroArgs, ArrayPtr<String> argValues);

  static int getOsErrorNumber(bool nonblocking);
  // Get the error code of the last error (e.g. from errno).  Returns -1 on EINTR.
};

template <typename... Params>
void Debug::log(const char* file, int line, LogSeverity severity, const char* macroArgs,
                Params&&... params) {
  String argValues[sizeof...(Params)] = {str(params)...};
  logInternal(file, line, severity, macroArgs, arrayPtr(argValues, sizeof...(Params)));
}

template <>
inline void Debug::log<>(const char* file, int line, LogSeverity severity, const char* macroArgs) {
  logInternal(file, line, severity, macroArgs, nullptr);
}

template <typename Code, typename... Params>
Debug::Fault::Fault(const char* file, int line, Code code,
                    const char* condition, const char* macroArgs, Params&&... params)
    : exception(nullptr) {
  String argValues[sizeof...(Params)] = {str(params)...};
  init(file, line, code, condition, macroArgs,
       arrayPtr(argValues, sizeof...(Params)));
}

inline Debug::Fault::Fault(const char* file, int line, int osErrorNumber,
                           const char* condition, const char* macroArgs)
    : exception(nullptr) {
  init(file, line, osErrorNumber, condition, macroArgs, nullptr);
}

inline Debug::Fault::Fault(const char* file, int line, kj::Exception::Type type,
                           const char* condition, const char* macroArgs)
    : exception(nullptr) {
  init(file, line, type, condition, macroArgs, nullptr);
}

#if _WIN32 || __CYGWIN__
inline Debug::Fault::Fault(const char* file, int line, Win32Result osErrorNumber,
                           const char* condition, const char* macroArgs)
    : exception(nullptr) {
  init(file, line, osErrorNumber, condition, macroArgs, nullptr);
}

inline Debug::Win32Result Debug::win32Call(int boolean) {
  return boolean ? Win32Result(0) : Win32Result(getWin32ErrorCode());
}
inline Debug::Win32Result Debug::win32Call(void* handle) {
  // Assume null and INVALID_HANDLE_VALUE mean failure.
  return win32Call(handle != nullptr && handle != (void*)-1);
}
inline Debug::Win32Result Debug::winsockCall(int result) {
  // Expect a return value of SOCKET_ERROR means failure.
  return win32Call(result != -1);
}
#endif

template <typename Call>
Debug::SyscallResult Debug::syscall(Call&& call, bool nonblocking) {
  while (call() < 0) {
    int errorNum = getOsErrorNumber(nonblocking);
    // getOsErrorNumber() returns -1 to indicate EINTR.
    // Also, if nonblocking is true, then it returns 0 on EAGAIN, which will then be treated as a
    // non-error.
    if (errorNum != -1) {
      return SyscallResult(errorNum);
    }
  }
  return SyscallResult(0);
}

template <typename Call>
int Debug::syscallError(Call&& call, bool nonblocking) {
  while (call() < 0) {
    int errorNum = getOsErrorNumber(nonblocking);
    // getOsErrorNumber() returns -1 to indicate EINTR.
    // Also, if nonblocking is true, then it returns 0 on EAGAIN, which will then be treated as a
    // non-error.
    if (errorNum != -1) {
      return errorNum;
    }
  }
  return 0;
}

template <typename... Params>
String Debug::makeDescription(const char* macroArgs, Params&&... params) {
  String argValues[sizeof...(Params)] = {str(params)...};
  return makeDescriptionInternal(macroArgs, arrayPtr(argValues, sizeof...(Params)));
}

template <>
inline String Debug::makeDescription<>(const char* macroArgs) {
  return makeDescriptionInternal(macroArgs, nullptr);
}

// =======================================================================================
// Magic Asserts!
//
// When KJ_ASSERT(foo == bar) fails, `foo` and `bar`'s actual values will be stringified in the
// error message. How does it work? We use template magic and operator precedence. The assertion
// actually evaluates something like this:
//
//     if (auto _kjCondition = kj::_::MAGIC_ASSERT << foo == bar)
//
// `<<` has operator precedence slightly above `==`, so `kj::_::MAGIC_ASSERT << foo` gets evaluated
// first. This wraps `foo` in a little wrapper that captures the comparison operators and keeps
// enough information around to be able to stringify the left and right sides of the comparison
// independently. As always, the stringification only actually occurs if the assert fails.
//
// You might ask why we use operator `<<` and not e.g. operator `<=`, since operators of the same
// precedence are evaluated left-to-right. The answer is that some compilers trigger all sorts of
// warnings when you seem to be using a comparison as the input to another comparison. The
// particular warning GCC produces is its general "-Wparentheses" warning which is broadly useful,
// so we don't want to disable it. `<<` also produces some warnings, but only on Clang and the
// specific warning is one we're comfortable disabling (see below). This does mean that we have to
// explicitly overload `operator<<` ourselves to make sure using it in an assert still works.
//
// You might also ask, if we're using operator `<<` anyway, why not start it from the right, in
// which case it would bind after computing any `<<` operators that were actually in the user's
// code? I tried this, but it resulted in a somewhat broader warning from clang that I felt worse
// about disabling (a warning about `<<` precedence not applying specifically to overloads) and
// also created ambiguous overload errors in the KJ units code.

#if __clang__
// We intentionally overload operator << for the specific purpose of evaluating it before
// evaluating comparison expressions, so stop Clang from warning about it. Unfortunately this means
// eliminating a warning that would otherwise be useful for people using iostreams... sorry.
#pragma GCC diagnostic ignored "-Woverloaded-shift-op-parentheses"
#endif

template <typename T>
struct DebugExpression;

template <typename T, typename = decltype(toCharSequence(instance<T&>()))>
inline auto tryToCharSequence(T* value) { return kj::toCharSequence(*value); }
inline StringPtr tryToCharSequence(...) { return "(can't stringify)"_kj; }
// SFINAE to stringify a value if and only if it can be stringified.

template <typename Left, typename Right>
struct DebugComparison {
  Left left;
  Right right;
  StringPtr op;
  bool result;

  inline operator bool() const { return KJ_LIKELY(result); }

  template <typename T> inline void operator&(T&& other) = delete;
  template <typename T> inline void operator^(T&& other) = delete;
  template <typename T> inline void operator|(T&& other) = delete;
};

template <typename Left, typename Right>
String KJ_STRINGIFY(DebugComparison<Left, Right>& cmp) {
  return _::concat(tryToCharSequence(&cmp.left), cmp.op, tryToCharSequence(&cmp.right));
}

template <typename T>
struct DebugExpression {
  DebugExpression(T&& value): value(kj::fwd<T>(value)) {}
  T value;

  // Handle comparison operations by constructing a DebugComparison value.
#define DEFINE_OPERATOR(OP) \
  template <typename U> \
  DebugComparison<T, U> operator OP(U&& other) { \
    bool result = value OP other; \
    return { kj::fwd<T>(value), kj::fwd<U>(other), " " #OP " "_kj, result }; \
  }
  DEFINE_OPERATOR(==);
  DEFINE_OPERATOR(!=);
  DEFINE_OPERATOR(<=);
  DEFINE_OPERATOR(>=);
  DEFINE_OPERATOR(< );
  DEFINE_OPERATOR(> );
#undef DEFINE_OPERATOR

  // Handle binary operators that have equal or lower precedence than comparisons by performing
  // the operation and wrapping the result.
#define DEFINE_OPERATOR(OP) \
  template <typename U> inline auto operator OP(U&& other) { \
    return DebugExpression<decltype(kj::fwd<T>(value) OP kj::fwd<U>(other))>(\
        kj::fwd<T>(value) OP kj::fwd<U>(other)); \
  }
  DEFINE_OPERATOR(<<);
  DEFINE_OPERATOR(>>);
  DEFINE_OPERATOR(&);
  DEFINE_OPERATOR(^);
  DEFINE_OPERATOR(|);
#undef DEFINE_OPERATOR

  inline operator bool() {
    // No comparison performed, we're just asserting the expression is truthy. This also covers
    // the case of the logic operators && and || -- we cannot overload those because doing so would
    // break short-circuiting behavior.
    return value;
  }
};

template <typename T>
StringPtr KJ_STRINGIFY(const DebugExpression<T>& exp) {
  // Hack: This will only ever be called in cases where the expression's truthiness was asserted
  //   directly, and was determined to be falsy.
  return "false"_kj;
}

struct DebugExpressionStart {
  template <typename T>
  DebugExpression<T> operator<<(T&& value) const {
    return DebugExpression<T>(kj::fwd<T>(value));
  }
};
static constexpr DebugExpressionStart MAGIC_ASSERT;

}  // namespace _ (private)
}  // namespace kj

KJ_END_HEADER
