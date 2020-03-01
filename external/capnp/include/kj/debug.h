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
// * For every macro `FOO` above, there is a `DFOO` version (or `RECOVERABLE_DFOO`) which is only
//   executed in debug mode, i.e. when KJ_DEBUG is defined.  KJ_DEBUG is defined automatically
//   by common.h when compiling without optimization (unless NDEBUG is defined), but you can also
//   define it explicitly (e.g. -DKJ_DEBUG).  Generally, production builds should NOT use KJ_DEBUG
//   as it may enable expensive checks that are unlikely to fail.

#ifndef KJ_DEBUG_H_
#define KJ_DEBUG_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "string.h"
#include "exception.h"

#ifdef ERROR
// This is problematic because windows.h #defines ERROR, which we use in an enum here.
#error "Make sure to to undefine ERROR (or just #include <kj/windows-sanity.h>) before this file"
#endif

namespace kj {

#if _MSC_VER
// MSVC does __VA_ARGS__ differently from GCC:
// - A trailing comma before an empty __VA_ARGS__ is removed automatically, whereas GCC wants
//   you to request this behavior with "##__VA_ARGS__".
// - If __VA_ARGS__ is passed directly as an argument to another macro, it will be treated as a
//   *single* argument rather than an argument list. This can be worked around by wrapping the
//   outer macro call in KJ_EXPAND(), which appraently forces __VA_ARGS__ to be expanded before
//   the macro is evaluated. I don't understand the C preprocessor.
// - Using "#__VA_ARGS__" to stringify __VA_ARGS__ expands to zero tokens when __VA_ARGS__ is
//   empty, rather than expanding to an empty string literal. We can work around by concatenating
//   with an empty string literal.

#define KJ_EXPAND(X) X

#define KJ_LOG(severity, ...) \
  if (!::kj::_::Debug::shouldLog(::kj::LogSeverity::severity)) {} else \
    ::kj::_::Debug::log(__FILE__, __LINE__, ::kj::LogSeverity::severity, \
                        "" #__VA_ARGS__, __VA_ARGS__)

#define KJ_DBG(...) KJ_EXPAND(KJ_LOG(DBG, __VA_ARGS__))

#define KJ_REQUIRE(cond, ...) \
  if (KJ_LIKELY(cond)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                                 #cond, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

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

#if _WIN32

#define KJ_WIN32(call, ...) \
  if (::kj::_::Debug::isWin32Success(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             ::kj::_::Debug::getWin32Error(), #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_WINSOCK(call, ...) \
  if ((call) != SOCKET_ERROR) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             ::kj::_::Debug::getWin32Error(), #call, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

#define KJ_FAIL_WIN32(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           ::kj::_::Debug::Win32Error(errorNumber), code, "" #__VA_ARGS__, __VA_ARGS__);; f.fatal())

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
  if (!::kj::_::Debug::shouldLog(::kj::LogSeverity::severity)) {} else \
    ::kj::_::Debug::log(__FILE__, __LINE__, ::kj::LogSeverity::severity, \
                        #__VA_ARGS__, ##__VA_ARGS__)

#define KJ_DBG(...) KJ_LOG(DBG, ##__VA_ARGS__)

#define KJ_REQUIRE(cond, ...) \
  if (KJ_LIKELY(cond)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                                 #cond, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

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

#if _WIN32

#define KJ_WIN32(call, ...) \
  if (::kj::_::Debug::isWin32Success(call)) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             ::kj::_::Debug::getWin32Error(), #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_WINSOCK(call, ...) \
  if ((call) != SOCKET_ERROR) {} else \
    for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
             ::kj::_::Debug::getWin32Error(), #call, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

#define KJ_FAIL_WIN32(code, errorNumber, ...) \
  for (::kj::_::Debug::Fault f(__FILE__, __LINE__, \
           ::kj::_::Debug::Win32Error(errorNumber), code, #__VA_ARGS__, ##__VA_ARGS__);; f.fatal())

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

#define KJ_REQUIRE_NONNULL(value, ...) \
  (*({ \
    auto _kj_result = ::kj::_::readMaybe(value); \
    if (KJ_UNLIKELY(!_kj_result)) { \
      ::kj::_::Debug::Fault(__FILE__, __LINE__, ::kj::Exception::Type::FAILED, \
                            #value " != nullptr", #__VA_ARGS__, ##__VA_ARGS__).fatal(); \
    } \
    kj::mv(_kj_result); \
  }))

#define KJ_EXCEPTION(type, ...) \
  ::kj::Exception(::kj::Exception::Type::type, __FILE__, __LINE__, \
      ::kj::_::Debug::makeDescription(#__VA_ARGS__, ##__VA_ARGS__))

#endif

#define KJ_SYSCALL_HANDLE_ERRORS(call) \
  if (int _kjSyscallError = ::kj::_::Debug::syscallError([&](){return (call);}, false)) \
    switch (int error = _kjSyscallError)
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

#define KJ_ASSERT KJ_REQUIRE
#define KJ_FAIL_ASSERT KJ_FAIL_REQUIRE
#define KJ_ASSERT_NONNULL KJ_REQUIRE_NONNULL
// Use "ASSERT" in place of "REQUIRE" when the problem is local to the immediate surrounding code.
// That is, if the assert ever fails, it indicates that the immediate surrounding code is broken.

#ifdef KJ_DEBUG
#define KJ_DLOG KJ_LOG
#define KJ_DASSERT KJ_ASSERT
#define KJ_DREQUIRE KJ_REQUIRE
#else
#define KJ_DLOG(...) do {} while (false)
#define KJ_DASSERT(...) do {} while (false)
#define KJ_DREQUIRE(...) do {} while (false)
#endif

namespace _ {  // private

class Debug {
public:
  Debug() = delete;

  typedef LogSeverity Severity;  // backwards-compatibility

#if _WIN32
  struct Win32Error {
    // Hack for overloading purposes.
    uint number;
    inline explicit Win32Error(uint number): number(number) {}
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
#if _WIN32
    Fault(const char* file, int line, Win32Error osErrorNumber,
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
#if _WIN32
    void init(const char* file, int line, Win32Error osErrorNumber,
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

#if _WIN32
  static bool isWin32Success(int boolean);
  static bool isWin32Success(void* handle);
  static Win32Error getWin32Error();
#endif

  class Context: public ExceptionCallback {
  public:
    Context();
    KJ_DISALLOW_COPY(Context);
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
    KJ_DISALLOW_COPY(ContextImpl);

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

#if _WIN32
inline Debug::Fault::Fault(const char* file, int line, Win32Error osErrorNumber,
                           const char* condition, const char* macroArgs)
    : exception(nullptr) {
  init(file, line, osErrorNumber, condition, macroArgs, nullptr);
}

inline bool Debug::isWin32Success(int boolean) {
  return boolean;
}
inline bool Debug::isWin32Success(void* handle) {
  // Assume null and INVALID_HANDLE_VALUE mean failure.
  return handle != nullptr && handle != (void*)-1;
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

}  // namespace _ (private)
}  // namespace kj

#endif  // KJ_DEBUG_H_
