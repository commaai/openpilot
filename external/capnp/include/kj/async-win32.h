// Copyright (c) 2016 Sandstorm Development Group, Inc. and contributors
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

#ifndef KJ_ASYNC_WIN32_H_
#define KJ_ASYNC_WIN32_H_

#if !_WIN32
#error "This file is Windows-specific. On Unix, include async-unix.h instead."
#endif

#include "async.h"
#include "time.h"
#include "io.h"
#include <atomic>
#include <inttypes.h>

// Include windows.h as lean as possible. (If you need more of the Windows API for your app,
// #include windows.h yourself before including this header.)
#define WIN32_LEAN_AND_MEAN 1
#define NOSERVICE 1
#define NOMCX 1
#define NOIME 1
#include <windows.h>
#include "windows-sanity.h"

namespace kj {

class Win32EventPort: public EventPort {
  // Abstract base interface for EventPorts that can listen on Win32 event types. Due to the
  // absurd complexity of the Win32 API, it's not possible to standardize on a single
  // implementation of EventPort. In particular, there is no way for a single thread to use I/O
  // completion ports (the most efficient way of handling I/O) while at the same time waiting for
  // signalable handles or UI messages.
  //
  // Note that UI messages are not supported at all by this interface because the message queue
  // is implemented by user32.dll and we want libkj to depend only on kernel32.dll. A separate
  // compat library could provide a Win32EventPort implementation that works with the UI message
  // queue.

public:
  // ---------------------------------------------------------------------------
  // overlapped I/O

  struct IoResult {
    DWORD errorCode;
    DWORD bytesTransferred;
  };

  class IoOperation {
  public:
    virtual LPOVERLAPPED getOverlapped() = 0;
    // Gets the OVERLAPPED structure to pass to the Win32 I/O call. Do NOT modify it; just pass it
    // on.

    virtual Promise<IoResult> onComplete() = 0;
    // After making the Win32 call, if the return value indicates that the operation was
    // successfully queued (i.e. the completion event will definitely occur), call this to wait
    // for completion.
    //
    // You MUST call this if the operation was successfully queued, and you MUST NOT call this
    // otherwise. If the Win32 call failed (without queuing any operation or event) then you should
    // simply drop the IoOperation object.
    //
    // Dropping the returned Promise cancels the operation via Win32's CancelIoEx(). The destructor
    // will wait for the cancellation to complete, such that after dropping the proimse it is safe
    // to free the buffer that the operation was reading from / writing to.
    //
    // You may safely drop the `IoOperation` while still waiting for this promise. You may not,
    // however, drop the `IoObserver`.
  };

  class IoObserver {
  public:
    virtual Own<IoOperation> newOperation(uint64_t offset) = 0;
    // Begin an I/O operation. For file operations, `offset` is the offset within the file at
    // which the operation will start. For stream operations, `offset` is ignored.
  };

  virtual Own<IoObserver> observeIo(HANDLE handle) = 0;
  // Given a handle which supports overlapped I/O, arrange to receive I/O completion events via
  // this EventPort.
  //
  // Different Win32EventPort implementations may handle this in different ways, such as by using
  // completion routines (APCs) or by using I/O completion ports. The caller should not assume
  // any particular technique.
  //
  // WARNING: It is only safe to call observeIo() on a particular handle once during its lifetime.
  //   You cannot observe the same handle from multiple Win32EventPorts, even if not at the same
  //   time. This is because the Win32 API provides no way to disassociate a handle from an I/O
  //   completion port once it is associated.

  // ---------------------------------------------------------------------------
  // signalable handles
  //
  // Warning: Due to limitations in the Win32 API, implementations of EventPort may be forced to
  //   spawn additional threads to wait for signaled objects. This is necessary if the EventPort
  //   implementation is based on I/O completion ports, or if you need to wait on more than 64
  //   handles at once.

  class SignalObserver {
  public:
    virtual Promise<void> onSignaled() = 0;
    // Returns a promise that completes the next time the handle enters the signaled state.
    //
    // Depending on the type of handle, the handle may automatically be reset to a non-signaled
    // state before the promise resolves. The underlying implementaiton uses WaitForSingleObject()
    // or an equivalent wait call, so check the documentation for that to understand the semantics.
    //
    // If the handle is a mutex and it is abandoned without being unlocked, the promise breaks with
    // an exception.

    virtual Promise<bool> onSignaledOrAbandoned() = 0;
    // Like onSingaled(), but instead of throwing when a mutex is abandoned, resolves to `true`.
    // Resolves to `false` for non-abandoned signals.
  };

  virtual Own<SignalObserver> observeSignalState(HANDLE handle) = 0;
  // Given a handle that supports waiting for it to become "signaled" via WaitForSingleObject(),
  // return an object that can wait for this state using the EventPort.

  // ---------------------------------------------------------------------------
  // APCs

  virtual void allowApc() = 0;
  // If this is ever called, the Win32EventPort will switch modes so that APCs can be scheduled
  // on the thread, e.g. through the Win32 QueueUserAPC() call. In the future, this may be enabled
  // by default. However, as of this writing, Wine does not support the necessary
  // GetQueuedCompletionStatusEx() call, thus allowApc() breaks Wine support. (Tested on Wine
  // 1.8.7.)
  //
  // If the event port implementation can't support APCs for some reason, this throws.

  // ---------------------------------------------------------------------------
  // time

  virtual Timer& getTimer() = 0;
};

class Win32WaitObjectThreadPool {
  // Helper class that implements Win32EventPort::observeSignalState() by spawning additional
  // threads as needed to perform the actual waiting.
  //
  // This class is intended to be used to assist in building Win32EventPort implementations.

public:
  Win32WaitObjectThreadPool(uint mainThreadCount = 0);
  // `mainThreadCount` indicates the number of objects the main thread is able to listen on
  // directly. Typically this would be zero (e.g. if the main thread watches an I/O completion
  // port) or MAXIMUM_WAIT_OBJECTS (e.g. if the main thread is a UI thread but can use
  // MsgWaitForMultipleObjectsEx() to wait on some handles at the same time as messages).

  Own<Win32EventPort::SignalObserver> observeSignalState(HANDLE handle);
  // Implemetns Win32EventPort::observeSignalState().

  uint prepareMainThreadWait(HANDLE* handles[]);
  // Call immediately before invoking WaitForMultipleObjects() or similar in the main thread.
  // Fills in `handles` with the handle pointers to wait on, and returns the number of handles
  // in this array. (The array should be allocated to be at least the size passed to the
  // constructor).
  //
  // There's no need to call this if `mainThreadCount` as passed to the constructor was zero.

  bool finishedMainThreadWait(DWORD returnCode);
  // Call immediately after invoking WaitForMultipleObjects() or similar in the main thread,
  // passing the value returend by that call. Returns true if the event indicated by `returnCode`
  // has been handled (i.e. it was WAIT_OBJECT_n or WAIT_ABANDONED_n where n is in-range for the
  // last call to prepareMainThreadWait()).
};

class Win32IocpEventPort final: public Win32EventPort {
  // An EventPort implementation which uses Windows I/O completion ports to listen for events.
  //
  // With this implementation, observeSignalState() requires spawning a separate thread.

public:
  Win32IocpEventPort();
  ~Win32IocpEventPort() noexcept(false);

  // implements EventPort ------------------------------------------------------
  bool wait() override;
  bool poll() override;
  void wake() const override;

  // implements Win32IocpEventPort ---------------------------------------------
  Own<IoObserver> observeIo(HANDLE handle) override;
  Own<SignalObserver> observeSignalState(HANDLE handle) override;
  Timer& getTimer() override { return timerImpl; }
  void allowApc() override { isAllowApc = true; }

private:
  class IoPromiseAdapter;
  class IoOperationImpl;
  class IoObserverImpl;

  AutoCloseHandle iocp;
  AutoCloseHandle thread;
  Win32WaitObjectThreadPool waitThreads;
  TimerImpl timerImpl;
  mutable std::atomic<bool> sentWake {false};
  bool isAllowApc = false;

  static TimePoint readClock();

  void waitIocp(DWORD timeoutMs);
  // Wait on the I/O completion port for up to timeoutMs and pump events. Does not advance the
  // timer; caller must do that.

  bool receivedWake();

  static AutoCloseHandle newIocpHandle();
  static AutoCloseHandle openCurrentThread();
};

} // namespace kj

#endif // KJ_ASYNC_WIN32_H_
