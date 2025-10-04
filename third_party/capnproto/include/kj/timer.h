// Copyright (c) 2014 Google Inc. (contributed by Remy Blank <rblank@google.com>)
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

#include <kj/time.h>
#include "async.h"

KJ_BEGIN_HEADER

namespace kj {

class Timer: public MonotonicClock {
  // Interface to time and timer functionality.
  //
  // Each `Timer` may have a different origin, and some `Timer`s may in fact tick at a different
  // rate than real time (e.g. a `Timer` could represent CPU time consumed by a thread).  However,
  // all `Timer`s are monotonic: time will never appear to move backwards, even if the calendar
  // date as tracked by the system is manually modified.
  //
  // That said, the `Timer` returned by `kj::setupAsyncIo().provider->getTimer()` in particular is
  // guaranteed to be synchronized with the `MonotonicClock` returned by
  // `systemPreciseMonotonicClock()` (or, more precisely, is updated to match that clock whenever
  // the loop waits).
  //
  // Note that the value returned by `Timer::now()` only changes each time the
  // event loop waits for I/O from the system. While the event loop is actively
  // running, the time stays constant. This is intended to make behavior more
  // deterministic and reproducible. However, if you need up-to-the-cycle
  // accurate time, then `Timer::now()` is not appropriate. Instead, use
  // `systemPreciseMonotonicClock()` directly in this case.

public:
  virtual TimePoint now() const = 0;
  // Returns the current value of a clock that moves steadily forward, independent of any
  // changes in the wall clock. The value is updated every time the event loop waits,
  // and is constant in-between waits.

  virtual Promise<void> atTime(TimePoint time) = 0;
  // Returns a promise that returns as soon as now() >= time.

  virtual Promise<void> afterDelay(Duration delay) = 0;
  // Equivalent to atTime(now() + delay).

  template <typename T>
  Promise<T> timeoutAt(TimePoint time, Promise<T>&& promise) KJ_WARN_UNUSED_RESULT;
  // Return a promise equivalent to `promise` but which throws an exception (and cancels the
  // original promise) if it hasn't completed by `time`. The thrown exception is of type
  // "OVERLOADED".

  template <typename T>
  Promise<T> timeoutAfter(Duration delay, Promise<T>&& promise) KJ_WARN_UNUSED_RESULT;
  // Return a promise equivalent to `promise` but which throws an exception (and cancels the
  // original promise) if it hasn't completed after `delay` from now. The thrown exception is of
  // type "OVERLOADED".

private:
  static kj::Exception makeTimeoutException();
};

class TimerImpl final: public Timer {
  // Implementation of Timer that expects an external caller -- usually, the EventPort
  // implementation -- to tell it when time has advanced.

public:
  TimerImpl(TimePoint startTime);
  ~TimerImpl() noexcept(false);

  Maybe<TimePoint> nextEvent();
  // Returns the time at which the next scheduled timer event will occur, or null if no timer
  // events are scheduled.

  Maybe<uint64_t> timeoutToNextEvent(TimePoint start, Duration unit, uint64_t max);
  // Convenience method which computes a timeout value to pass to an event-waiting system call to
  // cause it to time out when the next timer event occurs.
  //
  // `start` is the time at which the timeout starts counting. This is typically not the same as
  // now() since some time may have passed since the last time advanceTo() was called.
  //
  // `unit` is the time unit in which the timeout is measured. This is often MILLISECONDS. Note
  // that this method will fractional values *up*, to guarantee that the returned timeout waits
  // until just *after* the time the event is scheduled.
  //
  // The timeout will be clamped to `max`. Use this to avoid an overflow if e.g. the OS wants a
  // 32-bit value or a signed value.
  //
  // Returns nullptr if there are no future events.

  void advanceTo(TimePoint newTime);
  // Set the time to `time` and fire any at() events that have been passed.

  // implements Timer ----------------------------------------------------------
  TimePoint now() const override;
  Promise<void> atTime(TimePoint time) override;
  Promise<void> afterDelay(Duration delay) override;

private:
  struct Impl;
  class TimerPromiseAdapter;
  TimePoint time;
  Own<Impl> impl;
};

// =======================================================================================
// inline implementation details

template <typename T>
Promise<T> Timer::timeoutAt(TimePoint time, Promise<T>&& promise) {
  return promise.exclusiveJoin(atTime(time).then([]() -> kj::Promise<T> {
    return makeTimeoutException();
  }));
}

template <typename T>
Promise<T> Timer::timeoutAfter(Duration delay, Promise<T>&& promise) {
  return promise.exclusiveJoin(afterDelay(delay).then([]() -> kj::Promise<T> {
    return makeTimeoutException();
  }));
}

inline TimePoint TimerImpl::now() const { return time; }

}  // namespace kj

KJ_END_HEADER
