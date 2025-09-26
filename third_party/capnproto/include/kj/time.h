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

#include "units.h"
#include <inttypes.h>
#include "string.h"

KJ_BEGIN_HEADER

namespace kj {
namespace _ {  // private

class NanosecondLabel;
class TimeLabel;
class DateLabel;

static constexpr size_t TIME_STR_LEN = sizeof(int64_t) * 3 + 8;
// Maximum length of a stringified time. 3 digits per byte of integer, plus 8 digits to cover
// negative sign, decimal point, unit, NUL terminator, and anything else that might sneak in.

}  // namespace _ (private)

using Duration = Quantity<int64_t, _::NanosecondLabel>;
// A time value, in nanoseconds.

constexpr Duration NANOSECONDS = unit<Duration>();
constexpr Duration MICROSECONDS = 1000 * NANOSECONDS;
constexpr Duration MILLISECONDS = 1000 * MICROSECONDS;
constexpr Duration SECONDS = 1000 * MILLISECONDS;
constexpr Duration MINUTES = 60 * SECONDS;
constexpr Duration HOURS = 60 * MINUTES;
constexpr Duration DAYS = 24 * HOURS;

using TimePoint = Absolute<Duration, _::TimeLabel>;
// An absolute time measured by some particular instance of `Timer` or `MonotonicClock`. `Time`s
// from two different `Timer`s or `MonotonicClock`s may be measured from different origins and so
// are not necessarily compatible.

using Date = Absolute<Duration, _::DateLabel>;
// A point in real-world time, measured relative to the Unix epoch (Jan 1, 1970 00:00:00 UTC).

CappedArray<char, _::TIME_STR_LEN> KJ_STRINGIFY(TimePoint);
CappedArray<char, _::TIME_STR_LEN> KJ_STRINGIFY(Date);
CappedArray<char, _::TIME_STR_LEN> KJ_STRINGIFY(Duration);

constexpr Date UNIX_EPOCH = origin<Date>();
// The `Date` representing Jan 1, 1970 00:00:00 UTC.

class Clock {
  // Interface to read the current date and time.
public:
  virtual Date now() const = 0;
};

class MonotonicClock {
  // Interface to read time in a way that increases as real-world time increases, independent of
  // any manual changes to the calendar date/time. Such a clock never "goes backwards" even if the
  // system administrator changes the calendar time or suspends the system. However, this clock's
  // time points are only meaningful in comparison to other time points from the same clock, and
  // cannot be used to determine the current calendar date.

public:
  virtual TimePoint now() const = 0;
};

const Clock& nullClock();
// A clock which always returns UNIX_EPOCH as the current time. Useful when you don't care about
// time.

const Clock& systemCoarseCalendarClock();
const Clock& systemPreciseCalendarClock();
// A clock that reads the real system time.
//
// In well-designed code, this should only be called by the top-level dependency injector. All
// other modules should request that the caller provide a Clock so that alternate clock
// implementations can be injected for testing, simulation, reproducibility, and other purposes.
//
// The "coarse" version has precision around 1-10ms, while the "precise" version has precision
// better than 1us. The "precise" version may be slightly slower, though on modern hardware and
// a reasonable operating system the difference is usually negligible.
//
// Note: On Windows prior to Windows 8, there is no precise calendar clock; the "precise" clock
//   will be no more precise than the "coarse" clock in this case.

const MonotonicClock& systemCoarseMonotonicClock();
const MonotonicClock& systemPreciseMonotonicClock();
// A MonotonicClock that reads the real system time.
//
// In well-designed code, this should only be called by the top-level dependency injector. All
// other modules should request that the caller provide a Clock so that alternate clock
// implementations can be injected for testing, simulation, reproducibility, and other purposes.
//
// The "coarse" version has precision around 1-10ms, while the "precise" version has precision
// better than 1us. The "precise" version may be slightly slower, though on modern hardware and
// a reasonable operating system the difference is usually negligible.
}  // namespace kj

KJ_END_HEADER
