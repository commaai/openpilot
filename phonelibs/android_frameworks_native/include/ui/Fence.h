/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef ANDROID_FENCE_H
#define ANDROID_FENCE_H

#include <stdint.h>
#include <sys/types.h>

#include <ui/ANativeObjectBase.h>
#include <ui/PixelFormat.h>
#include <ui/Rect.h>
#include <utils/Flattenable.h>
#include <utils/String8.h>
#include <utils/Timers.h>

struct ANativeWindowBuffer;

namespace android {

// ===========================================================================
// Fence
// ===========================================================================

class Fence
    : public LightRefBase<Fence>, public Flattenable<Fence>
{
public:
    static const sp<Fence> NO_FENCE;

    // TIMEOUT_NEVER may be passed to the wait method to indicate that it
    // should wait indefinitely for the fence to signal.
    enum { TIMEOUT_NEVER = -1 };

    // Construct a new Fence object with an invalid file descriptor.  This
    // should be done when the Fence object will be set up by unflattening
    // serialized data.
    Fence();

    // Construct a new Fence object to manage a given fence file descriptor.
    // When the new Fence object is destructed the file descriptor will be
    // closed.
    Fence(int fenceFd);

    // Check whether the Fence has an open fence file descriptor. Most Fence
    // methods treat an invalid file descriptor just like a valid fence that
    // is already signalled, so using this is usually not necessary.
    bool isValid() const { return mFenceFd != -1; }

    // wait waits for up to timeout milliseconds for the fence to signal.  If
    // the fence signals then NO_ERROR is returned. If the timeout expires
    // before the fence signals then -ETIME is returned.  A timeout of
    // TIMEOUT_NEVER may be used to indicate that the call should wait
    // indefinitely for the fence to signal.
    status_t wait(int timeout);

    // waitForever is a convenience function for waiting forever for a fence to
    // signal (just like wait(TIMEOUT_NEVER)), but issuing an error to the
    // system log and fence state to the kernel log if the wait lasts longer
    // than a warning timeout.
    // The logname argument should be a string identifying
    // the caller and will be included in the log message.
    status_t waitForever(const char* logname);

    // merge combines two Fence objects, creating a new Fence object that
    // becomes signaled when both f1 and f2 are signaled (even if f1 or f2 is
    // destroyed before it becomes signaled).  The name argument specifies the
    // human-readable name to associated with the new Fence object.
    static sp<Fence> merge(const String8& name, const sp<Fence>& f1,
            const sp<Fence>& f2);

    // Return a duplicate of the fence file descriptor. The caller is
    // responsible for closing the returned file descriptor. On error, -1 will
    // be returned and errno will indicate the problem.
    int dup() const;

    // getSignalTime returns the system monotonic clock time at which the
    // fence transitioned to the signaled state.  If the fence is not signaled
    // then INT64_MAX is returned.  If the fence is invalid or if an error
    // occurs then -1 is returned.
    nsecs_t getSignalTime() const;

    // Flattenable interface
    size_t getFlattenedSize() const;
    size_t getFdCount() const;
    status_t flatten(void*& buffer, size_t& size, int*& fds, size_t& count) const;
    status_t unflatten(void const*& buffer, size_t& size, int const*& fds, size_t& count);

private:
    // Only allow instantiation using ref counting.
    friend class LightRefBase<Fence>;
    ~Fence();

    // Disallow copying
    Fence(const Fence& rhs);
    Fence& operator = (const Fence& rhs);
    const Fence& operator = (const Fence& rhs) const;

    int mFenceFd;
};

}; // namespace android

#endif // ANDROID_FENCE_H
