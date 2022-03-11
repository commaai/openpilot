/*
 * Copyright (C) 2014 The Android Open Source Project
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

#ifndef ANDROID_UI_FRAME_STATS_H
#define ANDROID_UI_FRAME_STATS_H

#include <utils/Flattenable.h>
#include <utils/Timers.h>
#include <utils/Vector.h>

namespace android {

class FrameStats : public LightFlattenable<FrameStats> {
public:
    FrameStats() : refreshPeriodNano(0) {}

    /*
     * Approximate refresh time, in nanoseconds.
     */
    nsecs_t refreshPeriodNano;

   /*
    * The times in nanoseconds for when the frame contents were posted by the producer (e.g.
    * the application). They are either explicitly set or defaulted to the time when
    * Surface::queueBuffer() was called.
    */
    Vector<nsecs_t> desiredPresentTimesNano;

   /*
    * The times in milliseconds for when the frame contents were presented on the screen.
    */
    Vector<nsecs_t> actualPresentTimesNano;

   /*
    * The times in nanoseconds for when the frame contents were ready to be presented. Note that
    * a frame can be posted and still it contents being rendered asynchronously in GL. In such a
    * case these are the times when the frame contents were completely rendered (i.e. their fences
    * signaled).
    */
    Vector<nsecs_t> frameReadyTimesNano;

    // LightFlattenable
    bool isFixedSize() const;
    size_t getFlattenedSize() const;
    status_t flatten(void* buffer, size_t size) const;
    status_t unflatten(void const* buffer, size_t size);
};

}; // namespace android

#endif // ANDROID_UI_FRAME_STATS_H
