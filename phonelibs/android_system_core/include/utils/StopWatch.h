/*
 * Copyright (C) 2005 The Android Open Source Project
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

#ifndef ANDROID_STOPWATCH_H
#define ANDROID_STOPWATCH_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Timers.h>

// ---------------------------------------------------------------------------

namespace android {

class StopWatch
{
public:
        StopWatch(  const char *name,
                    int clock = SYSTEM_TIME_MONOTONIC,
                    uint32_t flags = 0);
        ~StopWatch();
        
        const char* name() const;
        nsecs_t     lap();
        nsecs_t     elapsedTime() const;

        void        reset();
        
private:
    const char*     mName;
    int             mClock;
    uint32_t        mFlags;
    
    struct lap_t {
        nsecs_t     soFar;
        nsecs_t     thisLap;
    };
    
    nsecs_t         mStartTime;
    lap_t           mLaps[8];
    int             mNumLaps;
};


}; // namespace android


// ---------------------------------------------------------------------------

#endif // ANDROID_STOPWATCH_H
