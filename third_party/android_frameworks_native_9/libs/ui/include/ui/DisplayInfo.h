/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef ANDROID_UI_DISPLAY_INFO_H
#define ANDROID_UI_DISPLAY_INFO_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Timers.h>

namespace android {

struct DisplayInfo {
    uint32_t w{0};
    uint32_t h{0};
    float xdpi{0};
    float ydpi{0};
    float fps{0};
    float density{0};
    uint8_t orientation{0};
    bool secure{false};
    nsecs_t appVsyncOffset{0};
    nsecs_t presentationDeadline{0};
};

/* Display orientations as defined in Surface.java and ISurfaceComposer.h. */
enum {
    DISPLAY_ORIENTATION_0 = 0,
    DISPLAY_ORIENTATION_90 = 1,
    DISPLAY_ORIENTATION_180 = 2,
    DISPLAY_ORIENTATION_270 = 3
};

}; // namespace android

#endif // ANDROID_COMPOSER_DISPLAY_INFO_H
