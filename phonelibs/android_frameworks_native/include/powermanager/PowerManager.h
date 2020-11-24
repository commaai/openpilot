/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef ANDROID_POWERMANAGER_H
#define ANDROID_POWERMANAGER_H

namespace android {

// must be kept in sync with definitions in PowerManager.java
enum {
    POWERMANAGER_PARTIAL_WAKE_LOCK = 1, // equals PowerManager.PARTIAL_WAKE_LOCK constant
};

enum {
    USER_ACTIVITY_EVENT_OTHER = 0,
    USER_ACTIVITY_EVENT_BUTTON = 1,
    USER_ACTIVITY_EVENT_TOUCH = 2,

    USER_ACTIVITY_EVENT_LAST = USER_ACTIVITY_EVENT_TOUCH, // Last valid event code.
};

}; // namespace android

#endif // ANDROID_POWERMANAGER_H
