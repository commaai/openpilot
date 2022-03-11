/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef ANDROID_AUDIOMANAGER_H
#define ANDROID_AUDIOMANAGER_H

namespace android {

// must be kept in sync with definitions in AudioPlaybackConfiguration.java

#define PLAYER_PIID_INVALID -1

typedef enum {
    PLAYER_TYPE_SLES_AUDIOPLAYER_BUFFERQUEUE = 11,
    PLAYER_TYPE_SLES_AUDIOPLAYER_URI_FD = 12,
    PLAYER_TYPE_AAUDIO = 13,
    PLAYER_TYPE_HW_SOURCE = 14,
    PLAYER_TYPE_EXTERNAL_PROXY = 15,
} player_type_t;

typedef enum {
    PLAYER_STATE_UNKNOWN  = -1,
    PLAYER_STATE_RELEASED = 0,
    PLAYER_STATE_IDLE     = 1,
    PLAYER_STATE_STARTED  = 2,
    PLAYER_STATE_PAUSED   = 3,
    PLAYER_STATE_STOPPED  = 4,
} player_state_t;

}; // namespace android

#endif // ANDROID_AUDIOMANAGER_H
