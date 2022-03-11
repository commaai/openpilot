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

#ifndef ANDROID_IAUDIOMANAGER_H
#define ANDROID_IAUDIOMANAGER_H

#include <utils/Errors.h>
#include <binder/IInterface.h>
#include <hardware/power.h>
#include <system/audio.h>

namespace android {

// ----------------------------------------------------------------------------

class IAudioManager : public IInterface
{
public:
    // These transaction IDs must be kept in sync with the method order from
    // IAudioService.aidl.
    enum {
        TRACK_PLAYER                          = IBinder::FIRST_CALL_TRANSACTION,
        PLAYER_ATTRIBUTES                     = IBinder::FIRST_CALL_TRANSACTION + 1,
        PLAYER_EVENT                          = IBinder::FIRST_CALL_TRANSACTION + 2,
        RELEASE_PLAYER                        = IBinder::FIRST_CALL_TRANSACTION + 3,
    };

    DECLARE_META_INTERFACE(AudioManager)

    // The parcels created by these methods must be kept in sync with the
    // corresponding methods from IAudioService.aidl and objects it imports.
    virtual audio_unique_id_t trackPlayer(player_type_t playerType, audio_usage_t usage,
                audio_content_type_t content, const sp<IBinder>& player) = 0;
    /*oneway*/ virtual status_t playerAttributes(audio_unique_id_t piid, audio_usage_t usage,
                audio_content_type_t content)= 0;
    /*oneway*/ virtual status_t playerEvent(audio_unique_id_t piid, player_state_t event) = 0;
    /*oneway*/ virtual status_t releasePlayer(audio_unique_id_t piid) = 0;
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IAUDIOMANAGER_H
