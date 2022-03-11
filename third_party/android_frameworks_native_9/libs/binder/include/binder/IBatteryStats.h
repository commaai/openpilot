/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_IBATTERYSTATS_H
#define ANDROID_IBATTERYSTATS_H

#ifndef __ANDROID_VNDK__

#include <binder/IInterface.h>

namespace android {

// ----------------------------------------------------------------------

class IBatteryStats : public IInterface
{
public:
    DECLARE_META_INTERFACE(BatteryStats)

    virtual void noteStartSensor(int uid, int sensor) = 0;
    virtual void noteStopSensor(int uid, int sensor) = 0;
    virtual void noteStartVideo(int uid) = 0;
    virtual void noteStopVideo(int uid) = 0;
    virtual void noteStartAudio(int uid) = 0;
    virtual void noteStopAudio(int uid) = 0;
    virtual void noteResetVideo() = 0;
    virtual void noteResetAudio() = 0;
    virtual void noteFlashlightOn(int uid) = 0;
    virtual void noteFlashlightOff(int uid) = 0;
    virtual void noteStartCamera(int uid) = 0;
    virtual void noteStopCamera(int uid) = 0;
    virtual void noteResetCamera() = 0;
    virtual void noteResetFlashlight() = 0;

    enum {
        NOTE_START_SENSOR_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION,
        NOTE_STOP_SENSOR_TRANSACTION,
        NOTE_START_VIDEO_TRANSACTION,
        NOTE_STOP_VIDEO_TRANSACTION,
        NOTE_START_AUDIO_TRANSACTION,
        NOTE_STOP_AUDIO_TRANSACTION,
        NOTE_RESET_VIDEO_TRANSACTION,
        NOTE_RESET_AUDIO_TRANSACTION,
        NOTE_FLASHLIGHT_ON_TRANSACTION,
        NOTE_FLASHLIGHT_OFF_TRANSACTION,
        NOTE_START_CAMERA_TRANSACTION,
        NOTE_STOP_CAMERA_TRANSACTION,
        NOTE_RESET_CAMERA_TRANSACTION,
        NOTE_RESET_FLASHLIGHT_TRANSACTION
    };
};

// ----------------------------------------------------------------------

class BnBatteryStats : public BnInterface<IBatteryStats>
{
public:
    virtual status_t    onTransact( uint32_t code,
                                    const Parcel& data,
                                    Parcel* reply,
                                    uint32_t flags = 0);
};

// ----------------------------------------------------------------------

}; // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_IBATTERYSTATS_H
