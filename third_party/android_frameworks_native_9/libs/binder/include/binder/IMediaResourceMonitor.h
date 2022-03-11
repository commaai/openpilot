/*
 * Copyright 2016 The Android Open Source Project
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

#ifndef ANDROID_I_MEDIA_RESOURCE_MONITOR_H
#define ANDROID_I_MEDIA_RESOURCE_MONITOR_H

#ifndef __ANDROID_VNDK__

#include <binder/IInterface.h>

namespace android {

// ----------------------------------------------------------------------

class IMediaResourceMonitor : public IInterface {
public:
    DECLARE_META_INTERFACE(MediaResourceMonitor)

    // Values should be in sync with Intent.EXTRA_MEDIA_RESOURCE_TYPE_XXX.
    enum {
        TYPE_VIDEO_CODEC = 0,
        TYPE_AUDIO_CODEC = 1,
    };

    virtual void notifyResourceGranted(/*in*/ int32_t pid, /*in*/ const int32_t type) = 0;

    enum {
        NOTIFY_RESOURCE_GRANTED = IBinder::FIRST_CALL_TRANSACTION,
    };
};

// ----------------------------------------------------------------------

class BnMediaResourceMonitor : public BnInterface<IMediaResourceMonitor> {
public:
    virtual status_t onTransact(uint32_t code, const Parcel& data, Parcel* reply,
            uint32_t flags = 0);
};

// ----------------------------------------------------------------------

}; // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_I_MEDIA_RESOURCE_MONITOR_H
