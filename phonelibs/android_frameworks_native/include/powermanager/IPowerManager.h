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

#ifndef ANDROID_IPOWERMANAGER_H
#define ANDROID_IPOWERMANAGER_H

#include <utils/Errors.h>
#include <binder/IInterface.h>
#include <hardware/power.h>

namespace android {

// ----------------------------------------------------------------------------

// must be kept in sync with interface defined in IPowerManager.aidl
class IPowerManager : public IInterface
{
public:
    DECLARE_META_INTERFACE(PowerManager);

    // FIXME remove the bool isOneWay parameters as they are not oneway in the .aidl
    virtual status_t acquireWakeLock(int flags, const sp<IBinder>& lock, const String16& tag,
            const String16& packageName, bool isOneWay = false) = 0;
    virtual status_t acquireWakeLockWithUid(int flags, const sp<IBinder>& lock, const String16& tag,
            const String16& packageName, int uid, bool isOneWay = false) = 0;
    virtual status_t releaseWakeLock(const sp<IBinder>& lock, int flags, bool isOneWay = false) = 0;
    virtual status_t updateWakeLockUids(const sp<IBinder>& lock, int len, const int *uids,
            bool isOneWay = false) = 0;
    // oneway in the .aidl
    virtual status_t powerHint(int hintId, int data) = 0;
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IPOWERMANAGER_H
