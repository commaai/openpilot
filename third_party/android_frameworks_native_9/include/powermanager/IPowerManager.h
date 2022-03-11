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

class IPowerManager : public IInterface
{
public:
    // These transaction IDs must be kept in sync with the method order from
    // IPowerManager.aidl.
    enum {
        ACQUIRE_WAKE_LOCK            = IBinder::FIRST_CALL_TRANSACTION,
        ACQUIRE_WAKE_LOCK_UID        = IBinder::FIRST_CALL_TRANSACTION + 1,
        RELEASE_WAKE_LOCK            = IBinder::FIRST_CALL_TRANSACTION + 2,
        UPDATE_WAKE_LOCK_UIDS        = IBinder::FIRST_CALL_TRANSACTION + 3,
        POWER_HINT                   = IBinder::FIRST_CALL_TRANSACTION + 4,
        UPDATE_WAKE_LOCK_SOURCE      = IBinder::FIRST_CALL_TRANSACTION + 5,
        IS_WAKE_LOCK_LEVEL_SUPPORTED = IBinder::FIRST_CALL_TRANSACTION + 6,
        USER_ACTIVITY                = IBinder::FIRST_CALL_TRANSACTION + 7,
        WAKE_UP                      = IBinder::FIRST_CALL_TRANSACTION + 8,
        GO_TO_SLEEP                  = IBinder::FIRST_CALL_TRANSACTION + 9,
        NAP                          = IBinder::FIRST_CALL_TRANSACTION + 10,
        IS_INTERACTIVE               = IBinder::FIRST_CALL_TRANSACTION + 11,
        IS_POWER_SAVE_MODE           = IBinder::FIRST_CALL_TRANSACTION + 12,
        GET_POWER_SAVE_STATE         = IBinder::FIRST_CALL_TRANSACTION + 13,
        SET_POWER_SAVE_MODE          = IBinder::FIRST_CALL_TRANSACTION + 14,
        REBOOT                       = IBinder::FIRST_CALL_TRANSACTION + 17,
        REBOOT_SAFE_MODE             = IBinder::FIRST_CALL_TRANSACTION + 18,
        SHUTDOWN                     = IBinder::FIRST_CALL_TRANSACTION + 19,
        CRASH                        = IBinder::FIRST_CALL_TRANSACTION + 20,
    };

    DECLARE_META_INTERFACE(PowerManager)

    // The parcels created by these methods must be kept in sync with the
    // corresponding methods from IPowerManager.aidl.
    // FIXME remove the bool isOneWay parameters as they are not oneway in the .aidl
    virtual status_t acquireWakeLock(int flags, const sp<IBinder>& lock, const String16& tag,
            const String16& packageName, bool isOneWay = false) = 0;
    virtual status_t acquireWakeLockWithUid(int flags, const sp<IBinder>& lock, const String16& tag,
            const String16& packageName, int uid, bool isOneWay = false) = 0;
    virtual status_t releaseWakeLock(const sp<IBinder>& lock, int flags, bool isOneWay = false) = 0;
    virtual status_t updateWakeLockUids(const sp<IBinder>& lock, int len, const int *uids,
            bool isOneWay = false) = 0;
    virtual status_t powerHint(int hintId, int data) = 0;
    virtual status_t goToSleep(int64_t event_time_ms, int reason, int flags) = 0;
    virtual status_t reboot(bool confirm, const String16& reason, bool wait) = 0;
    virtual status_t shutdown(bool confirm, const String16& reason, bool wait) = 0;
    virtual status_t crash(const String16& message) = 0;
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IPOWERMANAGER_H
