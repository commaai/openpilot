/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef ANDROID_ACTIVITY_MANAGER_H
#define ANDROID_ACTIVITY_MANAGER_H

#ifndef __ANDROID_VNDK__

#include <binder/IActivityManager.h>

#include <utils/threads.h>

// ---------------------------------------------------------------------------
namespace android {

class ActivityManager
{
public:

    enum {
        // Flag for registerUidObserver: report uid gone
        UID_OBSERVER_GONE = 1<<1,
        // Flag for registerUidObserver: report uid has become idle
        UID_OBSERVER_IDLE = 1<<2,
        // Flag for registerUidObserver: report uid has become active
        UID_OBSERVER_ACTIVE = 1<<3
    };

    enum {
        // Not a real process state
        PROCESS_STATE_UNKNOWN = -1
    };

    ActivityManager();

    int openContentUri(const String16& stringUri);
    void registerUidObserver(const sp<IUidObserver>& observer,
                             const int32_t event,
                             const int32_t cutpoint,
                             const String16& callingPackage);
    void unregisterUidObserver(const sp<IUidObserver>& observer);
    bool isUidActive(const uid_t uid, const String16& callingPackage);

    status_t linkToDeath(const sp<IBinder::DeathRecipient>& recipient);
    status_t unlinkToDeath(const sp<IBinder::DeathRecipient>& recipient);

private:
    Mutex mLock;
    sp<IActivityManager> mService;
    sp<IActivityManager> getService();
};


}; // namespace android
// ---------------------------------------------------------------------------
#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_ACTIVITY_MANAGER_H
