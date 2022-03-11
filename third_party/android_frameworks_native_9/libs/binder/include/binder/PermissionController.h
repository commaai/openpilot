/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef ANDROID_PERMISSION_CONTROLLER_H
#define ANDROID_PERMISSION_CONTROLLER_H

#ifndef __ANDROID_VNDK__

#include <binder/IPermissionController.h>

#include <utils/threads.h>

// ---------------------------------------------------------------------------
namespace android {

class PermissionController
{
public:

    enum {
        MATCH_SYSTEM_ONLY = 1<<16,
        MATCH_UNINSTALLED_PACKAGES = 1<<13,
        MATCH_FACTORY_ONLY = 1<<21,
        MATCH_INSTANT = 1<<23
    };

    enum {
        MODE_ALLOWED = 0,
        MODE_IGNORED = 1,
        MODE_ERRORED = 2,
        MODE_DEFAULT = 3,
    };

    PermissionController();

    bool checkPermission(const String16& permission, int32_t pid, int32_t uid);
    int32_t noteOp(const String16& op, int32_t uid, const String16& packageName);
    void getPackagesForUid(const uid_t uid, Vector<String16>& packages);
    bool isRuntimePermission(const String16& permission);
    int getPackageUid(const String16& package, int flags);

private:
    Mutex mLock;
    sp<IPermissionController> mService;

    sp<IPermissionController> getService();
};


}; // namespace android
// ---------------------------------------------------------------------------
#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_PERMISSION_CONTROLLER_H
