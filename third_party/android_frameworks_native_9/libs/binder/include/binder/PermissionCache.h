/*
 * Copyright (C) 2009 The Android Open Source Project
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

#ifndef BINDER_PERMISSION_H
#define BINDER_PERMISSION_H

#ifndef __ANDROID_VNDK__

#include <stdint.h>
#include <unistd.h>

#include <utils/String16.h>
#include <utils/Singleton.h>
#include <utils/SortedVector.h>

namespace android {
// ---------------------------------------------------------------------------

/*
 * PermissionCache caches permission checks for a given uid.
 *
 * Currently the cache is not updated when there is a permission change,
 * for instance when an application is uninstalled.
 *
 * IMPORTANT: for the reason stated above, only system permissions are safe
 * to cache. This restriction may be lifted at a later time.
 *
 */

class PermissionCache : Singleton<PermissionCache> {
    struct Entry {
        String16    name;
        uid_t       uid;
        bool        granted;
        inline bool operator < (const Entry& e) const {
            return (uid == e.uid) ? (name < e.name) : (uid < e.uid);
        }
    };
    mutable Mutex mLock;
    // we pool all the permission names we see, as many permissions checks
    // will have identical names
    SortedVector< String16 > mPermissionNamesPool;
    // this is our cache per say. it stores pooled names.
    SortedVector< Entry > mCache;

    // free the whole cache, but keep the permission name pool
    void purge();

    status_t check(bool* granted,
            const String16& permission, uid_t uid) const;

    void cache(const String16& permission, uid_t uid, bool granted);

public:
    PermissionCache();

    static bool checkCallingPermission(const String16& permission);

    static bool checkCallingPermission(const String16& permission,
                                int32_t* outPid, int32_t* outUid);

    static bool checkPermission(const String16& permission,
            pid_t pid, uid_t uid);
};

// ---------------------------------------------------------------------------
}; // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif /* BINDER_PERMISSION_H */
