/*
 * Copyright 2015 The Android Open Source Project
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

#ifndef ANDROID_PROCESS_INFO_SERVICE_H
#define ANDROID_PROCESS_INFO_SERVICE_H

#ifndef __ANDROID_VNDK__

#include <binder/IProcessInfoService.h>
#include <utils/Errors.h>
#include <utils/Singleton.h>
#include <sys/types.h>

namespace android {

// ----------------------------------------------------------------------

class ProcessInfoService : public Singleton<ProcessInfoService> {

    friend class Singleton<ProcessInfoService>;
    sp<IProcessInfoService> mProcessInfoService;
    Mutex mProcessInfoLock;

    ProcessInfoService();

    status_t getProcessStatesImpl(size_t length, /*in*/ int32_t* pids, /*out*/ int32_t* states);
    status_t getProcessStatesScoresImpl(size_t length, /*in*/ int32_t* pids,
            /*out*/ int32_t* states, /*out*/ int32_t *scores);
    void updateBinderLocked();

    static const int BINDER_ATTEMPT_LIMIT = 5;

public:

    /**
     * For each PID in the given "pids" input array, write the current process state
     * for that process into the "states" output array, or
     * ActivityManager.PROCESS_STATE_NONEXISTENT * to indicate that no process with the given PID
     * exists.
     *
     * Returns NO_ERROR if this operation was successful, or a negative error code otherwise.
     */
    static status_t getProcessStatesFromPids(size_t length, /*in*/ int32_t* pids,
            /*out*/ int32_t* states) {
        return ProcessInfoService::getInstance().getProcessStatesImpl(length, /*in*/ pids,
                /*out*/ states);
    }

    /**
     * For each PID in the given "pids" input array, write the current process state
     * for that process into the "states" output array, or
     * ActivityManager.PROCESS_STATE_NONEXISTENT * to indicate that no process with the given PID
     * exists. OoM scores will also be written in the "scores" output array.
     * Please also note that clients calling this method need to have
     * "GET_PROCESS_STATE_AND_OOM_SCORE" permission.
     *
     * Returns NO_ERROR if this operation was successful, or a negative error code otherwise.
     */
    static status_t getProcessStatesScoresFromPids(size_t length, /*in*/ int32_t* pids,
            /*out*/ int32_t* states, /*out*/ int32_t *scores) {
        return ProcessInfoService::getInstance().getProcessStatesScoresImpl(
                length, /*in*/ pids, /*out*/ states, /*out*/ scores);
    }
};

// ----------------------------------------------------------------------

}; // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_PROCESS_INFO_SERVICE_H

