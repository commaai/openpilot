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

#ifndef ANDROID_I_PROCESS_INFO_SERVICE_H
#define ANDROID_I_PROCESS_INFO_SERVICE_H

#ifndef __ANDROID_VNDK__

#include <binder/IInterface.h>

namespace android {

// ----------------------------------------------------------------------

class IProcessInfoService : public IInterface {
public:
    DECLARE_META_INTERFACE(ProcessInfoService)

    virtual status_t    getProcessStatesFromPids( size_t length,
                                                  /*in*/ int32_t* pids,
                                                  /*out*/ int32_t* states) = 0;

    virtual status_t    getProcessStatesAndOomScoresFromPids( size_t length,
                                                  /*in*/ int32_t* pids,
                                                  /*out*/ int32_t* states,
                                                  /*out*/ int32_t* scores) = 0;

    enum {
        GET_PROCESS_STATES_FROM_PIDS = IBinder::FIRST_CALL_TRANSACTION,
        GET_PROCESS_STATES_AND_OOM_SCORES_FROM_PIDS,
    };
};

// ----------------------------------------------------------------------

}; // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif // ANDROID_I_PROCESS_INFO_SERVICE_H
