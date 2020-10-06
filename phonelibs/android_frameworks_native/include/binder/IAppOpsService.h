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

//
#ifndef ANDROID_IAPP_OPS_SERVICE_H
#define ANDROID_IAPP_OPS_SERVICE_H

#include <binder/IAppOpsCallback.h>
#include <binder/IInterface.h>

namespace android {

// ----------------------------------------------------------------------

class IAppOpsService : public IInterface
{
public:
    DECLARE_META_INTERFACE(AppOpsService);

    virtual int32_t checkOperation(int32_t code, int32_t uid, const String16& packageName) = 0;
    virtual int32_t noteOperation(int32_t code, int32_t uid, const String16& packageName) = 0;
    virtual int32_t startOperation(const sp<IBinder>& token, int32_t code, int32_t uid,
            const String16& packageName) = 0;
    virtual void finishOperation(const sp<IBinder>& token, int32_t code, int32_t uid,
            const String16& packageName) = 0;
    virtual void startWatchingMode(int32_t op, const String16& packageName,
            const sp<IAppOpsCallback>& callback) = 0;
    virtual void stopWatchingMode(const sp<IAppOpsCallback>& callback) = 0;
    virtual sp<IBinder> getToken(const sp<IBinder>& clientToken) = 0;
    virtual int32_t permissionToOpCode(const String16& permission) = 0;

    enum {
        CHECK_OPERATION_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION,
        NOTE_OPERATION_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+1,
        START_OPERATION_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+2,
        FINISH_OPERATION_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+3,
        START_WATCHING_MODE_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+4,
        STOP_WATCHING_MODE_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+5,
        GET_TOKEN_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+6,
        PERMISSION_TO_OP_CODE_TRANSACTION = IBinder::FIRST_CALL_TRANSACTION+7,
    };

    enum {
        MODE_ALLOWED = 0,
        MODE_IGNORED = 1,
        MODE_ERRORED = 2
    };
};

// ----------------------------------------------------------------------

class BnAppOpsService : public BnInterface<IAppOpsService>
{
public:
    virtual status_t    onTransact( uint32_t code,
                                    const Parcel& data,
                                    Parcel* reply,
                                    uint32_t flags = 0);
};

// ----------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IAPP_OPS_SERVICE_H
