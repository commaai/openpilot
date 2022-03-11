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

#ifndef ANDROID_VR_MANAGER_H
#define ANDROID_VR_MANAGER_H

#include <binder/IInterface.h>

namespace android {

// Must be kept in sync with interface defined in IVrStateCallbacks.aidl.

class IVrStateCallbacks : public IInterface {
public:
    DECLARE_META_INTERFACE(VrStateCallbacks)

    virtual void onVrStateChanged(bool enabled) = 0;
};

enum VrStateCallbacksTransaction {
    ON_VR_STATE_CHANGED = IBinder::FIRST_CALL_TRANSACTION,
};

class BnVrStateCallbacks : public BnInterface<IVrStateCallbacks> {
public:
    status_t onTransact(uint32_t code, const Parcel& data,
                        Parcel* reply, uint32_t flags = 0) override;
};


// Must be kept in sync with interface defined in
// IPersistentVrStateCallbacks.aidl.

class IPersistentVrStateCallbacks : public IInterface {
public:
    DECLARE_META_INTERFACE(PersistentVrStateCallbacks)

    virtual void onPersistentVrStateChanged(bool enabled) = 0;
};

enum PersistentVrStateCallbacksTransaction {
    ON_PERSISTENT_VR_STATE_CHANGED = IBinder::FIRST_CALL_TRANSACTION,
};

class BnPersistentVrStateCallbacks
        : public BnInterface<IPersistentVrStateCallbacks> {
public:
    status_t onTransact(uint32_t code, const Parcel& data,
                        Parcel* reply, uint32_t flags = 0) override;
};


// Must be kept in sync with interface defined in IVrManager.aidl.

class IVrManager : public IInterface {
public:
    DECLARE_META_INTERFACE(VrManager)

    virtual void registerListener(const sp<IVrStateCallbacks>& cb) = 0;
    virtual void unregisterListener(const sp<IVrStateCallbacks>& cb) = 0;
    virtual void registerPersistentVrStateListener(
        const sp<IPersistentVrStateCallbacks>& cb) = 0;
    virtual void unregisterPersistentVrStateListener(
        const sp<IPersistentVrStateCallbacks>& cb) = 0;
    virtual bool getVrModeState() = 0;
};

enum VrManagerTransaction {
    REGISTER_LISTENER = IBinder::FIRST_CALL_TRANSACTION,
    UNREGISTER_LISTENER,
    REGISTER_PERSISTENT_VR_STATE_LISTENER,
    UNREGISTER_PERSISTENT_VR_STATE_LISTENER,
    GET_VR_MODE_STATE,
};

};  // namespace android

#endif // ANDROID_VR_MANAGER_H
