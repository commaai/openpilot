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

#ifndef ANDROID_IBATTERYPROPERTIESLISTENER_H
#define ANDROID_IBATTERYPROPERTIESLISTENER_H

#include <binder/IBinder.h>
#include <binder/IInterface.h>

#include <batteryservice/BatteryService.h>

namespace android {

// must be kept in sync with interface defined in IBatteryPropertiesListener.aidl
enum {
        TRANSACT_BATTERYPROPERTIESCHANGED = IBinder::FIRST_CALL_TRANSACTION,
};

// ----------------------------------------------------------------------------

class IBatteryPropertiesListener : public IInterface {
public:
    DECLARE_META_INTERFACE(BatteryPropertiesListener);

    virtual void batteryPropertiesChanged(struct BatteryProperties props) = 0;
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IBATTERYPROPERTIESLISTENER_H
