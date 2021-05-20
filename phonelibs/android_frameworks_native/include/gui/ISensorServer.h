/*
 * Copyright (C) 2010 The Android Open Source Project
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

#ifndef ANDROID_GUI_ISENSORSERVER_H
#define ANDROID_GUI_ISENSORSERVER_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>
#include <utils/RefBase.h>

#include <binder/IInterface.h>

namespace android {
// ----------------------------------------------------------------------------

class Sensor;
class ISensorEventConnection;
class String8;

class ISensorServer : public IInterface
{
public:
    DECLARE_META_INTERFACE(SensorServer);

    virtual Vector<Sensor> getSensorList(const String16& opPackageName) = 0;
    virtual sp<ISensorEventConnection> createSensorEventConnection(const String8& packageName,
             int mode, const String16& opPackageName) = 0;
    virtual int32_t isDataInjectionEnabled() = 0;
};

// ----------------------------------------------------------------------------

class BnSensorServer : public BnInterface<ISensorServer>
{
public:
    virtual status_t    onTransact( uint32_t code,
                                    const Parcel& data,
                                    Parcel* reply,
                                    uint32_t flags = 0);
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_ISENSORSERVER_H
