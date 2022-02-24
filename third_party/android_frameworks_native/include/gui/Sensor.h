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

#ifndef ANDROID_GUI_SENSOR_H
#define ANDROID_GUI_SENSOR_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>
#include <utils/Flattenable.h>
#include <utils/String8.h>
#include <utils/Timers.h>

#include <hardware/sensors.h>

#include <android/sensor.h>

// ----------------------------------------------------------------------------
// Concrete types for the NDK
struct ASensor { };

// ----------------------------------------------------------------------------
namespace android {
// ----------------------------------------------------------------------------

class Parcel;

// ----------------------------------------------------------------------------

class Sensor : public ASensor, public LightFlattenable<Sensor>
{
public:
    enum {
        TYPE_ACCELEROMETER  = ASENSOR_TYPE_ACCELEROMETER,
        TYPE_MAGNETIC_FIELD = ASENSOR_TYPE_MAGNETIC_FIELD,
        TYPE_GYROSCOPE      = ASENSOR_TYPE_GYROSCOPE,
        TYPE_LIGHT          = ASENSOR_TYPE_LIGHT,
        TYPE_PROXIMITY      = ASENSOR_TYPE_PROXIMITY
    };

            Sensor();
            Sensor(struct sensor_t const* hwSensor, int halVersion = 0);
            ~Sensor();

    const String8& getName() const;
    const String8& getVendor() const;
    int32_t getHandle() const;
    int32_t getType() const;
    float getMinValue() const;
    float getMaxValue() const;
    float getResolution() const;
    float getPowerUsage() const;
    int32_t getMinDelay() const;
    nsecs_t getMinDelayNs() const;
    int32_t getVersion() const;
    uint32_t getFifoReservedEventCount() const;
    uint32_t getFifoMaxEventCount() const;
    const String8& getStringType() const;
    const String8& getRequiredPermission() const;
    bool isRequiredPermissionRuntime() const;
    int32_t getRequiredAppOp() const;
    int32_t getMaxDelay() const;
    uint32_t getFlags() const;
    bool isWakeUpSensor() const;
    int32_t getReportingMode() const;

    // LightFlattenable protocol
    inline bool isFixedSize() const { return false; }
    size_t getFlattenedSize() const;
    status_t flatten(void* buffer, size_t size) const;
    status_t unflatten(void const* buffer, size_t size);

private:
    String8 mName;
    String8 mVendor;
    int32_t mHandle;
    int32_t mType;
    float   mMinValue;
    float   mMaxValue;
    float   mResolution;
    float   mPower;
    int32_t mMinDelay;
    int32_t mVersion;
    uint32_t mFifoReservedEventCount;
    uint32_t mFifoMaxEventCount;
    String8 mStringType;
    String8 mRequiredPermission;
    bool mRequiredPermissionRuntime = false;
    int32_t mRequiredAppOp;
    int32_t mMaxDelay;
    uint32_t mFlags;
    static void flattenString8(void*& buffer, size_t& size, const String8& string8);
    static bool unflattenString8(void const*& buffer, size_t& size, String8& outputString8);
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_SENSOR_H
