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

#ifndef ANDROID_SENSOR_EVENT_QUEUE_H
#define ANDROID_SENSOR_EVENT_QUEUE_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>
#include <utils/RefBase.h>
#include <utils/Timers.h>
#include <utils/String16.h>

#include <gui/BitTube.h>

// ----------------------------------------------------------------------------
#define WAKE_UP_SENSOR_EVENT_NEEDS_ACK (1U << 31)
struct ALooper;
struct ASensorEvent;

// Concrete types for the NDK
struct ASensorEventQueue {
    ALooper* looper;
};

// ----------------------------------------------------------------------------
namespace android {
// ----------------------------------------------------------------------------

class ISensorEventConnection;
class Sensor;
class Looper;

// ----------------------------------------------------------------------------

class SensorEventQueue : public ASensorEventQueue, public RefBase
{
public:

    enum { MAX_RECEIVE_BUFFER_EVENT_COUNT = 256 };

    SensorEventQueue(const sp<ISensorEventConnection>& connection);
    virtual ~SensorEventQueue();
    virtual void onFirstRef();

    int getFd() const;

    static ssize_t write(const sp<BitTube>& tube,
            ASensorEvent const* events, size_t numEvents);

    ssize_t read(ASensorEvent* events, size_t numEvents);

    status_t waitForEvent() const;
    status_t wake() const;

    status_t enableSensor(Sensor const* sensor) const;
    status_t disableSensor(Sensor const* sensor) const;
    status_t setEventRate(Sensor const* sensor, nsecs_t ns) const;

    // these are here only to support SensorManager.java
    status_t enableSensor(int32_t handle, int32_t samplingPeriodUs, int maxBatchReportLatencyUs,
                          int reservedFlags) const;
    status_t disableSensor(int32_t handle) const;
    status_t flush() const;
    // Send an ack for every wake_up sensor event that is set to WAKE_UP_SENSOR_EVENT_NEEDS_ACK.
    void sendAck(const ASensorEvent* events, int count);

    status_t injectSensorEvent(const ASensorEvent& event);
private:
    sp<Looper> getLooper() const;
    sp<ISensorEventConnection> mSensorEventConnection;
    sp<BitTube> mSensorChannel;
    mutable Mutex mLock;
    mutable sp<Looper> mLooper;
    ASensorEvent* mRecBuffer;
    size_t mAvailable;
    size_t mConsumed;
    uint32_t mNumAcksToSend;
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_SENSOR_EVENT_QUEUE_H
