/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef ANDROID_GUI_IDISPLAY_EVENT_CONNECTION_H
#define ANDROID_GUI_IDISPLAY_EVENT_CONNECTION_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>
#include <utils/RefBase.h>

#include <binder/IInterface.h>

namespace android {
// ----------------------------------------------------------------------------

class BitTube;

class IDisplayEventConnection : public IInterface
{
public:

    DECLARE_META_INTERFACE(DisplayEventConnection);

    /*
     * getDataChannel() returns a BitTube where to receive the events from
     */
    virtual sp<BitTube> getDataChannel() const = 0;

    /*
     * setVsyncRate() sets the vsync event delivery rate. A value of
     * 1 returns every vsync events. A value of 2 returns every other events,
     * etc... a value of 0 returns no event unless  requestNextVsync() has
     * been called.
     */
    virtual void setVsyncRate(uint32_t count) = 0;

    /*
     * requestNextVsync() schedules the next vsync event. It has no effect
     * if the vsync rate is > 0.
     */
    virtual void requestNextVsync() = 0;    // asynchronous
};

// ----------------------------------------------------------------------------

class BnDisplayEventConnection : public BnInterface<IDisplayEventConnection>
{
public:
    virtual status_t    onTransact( uint32_t code,
                                    const Parcel& data,
                                    Parcel* reply,
                                    uint32_t flags = 0);
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_IDISPLAY_EVENT_CONNECTION_H
