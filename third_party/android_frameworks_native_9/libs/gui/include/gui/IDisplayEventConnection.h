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

#pragma once

#include <binder/IInterface.h>
#include <binder/SafeInterface.h>

#include <utils/Errors.h>

#include <cstdint>

namespace android {

namespace gui {
class BitTube;
} // namespace gui

class IDisplayEventConnection : public IInterface {
public:
    DECLARE_META_INTERFACE(DisplayEventConnection)

    /*
     * stealReceiveChannel() returns a BitTube to receive events from. Only the receive file
     * descriptor of outChannel will be initialized, and this effectively "steals" the receive
     * channel from the remote end (such that the remote end can only use its send channel).
     */
    virtual status_t stealReceiveChannel(gui::BitTube* outChannel) = 0;

    /*
     * setVsyncRate() sets the vsync event delivery rate. A value of 1 returns every vsync event.
     * A value of 2 returns every other event, etc. A value of 0 returns no event unless
     * requestNextVsync() has been called.
     */
    virtual status_t setVsyncRate(uint32_t count) = 0;

    /*
     * requestNextVsync() schedules the next vsync event. It has no effect if the vsync rate is > 0.
     */
    virtual void requestNextVsync() = 0; // Asynchronous
};

class BnDisplayEventConnection : public SafeBnInterface<IDisplayEventConnection> {
public:
    BnDisplayEventConnection()
          : SafeBnInterface<IDisplayEventConnection>("BnDisplayEventConnection") {}

    status_t onTransact(uint32_t code, const Parcel& data, Parcel* reply,
                        uint32_t flags = 0) override;
};

} // namespace android
