/*
 * Copyright 2013 The Android Open Source Project
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

#ifndef ANDROID_DISCONNECT_WAITER_H
#define ANDROID_DISCONNECT_WAITER_H

#include <gui/IConsumerListener.h>

#include <utils/Condition.h>
#include <utils/Mutex.h>

namespace android {

// Note that GLConsumer will lose the notifications
// onBuffersReleased and onFrameAvailable as there is currently
// no way to forward the events.  This DisconnectWaiter will not let the
// disconnect finish until finishDisconnect() is called.  It will
// also block until a disconnect is called
class DisconnectWaiter : public BnConsumerListener {
public:
    DisconnectWaiter () :
        mWaitForDisconnect(false),
        mPendingFrames(0) {
    }

    void waitForFrame() {
        Mutex::Autolock lock(mMutex);
        while (mPendingFrames == 0) {
            mFrameCondition.wait(mMutex);
        }
        mPendingFrames--;
    }

    virtual void onFrameAvailable(const BufferItem& /* item */) {
        Mutex::Autolock lock(mMutex);
        mPendingFrames++;
        mFrameCondition.signal();
    }

    virtual void onBuffersReleased() {
        Mutex::Autolock lock(mMutex);
        while (!mWaitForDisconnect) {
            mDisconnectCondition.wait(mMutex);
        }
    }

    virtual void onSidebandStreamChanged() {}

    void finishDisconnect() {
        Mutex::Autolock lock(mMutex);
        mWaitForDisconnect = true;
        mDisconnectCondition.signal();
    }

private:
    Mutex mMutex;

    bool mWaitForDisconnect;
    Condition mDisconnectCondition;

    int mPendingFrames;
    Condition mFrameCondition;
};

} // namespace android

#endif
