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

#ifndef ANDROID_FRAME_WAITER_H
#define ANDROID_FRAME_WAITER_H

#include <gui/GLConsumer.h>

namespace android {

class FrameWaiter : public GLConsumer::FrameAvailableListener {
public:
    FrameWaiter():
            mPendingFrames(0) {
    }

    void waitForFrame() {
        Mutex::Autolock lock(mMutex);
        while (mPendingFrames == 0) {
            mCondition.wait(mMutex);
        }
        mPendingFrames--;
    }

    virtual void onFrameAvailable(const BufferItem& /* item */) {
        Mutex::Autolock lock(mMutex);
        mPendingFrames++;
        mCondition.signal();
    }

private:
    int mPendingFrames;
    Mutex mMutex;
    Condition mCondition;
};

} // namespace android

#endif
