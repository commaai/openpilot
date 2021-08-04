/*
 * Copyright (C) 2006 The Android Open Source Project
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

#ifndef ANDROID_BUFFEREDTEXTOUTPUT_H
#define ANDROID_BUFFEREDTEXTOUTPUT_H

#include <binder/TextOutput.h>
#include <utils/threads.h>
#include <sys/uio.h>

// ---------------------------------------------------------------------------
namespace android {

class BufferedTextOutput : public TextOutput
{
public:
    //** Flags for constructor */
    enum {
        MULTITHREADED = 0x0001
    };
    
                        BufferedTextOutput(uint32_t flags = 0);
    virtual             ~BufferedTextOutput();
    
    virtual status_t    print(const char* txt, size_t len);
    virtual void        moveIndent(int delta);
    
    virtual void        pushBundle();
    virtual void        popBundle();
    
protected:
    virtual status_t    writeLines(const struct iovec& vec, size_t N) = 0;

private:
    struct BufferState;
    struct ThreadState;
    
    static  ThreadState*getThreadState();
    static  void        threadDestructor(void *st);
    
            BufferState*getBuffer() const;
            
    uint32_t            mFlags;
    const int32_t       mSeq;
    const int32_t       mIndex;
    
    Mutex               mLock;
    BufferState*        mGlobalState;
};

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_BUFFEREDTEXTOUTPUT_H
