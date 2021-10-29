/*
 * Copyright (C) 2005 The Android Open Source Project
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

#ifndef ANDROID_PROCESS_STATE_H
#define ANDROID_PROCESS_STATE_H

#include <binder/IBinder.h>
#include <utils/KeyedVector.h>
#include <utils/String8.h>
#include <utils/String16.h>

#include <utils/threads.h>

#include <pthread.h>

// ---------------------------------------------------------------------------
namespace android {

class IPCThreadState;

class ProcessState : public virtual RefBase
{
public:
    static  sp<ProcessState>    self();

            void                setContextObject(const sp<IBinder>& object);
            sp<IBinder>         getContextObject(const sp<IBinder>& caller);
        
            void                setContextObject(const sp<IBinder>& object,
                                                 const String16& name);
            sp<IBinder>         getContextObject(const String16& name,
                                                 const sp<IBinder>& caller);

            void                startThreadPool();
                        
    typedef bool (*context_check_func)(const String16& name,
                                       const sp<IBinder>& caller,
                                       void* userData);
        
            bool                isContextManager(void) const;
            bool                becomeContextManager(
                                    context_check_func checkFunc,
                                    void* userData);

            sp<IBinder>         getStrongProxyForHandle(int32_t handle);
            wp<IBinder>         getWeakProxyForHandle(int32_t handle);
            void                expungeHandle(int32_t handle, IBinder* binder);

            void                spawnPooledThread(bool isMain);
            
            status_t            setThreadPoolMaxThreadCount(size_t maxThreads);
            void                giveThreadPoolName();

private:
    friend class IPCThreadState;
    
                                ProcessState();
                                ~ProcessState();

                                ProcessState(const ProcessState& o);
            ProcessState&       operator=(const ProcessState& o);
            String8             makeBinderThreadName();

            struct handle_entry {
                IBinder* binder;
                RefBase::weakref_type* refs;
            };

            handle_entry*       lookupHandleLocked(int32_t handle);

            int                 mDriverFD;
            void*               mVMStart;

            // Protects thread count variable below.
            pthread_mutex_t     mThreadCountLock;
            pthread_cond_t      mThreadCountDecrement;
            // Number of binder threads current executing a command.
            size_t              mExecutingThreadsCount;
            // Maximum number for binder threads allowed for this process.
            size_t              mMaxThreads;

    mutable Mutex               mLock;  // protects everything below.

            Vector<handle_entry>mHandleToObject;

            bool                mManagesContexts;
            context_check_func  mBinderContextCheckFunc;
            void*               mBinderContextUserData;

            KeyedVector<String16, sp<IBinder> >
                                mContexts;


            String8             mRootDir;
            bool                mThreadPoolStarted;
    volatile int32_t            mThreadPoolSeq;
};
    
}; // namespace android

// ---------------------------------------------------------------------------

#endif // ANDROID_PROCESS_STATE_H
