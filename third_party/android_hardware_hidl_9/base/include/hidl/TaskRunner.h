/*
 * Copyright (C) 2016 The Android Open Source Project
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
#ifndef ANDROID_HIDL_TASK_RUNNER_H
#define ANDROID_HIDL_TASK_RUNNER_H

#include <memory>
#include <thread>

namespace android {
namespace hardware {
namespace details {

using Task = std::function<void(void)>;

template <typename T>
struct SynchronizedQueue;

/*
 * A background infinite loop that runs the Tasks push()'ed.
 * Equivalent to a simple single-threaded Looper.
 */
class TaskRunner {
public:

    /* Create an empty task runner. Nothing will be done until start() is called. */
    TaskRunner();

    /*
     * Notify the background thread to terminate and return immediately.
     * Tasks in the queue will continue to be done sequentially in background
     * until all tasks are finished.
     */
    ~TaskRunner();

    /*
     * Sets the queue limit. Fails the push operation once the limit is reached.
     * This function is named start for legacy reasons and to maintain ABI
     * stability, but the underlying thread running tasks isn't started until
     * the first task is pushed.
     */
    void start(size_t limit);

    /*
     * Add a task. Return true if successful, false if
     * the queue's size exceeds limit or t doesn't contain a callable target.
     */
    bool push(const Task &t);

private:
    std::shared_ptr<SynchronizedQueue<Task>> mQueue;
};

} // namespace details
} // namespace hardware
} // namespace android

#endif // ANDROID_HIDL_TASK_RUNNER_H
