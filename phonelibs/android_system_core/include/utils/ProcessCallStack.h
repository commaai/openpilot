/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_PROCESS_CALLSTACK_H
#define ANDROID_PROCESS_CALLSTACK_H

#include <utils/CallStack.h>
#include <android/log.h>
#include <utils/KeyedVector.h>
#include <utils/String8.h>

#include <time.h>
#include <sys/types.h>

namespace android {

class Printer;

// Collect/print the call stack (function, file, line) traces for all threads in a process.
class ProcessCallStack {
public:
    // Create an empty call stack. No-op.
    ProcessCallStack();
    // Copy the existing process callstack (no other side effects).
    ProcessCallStack(const ProcessCallStack& rhs);
    ~ProcessCallStack();

    // Immediately collect the stack traces for all threads.
    void update();

    // Print all stack traces to the log using the supplied logtag.
    void log(const char* logtag, android_LogPriority priority = ANDROID_LOG_DEBUG,
             const char* prefix = 0) const;

    // Dump all stack traces to the specified file descriptor.
    void dump(int fd, int indent = 0, const char* prefix = 0) const;

    // Return a string (possibly very long) containing all the stack traces.
    String8 toString(const char* prefix = 0) const;

    // Dump a serialized representation of all the stack traces to the specified printer.
    void print(Printer& printer) const;

    // Get the number of threads whose stack traces were collected.
    size_t size() const;

private:
    void printInternal(Printer& printer, Printer& csPrinter) const;

    // Reset the process's stack frames and metadata.
    void clear();

    struct ThreadInfo {
        CallStack callStack;
        String8 threadName;
    };

    // tid -> ThreadInfo
    KeyedVector<pid_t, ThreadInfo> mThreadMap;
    // Time that update() was last called
    struct tm mTimeUpdated;
};

}; // namespace android

#endif // ANDROID_PROCESS_CALLSTACK_H
