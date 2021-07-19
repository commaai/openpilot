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

//
// C/C++ logging functions.  See the logging documentation for API details.
//
// We'd like these to be available from C code (in case we import some from
// somewhere), so this has a C interface.
//
// The output will be correct when the log file is shared between multiple
// threads and/or multiple processes so long as the operating system
// supports O_APPEND.  These calls have mutex-protected data structures
// and so are NOT reentrant.  Do not use LOG in a signal handler.
//
#ifndef _LIBS_UTILS_LOG_H
#define _LIBS_UTILS_LOG_H

#include <cutils/log.h>
#include <sys/types.h>

#ifdef __cplusplus

namespace android {

/*
 * A very simple utility that yells in the log when an operation takes too long.
 */
class LogIfSlow {
public:
    LogIfSlow(const char* tag, android_LogPriority priority,
            int timeoutMillis, const char* message);
    ~LogIfSlow();

private:
    const char* const mTag;
    const android_LogPriority mPriority;
    const int mTimeoutMillis;
    const char* const mMessage;
    const int64_t mStart;
};

/*
 * Writes the specified debug log message if this block takes longer than the
 * specified number of milliseconds to run.  Includes the time actually taken.
 *
 * {
 *     ALOGD_IF_SLOW(50, "Excessive delay doing something.");
 *     doSomething();
 * }
 */
#define ALOGD_IF_SLOW(timeoutMillis, message) \
    android::LogIfSlow _logIfSlow(LOG_TAG, ANDROID_LOG_DEBUG, timeoutMillis, message);

} // namespace android

#endif // __cplusplus

#endif // _LIBS_UTILS_LOG_H
