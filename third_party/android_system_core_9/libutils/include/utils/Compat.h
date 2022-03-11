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

#ifndef __LIB_UTILS_COMPAT_H
#define __LIB_UTILS_COMPAT_H

#include <unistd.h>

#if defined(__APPLE__)

/* Mac OS has always had a 64-bit off_t, so it doesn't have off64_t. */

typedef off_t off64_t;

static inline off64_t lseek64(int fd, off64_t offset, int whence) {
    return lseek(fd, offset, whence);
}

static inline ssize_t pread64(int fd, void* buf, size_t nbytes, off64_t offset) {
    return pread(fd, buf, nbytes, offset);
}

static inline ssize_t pwrite64(int fd, const void* buf, size_t nbytes, off64_t offset) {
    return pwrite(fd, buf, nbytes, offset);
}

static inline int ftruncate64(int fd, off64_t length) {
    return ftruncate(fd, length);
}

#endif /* __APPLE__ */

#if defined(_WIN32)
#define O_CLOEXEC O_NOINHERIT
#define O_NOFOLLOW 0
#define DEFFILEMODE 0666
#endif /* _WIN32 */

#define ZD "%zd"
#define ZD_TYPE ssize_t

/*
 * Needed for cases where something should be constexpr if possible, but not
 * being constexpr is fine if in pre-C++11 code (such as a const static float
 * member variable).
 */
#if __cplusplus >= 201103L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

/*
 * TEMP_FAILURE_RETRY is defined by some, but not all, versions of
 * <unistd.h>. (Alas, it is not as standard as we'd hoped!) So, if it's
 * not already defined, then define it here.
 */
#ifndef TEMP_FAILURE_RETRY
/* Used to retry syscalls that can return EINTR. */
#define TEMP_FAILURE_RETRY(exp) ({         \
    typeof (exp) _rc;                      \
    do {                                   \
        _rc = (exp);                       \
    } while (_rc == -1 && errno == EINTR); \
    _rc; })
#endif

#if defined(_WIN32)
#define OS_PATH_SEPARATOR '\\'
#else
#define OS_PATH_SEPARATOR '/'
#endif

#endif /* __LIB_UTILS_COMPAT_H */
