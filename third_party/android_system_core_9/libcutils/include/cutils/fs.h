/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef __CUTILS_FS_H
#define __CUTILS_FS_H

#include <sys/types.h>
#include <unistd.h>

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

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Ensure that directory exists with given mode and owners.  If it exists
 * with a different mode or owners, they are fixed to match the given values.
 */
extern int fs_prepare_dir(const char* path, mode_t mode, uid_t uid, gid_t gid);

/*
 * Ensure that directory exists with given mode and owners.  If it exists
 * with different owners, they are not fixed and -1 is returned.
 */
extern int fs_prepare_dir_strict(const char* path, mode_t mode, uid_t uid, gid_t gid);

/*
 * Ensure that file exists with given mode and owners.  If it exists
 * with different owners, they are not fixed and -1 is returned.
 */
extern int fs_prepare_file_strict(const char* path, mode_t mode, uid_t uid, gid_t gid);


/*
 * Read single plaintext integer from given file, correctly handling files
 * partially written with fs_write_atomic_int().
 */
extern int fs_read_atomic_int(const char* path, int* value);

/*
 * Write single plaintext integer to given file, creating backup while
 * in progress.
 */
extern int fs_write_atomic_int(const char* path, int value);

/*
 * Ensure that all directories along given path exist, creating parent
 * directories as needed.  Validates that given path is absolute and that
 * it contains no relative "." or ".." paths or symlinks.  Last path segment
 * is treated as filename and ignored, unless the path ends with "/".
 */
extern int fs_mkdirs(const char* path, mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_FS_H */
