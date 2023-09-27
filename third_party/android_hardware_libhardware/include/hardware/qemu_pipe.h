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
#ifndef ANDROID_INCLUDE_HARDWARE_QEMU_PIPE_H
#define ANDROID_INCLUDE_HARDWARE_QEMU_PIPE_H

#include <sys/cdefs.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>  /* for pthread_once() */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifndef D
#  define  D(...)   do{}while(0)
#endif

/* Try to open a new Qemu fast-pipe. This function returns a file descriptor
 * that can be used to communicate with a named service managed by the
 * emulator.
 *
 * This file descriptor can be used as a standard pipe/socket descriptor.
 *
 * 'pipeName' is the name of the emulator service you want to connect to.
 * E.g. 'opengles' or 'camera'.
 *
 * On success, return a valid file descriptor
 * Returns -1 on error, and errno gives the error code, e.g.:
 *
 *    EINVAL  -> unknown/unsupported pipeName
 *    ENOSYS  -> fast pipes not available in this system.
 *
 * ENOSYS should never happen, except if you're trying to run within a
 * misconfigured emulator.
 *
 * You should be able to open several pipes to the same pipe service,
 * except for a few special cases (e.g. GSM modem), where EBUSY will be
 * returned if more than one client tries to connect to it.
 */
static __inline__ int
qemu_pipe_open(const char*  pipeName)
{
    char  buff[256];
    int   buffLen;
    int   fd, ret;

    if (pipeName == NULL || pipeName[0] == '\0') {
        errno = EINVAL;
        return -1;
    }

    snprintf(buff, sizeof buff, "pipe:%s", pipeName);

    fd = open("/dev/qemu_pipe", O_RDWR);
    if (fd < 0 && errno == ENOENT)
        fd = open("/dev/goldfish_pipe", O_RDWR);
    if (fd < 0) {
        D("%s: Could not open /dev/qemu_pipe: %s", __FUNCTION__, strerror(errno));
        //errno = ENOSYS;
        return -1;
    }

    buffLen = strlen(buff);

    ret = TEMP_FAILURE_RETRY(write(fd, buff, buffLen+1));
    if (ret != buffLen+1) {
        D("%s: Could not connect to %s pipe service: %s", __FUNCTION__, pipeName, strerror(errno));
        if (ret == 0) {
            errno = ECONNRESET;
        } else if (ret > 0) {
            errno = EINVAL;
        }
        return -1;
    }

    return fd;
}

#endif /* ANDROID_INCLUDE_HARDWARE_QEMUD_PIPE_H */
