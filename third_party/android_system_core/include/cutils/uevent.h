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

#ifndef __CUTILS_UEVENT_H
#define __CUTILS_UEVENT_H

#include <stdbool.h>
#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif

int uevent_open_socket(int buf_sz, bool passcred);
ssize_t uevent_kernel_multicast_recv(int socket, void *buffer, size_t length);
ssize_t uevent_kernel_multicast_uid_recv(int socket, void *buffer, size_t length, uid_t *uid);
ssize_t uevent_kernel_recv(int socket, void *buffer, size_t length, bool require_group, uid_t *uid);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_UEVENT_H */
