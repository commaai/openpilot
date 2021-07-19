/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef __CUTILS_IOSCHED_POLICY_H
#define __CUTILS_IOSCHED_POLICY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IoSchedClass_NONE,
    IoSchedClass_RT,
    IoSchedClass_BE,
    IoSchedClass_IDLE,
} IoSchedClass;

extern int android_set_ioprio(int pid, IoSchedClass clazz, int ioprio);
extern int android_get_ioprio(int pid, IoSchedClass *clazz, int *ioprio);

extern int android_set_rt_ioprio(int pid, int rt);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_IOSCHED_POLICY_H */ 
