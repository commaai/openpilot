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

#ifndef __CUTILS_MULTIUSER_H
#define __CUTILS_MULTIUSER_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uid_t userid_t;
typedef uid_t appid_t;

extern userid_t multiuser_get_user_id(uid_t uid);
extern appid_t multiuser_get_app_id(uid_t uid);

extern uid_t multiuser_get_uid(userid_t user_id, appid_t app_id);

extern gid_t multiuser_get_cache_gid(userid_t user_id, appid_t app_id);
extern gid_t multiuser_get_ext_gid(userid_t user_id, appid_t app_id);
extern gid_t multiuser_get_ext_cache_gid(userid_t user_id, appid_t app_id);
extern gid_t multiuser_get_shared_gid(userid_t user_id, appid_t app_id);

/* TODO: switch callers over to multiuser_get_shared_gid() */
extern gid_t multiuser_get_shared_app_gid(uid_t uid);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_MULTIUSER_H */
