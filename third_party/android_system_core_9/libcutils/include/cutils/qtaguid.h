/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __CUTILS_QTAGUID_H
#define __CUTILS_QTAGUID_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Set tags (and owning UIDs) for network sockets.
 */
extern int qtaguid_tagSocket(int sockfd, int tag, uid_t uid);

/*
 * Untag a network socket before closing.
 */
extern int qtaguid_untagSocket(int sockfd);

/*
 * For the given uid, switch counter sets.
 * The kernel only keeps a limited number of sets.
 * 2 for now.
 */
extern int qtaguid_setCounterSet(int counterSetNum, uid_t uid);

/*
 * Delete all tag info that relates to the given tag an uid.
 * If the tag is 0, then ALL info about the uid is freed.
 * The delete data also affects active tagged sockets, which are
 * then untagged.
 * The calling process can only operate on its own tags.
 * Unless it is part of the happy AID_NET_BW_ACCT group.
 * In which case it can clobber everything.
 */
extern int qtaguid_deleteTagData(int tag, uid_t uid);

/*
 * Enable/disable qtaguid functionnality at a lower level.
 * When pacified, the kernel will accept commands but do nothing.
 */
extern int qtaguid_setPacifier(int on);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_QTAG_UID_H */
