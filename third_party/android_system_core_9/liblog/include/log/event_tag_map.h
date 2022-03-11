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

#ifndef _LIBS_CUTILS_EVENTTAGMAP_H
#define _LIBS_CUTILS_EVENTTAGMAP_H

#ifdef __cplusplus
extern "C" {
#endif

#define EVENT_TAG_MAP_FILE "/system/etc/event-log-tags"

struct EventTagMap;
typedef struct EventTagMap EventTagMap;

/*
 * Open the specified file as an event log tag map.
 *
 * Returns NULL on failure.
 */
EventTagMap* android_openEventTagMap(const char* fileName);

/*
 * Close the map.
 */
void android_closeEventTagMap(EventTagMap* map);

/*
 * Look up a tag by index.  Returns the tag string, or NULL if not found.
 */
const char* android_lookupEventTag(const EventTagMap* map, unsigned int tag)
    __attribute__((
        deprecated("use android_lookupEventTag_len() instead to minimize "
                   "MAP_PRIVATE copy-on-write memory impact")));

/*
 * Look up a tag by index.  Returns the tag string & string length, or NULL if
 * not found.  Returned string is not guaranteed to be nul terminated.
 */
const char* android_lookupEventTag_len(const EventTagMap* map, size_t* len,
                                       unsigned int tag);

/*
 * Look up a format by index. Returns the format string & string length,
 * or NULL if not found. Returned string is not guaranteed to be nul terminated.
 */
const char* android_lookupEventFormat_len(const EventTagMap* map, size_t* len,
                                          unsigned int tag);

/*
 * Look up tagname, generate one if necessary, and return a tag
 */
int android_lookupEventTagNum(EventTagMap* map, const char* tagname,
                              const char* format, int prio);

#ifdef __cplusplus
}
#endif

#endif /*_LIBS_CUTILS_EVENTTAGMAP_H*/
