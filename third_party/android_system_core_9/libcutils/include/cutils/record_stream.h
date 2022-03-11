/*
 * Copyright (C) 2006 The Android Open Source Project
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

/*
 * A simple utility for reading fixed records out of a stream fd
 */

#ifndef _CUTILS_RECORD_STREAM_H
#define _CUTILS_RECORD_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct RecordStream RecordStream;

extern RecordStream *record_stream_new(int fd, size_t maxRecordLen);
extern void record_stream_free(RecordStream *p_rs);

extern int record_stream_get_next (RecordStream *p_rs, void ** p_outRecord, 
                                    size_t *p_outRecordLen);

#ifdef __cplusplus
}
#endif


#endif /*_CUTILS_RECORD_STREAM_H*/

