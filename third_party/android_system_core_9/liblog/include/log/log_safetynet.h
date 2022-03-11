/*
**
** Copyright 2017, The Android Open Source Project
**
** This file is dual licensed.  It may be redistributed and/or modified
** under the terms of the Apache 2.0 License OR version 2 of the GNU
** General Public License.
*/

#ifndef _LIBS_LOG_SAFETYNET_H
#define _LIBS_LOG_SAFETYNET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE 1
#elif __ANDROID_API__ > 22 /* > Lollipop */
#define __ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE

#define android_errorWriteLog(tag, subTag) \
  __android_log_error_write(tag, subTag, -1, NULL, 0)

#define android_errorWriteWithInfoLog(tag, subTag, uid, data, dataLen) \
  __android_log_error_write(tag, subTag, uid, data, dataLen)

int __android_log_error_write(int tag, const char* subTag, int32_t uid,
                              const char* data, uint32_t dataLen);

#endif /* __ANDROID_USE_LIBLOG_SAFETYNET_INTERFACE */

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_SAFETYNET_H */
