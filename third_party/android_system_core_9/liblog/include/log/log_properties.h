/*
**
** Copyright 2017, The Android Open Source Project
**
** This file is dual licensed.  It may be redistributed and/or modified
** under the terms of the Apache 2.0 License OR version 2 of the GNU
** General Public License.
*/

#ifndef _LIBS_LOG_PROPERTIES_H
#define _LIBS_LOG_PROPERTIES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __ANDROID_USE_LIBLOG_IS_DEBUGGABLE_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_IS_DEBUGGABLE_INTERFACE 1
#elif __ANDROID_API__ > 24 /* > Nougat */
#define __ANDROID_USE_LIBLOG_IS_DEBUGGABLE_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_IS_DEBUGGABLE_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_IS_DEBUGGABLE_INTERFACE
int __android_log_is_debuggable();
#endif

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_PROPERTIES_H */
