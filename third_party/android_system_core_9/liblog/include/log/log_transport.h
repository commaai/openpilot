/*
**
** Copyright 2017, The Android Open Source Project
**
** This file is dual licensed.  It may be redistributed and/or modified
** under the terms of the Apache 2.0 License OR version 2 of the GNU
** General Public License.
*/

#ifndef _LIBS_LOG_TRANSPORT_H
#define _LIBS_LOG_TRANSPORT_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Logging transports, bit mask to select features. Function returns selection.
 */
/* clang-format off */
#define LOGGER_DEFAULT 0x00
#define LOGGER_LOGD    0x01
#define LOGGER_KERNEL  0x02 /* Reserved/Deprecated */
#define LOGGER_NULL    0x04 /* Does not release resources of other selections */
#define LOGGER_LOCAL   0x08 /* logs sent to local memory */
#define LOGGER_STDERR  0x10 /* logs sent to stderr */
/* clang-format on */

/* Both return the selected transport flag mask, or negative errno */
int android_set_log_transport(int transport_flag);
int android_get_log_transport();

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_TRANSPORT_H */
