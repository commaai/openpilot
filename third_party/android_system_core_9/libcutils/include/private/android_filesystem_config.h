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

/*
 * This file is consumed by build/tools/fs_config and is used
 * for generating various files. Anything #define AID_<name>
 * becomes the mapping for getpwnam/getpwuid, etc. The <name>
 * field is lowercased.
 * For example:
 * #define AID_FOO_BAR 6666 becomes a friendly name of "foo_bar"
 *
 * The above holds true with the exception of:
 *   mediacodec
 *   mediaex
 *   mediadrm
 * Whose friendly names do not match the #define statements.
 *
 * Additionally, AID_OEM_RESERVED_START and AID_OEM_RESERVED_END
 * can be used to define reserved OEM ranges used for sanity checks
 * during the build process. The rules are, they must end with START/END
 * The proper convention is incrementing a number like so:
 * AID_OEM_RESERVED_START
 * AID_OEM_RESERVED_1_START
 * AID_OEM_RESERVED_2_START
 * ...
 * The same applies to the END.
 * They are not required to be in order, but must not overlap each other and
 * must define a START and END'ing range. START must be smaller than END.
 */

#ifndef _ANDROID_FILESYSTEM_CONFIG_H_
#define _ANDROID_FILESYSTEM_CONFIG_H_

#include <sys/types.h>

#if !defined(__ANDROID_VNDK__) && !defined(EXCLUDE_FS_CONFIG_STRUCTURES)
#include <private/fs_config.h>
#endif

/* This is the master Users and Groups config for the platform.
 * DO NOT EVER RENUMBER
 */

#define AID_ROOT 0 /* traditional unix root user */
/* The following are for LTP and should only be used for testing */
#define AID_DAEMON 1 /* traditional unix daemon owner */
#define AID_BIN 2    /* traditional unix binaries owner */

#define AID_SYSTEM 1000 /* system server */

#define AID_RADIO 1001           /* telephony subsystem, RIL */
#define AID_BLUETOOTH 1002       /* bluetooth subsystem */
#define AID_GRAPHICS 1003        /* graphics devices */
#define AID_INPUT 1004           /* input devices */
#define AID_AUDIO 1005           /* audio devices */
#define AID_CAMERA 1006          /* camera devices */
#define AID_LOG 1007             /* log devices */
#define AID_COMPASS 1008         /* compass device */
#define AID_MOUNT 1009           /* mountd socket */
#define AID_WIFI 1010            /* wifi subsystem */
#define AID_ADB 1011             /* android debug bridge (adbd) */
#define AID_INSTALL 1012         /* group for installing packages */
#define AID_MEDIA 1013           /* mediaserver process */
#define AID_DHCP 1014            /* dhcp client */
#define AID_SDCARD_RW 1015       /* external storage write access */
#define AID_VPN 1016             /* vpn system */
#define AID_KEYSTORE 1017        /* keystore subsystem */
#define AID_USB 1018             /* USB devices */
#define AID_DRM 1019             /* DRM server */
#define AID_MDNSR 1020           /* MulticastDNSResponder (service discovery) */
#define AID_GPS 1021             /* GPS daemon */
#define AID_UNUSED1 1022         /* deprecated, DO NOT USE */
#define AID_MEDIA_RW 1023        /* internal media storage write access */
#define AID_MTP 1024             /* MTP USB driver access */
#define AID_UNUSED2 1025         /* deprecated, DO NOT USE */
#define AID_DRMRPC 1026          /* group for drm rpc */
#define AID_NFC 1027             /* nfc subsystem */
#define AID_SDCARD_R 1028        /* external storage read access */
#define AID_CLAT 1029            /* clat part of nat464 */
#define AID_LOOP_RADIO 1030      /* loop radio devices */
#define AID_MEDIA_DRM 1031       /* MediaDrm plugins */
#define AID_PACKAGE_INFO 1032    /* access to installed package details */
#define AID_SDCARD_PICS 1033     /* external storage photos access */
#define AID_SDCARD_AV 1034       /* external storage audio/video access */
#define AID_SDCARD_ALL 1035      /* access all users external storage */
#define AID_LOGD 1036            /* log daemon */
#define AID_SHARED_RELRO 1037    /* creator of shared GNU RELRO files */
#define AID_DBUS 1038            /* dbus-daemon IPC broker process */
#define AID_TLSDATE 1039         /* tlsdate unprivileged user */
#define AID_MEDIA_EX 1040        /* mediaextractor process */
#define AID_AUDIOSERVER 1041     /* audioserver process */
#define AID_METRICS_COLL 1042    /* metrics_collector process */
#define AID_METRICSD 1043        /* metricsd process */
#define AID_WEBSERV 1044         /* webservd process */
#define AID_DEBUGGERD 1045       /* debuggerd unprivileged user */
#define AID_MEDIA_CODEC 1046     /* mediacodec process */
#define AID_CAMERASERVER 1047    /* cameraserver process */
#define AID_FIREWALL 1048        /* firewalld process */
#define AID_TRUNKS 1049          /* trunksd process (TPM daemon) */
#define AID_NVRAM 1050           /* Access-controlled NVRAM */
#define AID_DNS 1051             /* DNS resolution daemon (system: netd) */
#define AID_DNS_TETHER 1052      /* DNS resolution daemon (tether: dnsmasq) */
#define AID_WEBVIEW_ZYGOTE 1053  /* WebView zygote process */
#define AID_VEHICLE_NETWORK 1054 /* Vehicle network service */
#define AID_MEDIA_AUDIO 1055     /* GID for audio files on internal media storage */
#define AID_MEDIA_VIDEO 1056     /* GID for video files on internal media storage */
#define AID_MEDIA_IMAGE 1057     /* GID for image files on internal media storage */
#define AID_TOMBSTONED 1058      /* tombstoned user */
#define AID_MEDIA_OBB 1059       /* GID for OBB files on internal media storage */
#define AID_ESE 1060             /* embedded secure element (eSE) subsystem */
#define AID_OTA_UPDATE 1061      /* resource tracking UID for OTA updates */
#define AID_AUTOMOTIVE_EVS 1062  /* Automotive rear and surround view system */
#define AID_LOWPAN 1063          /* LoWPAN subsystem */
#define AID_HSM 1064             /* hardware security module subsystem */
#define AID_RESERVED_DISK 1065   /* GID that has access to reserved disk space */
#define AID_STATSD 1066          /* statsd daemon */
#define AID_INCIDENTD 1067       /* incidentd daemon */
#define AID_SECURE_ELEMENT 1068  /* secure element subsystem */
/* Changes to this file must be made in AOSP, *not* in internal branches. */

#define AID_SHELL 2000 /* adb and debug shell user */
#define AID_CACHE 2001 /* cache access */
#define AID_DIAG 2002  /* access to diagnostic resources */

/* The range 2900-2999 is reserved for OEM, and must never be
 * used here */
#define AID_OEM_RESERVED_START 2900
#define AID_OEM_RESERVED_END 2999

/* The 3000 series are intended for use as supplemental group id's only.
 * They indicate special Android capabilities that the kernel is aware of. */
#define AID_NET_BT_ADMIN 3001 /* bluetooth: create any socket */
#define AID_NET_BT 3002       /* bluetooth: create sco, rfcomm or l2cap sockets */
#define AID_INET 3003         /* can create AF_INET and AF_INET6 sockets */
#define AID_NET_RAW 3004      /* can create raw INET sockets */
#define AID_NET_ADMIN 3005    /* can configure interfaces and routing tables. */
#define AID_NET_BW_STATS 3006 /* read bandwidth statistics */
#define AID_NET_BW_ACCT 3007  /* change bandwidth statistics accounting */
#define AID_READPROC 3009     /* Allow /proc read access */
#define AID_WAKELOCK 3010     /* Allow system wakelock read/write access */
#define AID_UHID 3011         /* Allow read/write to /dev/uhid node */

/* The range 5000-5999 is also reserved for OEM, and must never be used here. */
#define AID_OEM_RESERVED_2_START 5000
#define AID_OEM_RESERVED_2_END 5999

#define AID_EVERYBODY 9997 /* shared between all apps in the same profile */
#define AID_MISC 9998      /* access to misc storage */
#define AID_NOBODY 9999

#define AID_APP 10000       /* TODO: switch users over to AID_APP_START */
#define AID_APP_START 10000 /* first app user */
#define AID_APP_END 19999   /* last app user */

#define AID_CACHE_GID_START 20000 /* start of gids for apps to mark cached data */
#define AID_CACHE_GID_END 29999   /* end of gids for apps to mark cached data */

#define AID_EXT_GID_START 30000 /* start of gids for apps to mark external data */
#define AID_EXT_GID_END 39999   /* end of gids for apps to mark external data */

#define AID_EXT_CACHE_GID_START 40000 /* start of gids for apps to mark external cached data */
#define AID_EXT_CACHE_GID_END 49999   /* end of gids for apps to mark external cached data */

#define AID_SHARED_GID_START 50000 /* start of gids for apps in each user to share */
#define AID_SHARED_GID_END 59999   /* end of gids for apps in each user to share */

/*
 * This is a magic number in the kernel and not something that was picked
 * arbitrarily. This value is returned whenever a uid that has no mapping in the
 * user namespace is returned to userspace:
 * https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/linux/highuid.h?h=v4.4#n40
 */
#define AID_OVERFLOWUID 65534 /* unmapped user in the user namespace */

#define AID_ISOLATED_START 99000 /* start of uids for fully isolated sandboxed processes */
#define AID_ISOLATED_END 99999   /* end of uids for fully isolated sandboxed processes */

#define AID_USER 100000        /* TODO: switch users over to AID_USER_OFFSET */
#define AID_USER_OFFSET 100000 /* offset for uid ranges for each user */

/*
 * android_ids has moved to pwd/grp functionality.
 * If you need to add one, the structure is now
 * auto-generated based on the AID_ constraints
 * documented at the top of this header file.
 * Also see build/tools/fs_config for more details.
 */

#endif
