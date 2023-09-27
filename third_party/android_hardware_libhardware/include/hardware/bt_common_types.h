/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/******************************************************************************
 *
 *  This file contains constants and definitions that can be used commonly between JNI and stack layer
 *
 ******************************************************************************/
#ifndef ANDROID_INCLUDE_BT_COMMON_TYPES_H
#define ANDROID_INCLUDE_BT_COMMON_TYPES_H

#include "bluetooth.h"

typedef struct
{
    uint8_t  client_if;
    uint8_t  filt_index;
    uint8_t  advertiser_state;
    uint8_t  advertiser_info_present;
    uint8_t  addr_type;
    uint8_t  tx_power;
    int8_t  rssi_value;
    uint16_t time_stamp;
    bt_bdaddr_t bd_addr;
    uint8_t  adv_pkt_len;
    uint8_t  *p_adv_pkt_data;
    uint8_t  scan_rsp_len;
    uint8_t  *p_scan_rsp_data;
} btgatt_track_adv_info_t;

#endif  /* ANDROID_INCLUDE_BT_COMMON_TYPES_H */
