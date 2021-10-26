/*
 * Copyright (c) 2013, The Linux Foundation. All rights reserved.
 * Not a Contribution
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

#ifndef ANDROID_INCLUDE_BT_HD_H
#define ANDROID_INCLUDE_BT_HD_H

#include <stdint.h>

__BEGIN_DECLS

typedef enum
{
    BTHD_REPORT_TYPE_OTHER = 0,
    BTHD_REPORT_TYPE_INPUT,
    BTHD_REPORT_TYPE_OUTPUT,
    BTHD_REPORT_TYPE_FEATURE,
    BTHD_REPORT_TYPE_INTRDATA // special value for reports to be sent on INTR (INPUT is assumed)
} bthd_report_type_t;

typedef enum
{
    BTHD_APP_STATE_NOT_REGISTERED,
    BTHD_APP_STATE_REGISTERED
} bthd_application_state_t;

typedef enum
{
    BTHD_CONN_STATE_CONNECTED,
    BTHD_CONN_STATE_CONNECTING,
    BTHD_CONN_STATE_DISCONNECTED,
    BTHD_CONN_STATE_DISCONNECTING,
    BTHD_CONN_STATE_UNKNOWN
} bthd_connection_state_t;

typedef struct
{
    const char      *name;
    const char      *description;
    const char      *provider;
    uint8_t         subclass;
    uint8_t         *desc_list;
    int             desc_list_len;
} bthd_app_param_t;

typedef struct
{
    uint8_t  service_type;
    uint32_t token_rate;
    uint32_t token_bucket_size;
    uint32_t peak_bandwidth;
    uint32_t access_latency;
    uint32_t delay_variation;
} bthd_qos_param_t;

typedef void (* bthd_application_state_callback)(bt_bdaddr_t *bd_addr, bthd_application_state_t state);
typedef void (* bthd_connection_state_callback)(bt_bdaddr_t *bd_addr, bthd_connection_state_t state);
typedef void (* bthd_get_report_callback)(uint8_t type, uint8_t id, uint16_t buffer_size);
typedef void (* bthd_set_report_callback)(uint8_t type, uint8_t id, uint16_t len, uint8_t *p_data);
typedef void (* bthd_set_protocol_callback)(uint8_t protocol);
typedef void (* bthd_intr_data_callback)(uint8_t report_id, uint16_t len, uint8_t *p_data);
typedef void (* bthd_vc_unplug_callback)(void);

/** BT-HD callbacks */
typedef struct {
    size_t      size;

    bthd_application_state_callback application_state_cb;
    bthd_connection_state_callback  connection_state_cb;
    bthd_get_report_callback        get_report_cb;
    bthd_set_report_callback        set_report_cb;
    bthd_set_protocol_callback      set_protocol_cb;
    bthd_intr_data_callback         intr_data_cb;
    bthd_vc_unplug_callback         vc_unplug_cb;
} bthd_callbacks_t;

/** BT-HD interface */
typedef struct {

    size_t          size;

    /** init interface and register callbacks */
    bt_status_t (*init)(bthd_callbacks_t* callbacks);

    /** close interface */
    void  (*cleanup)(void);

    /** register application */
    bt_status_t (*register_app)(bthd_app_param_t *app_param, bthd_qos_param_t *in_qos,
                                            bthd_qos_param_t *out_qos);

    /** unregister application */
    bt_status_t (*unregister_app)(void);

    /** connects to host with virtual cable */
    bt_status_t (*connect)(void);

    /** disconnects from currently connected host */
    bt_status_t (*disconnect)(void);

    /** send report */
    bt_status_t (*send_report)(bthd_report_type_t type, uint8_t id, uint16_t len, uint8_t *p_data);

    /** notifies error for invalid SET_REPORT */
    bt_status_t (*report_error)(uint8_t error);

    /** send Virtual Cable Unplug  */
    bt_status_t (*virtual_cable_unplug)(void);

} bthd_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_HD_H */

