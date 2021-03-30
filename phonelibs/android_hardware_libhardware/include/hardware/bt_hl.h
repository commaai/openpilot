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

#ifndef ANDROID_INCLUDE_BT_HL_H
#define ANDROID_INCLUDE_BT_HL_H

__BEGIN_DECLS

/* HL connection states */

typedef enum
{
    BTHL_MDEP_ROLE_SOURCE,
    BTHL_MDEP_ROLE_SINK
} bthl_mdep_role_t;

typedef enum {
    BTHL_APP_REG_STATE_REG_SUCCESS,
    BTHL_APP_REG_STATE_REG_FAILED,
    BTHL_APP_REG_STATE_DEREG_SUCCESS,
    BTHL_APP_REG_STATE_DEREG_FAILED
} bthl_app_reg_state_t;

typedef enum
{
    BTHL_CHANNEL_TYPE_RELIABLE,
    BTHL_CHANNEL_TYPE_STREAMING,
    BTHL_CHANNEL_TYPE_ANY
} bthl_channel_type_t;


/* HL connection states */
typedef enum {
    BTHL_CONN_STATE_CONNECTING,
    BTHL_CONN_STATE_CONNECTED,
    BTHL_CONN_STATE_DISCONNECTING,
    BTHL_CONN_STATE_DISCONNECTED,
    BTHL_CONN_STATE_DESTROYED
} bthl_channel_state_t;

typedef struct
{
    bthl_mdep_role_t        mdep_role;
    int                     data_type;
    bthl_channel_type_t     channel_type;
    const char                   *mdep_description; /* MDEP description to be used in the SDP (optional); null terminated */
} bthl_mdep_cfg_t;

typedef struct
{
    const char      *application_name;
    const char      *provider_name;   /* provider name to be used in the SDP (optional); null terminated */
    const char      *srv_name;        /* service name to be used in the SDP (optional); null terminated*/
    const char      *srv_desp;        /* service description to be used in the SDP (optional); null terminated */
    int             number_of_mdeps;
    bthl_mdep_cfg_t *mdep_cfg;  /* Dynamic array */
} bthl_reg_param_t;

/** Callback for application registration status.
 *  state will have one of the values from  bthl_app_reg_state_t
 */
typedef void (* bthl_app_reg_state_callback)(int app_id, bthl_app_reg_state_t state);

/** Callback for channel connection state change.
 *  state will have one of the values from
 *  bthl_connection_state_t and fd (file descriptor)
 */
typedef void (* bthl_channel_state_callback)(int app_id, bt_bdaddr_t *bd_addr, int mdep_cfg_index, int channel_id, bthl_channel_state_t state, int fd);

/** BT-HL callback structure. */
typedef struct {
    /** set to sizeof(bthl_callbacks_t) */
    size_t      size;
    bthl_app_reg_state_callback     app_reg_state_cb;
    bthl_channel_state_callback     channel_state_cb;
} bthl_callbacks_t;


/** Represents the standard BT-HL interface. */
typedef struct {

    /** set to sizeof(bthl_interface_t)  */
    size_t          size;

    /**
     * Register the Bthl callbacks
     */
    bt_status_t (*init)( bthl_callbacks_t* callbacks );

    /** Register HL application */
    bt_status_t (*register_application) ( bthl_reg_param_t *p_reg_param, int *app_id);

    /** Unregister HL application */
    bt_status_t (*unregister_application) (int app_id);

    /** connect channel */
    bt_status_t (*connect_channel)(int app_id, bt_bdaddr_t *bd_addr, int mdep_cfg_index, int *channel_id);

    /** destroy channel */
    bt_status_t (*destroy_channel)(int channel_id);

    /** Close the  Bthl callback **/
    void (*cleanup)(void);

} bthl_interface_t;
__END_DECLS

#endif /* ANDROID_INCLUDE_BT_HL_H */


