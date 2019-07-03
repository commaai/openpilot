/*
 * Copyright (C) 2012-2014 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_BT_HF_CLIENT_H
#define ANDROID_INCLUDE_BT_HF_CLIENT_H

__BEGIN_DECLS

typedef enum {
    BTHF_CLIENT_CONNECTION_STATE_DISCONNECTED = 0,
    BTHF_CLIENT_CONNECTION_STATE_CONNECTING,
    BTHF_CLIENT_CONNECTION_STATE_CONNECTED,
    BTHF_CLIENT_CONNECTION_STATE_SLC_CONNECTED,
    BTHF_CLIENT_CONNECTION_STATE_DISCONNECTING
} bthf_client_connection_state_t;

typedef enum {
    BTHF_CLIENT_AUDIO_STATE_DISCONNECTED = 0,
    BTHF_CLIENT_AUDIO_STATE_CONNECTING,
    BTHF_CLIENT_AUDIO_STATE_CONNECTED,
    BTHF_CLIENT_AUDIO_STATE_CONNECTED_MSBC,
} bthf_client_audio_state_t;

typedef enum {
    BTHF_CLIENT_VR_STATE_STOPPED = 0,
    BTHF_CLIENT_VR_STATE_STARTED
} bthf_client_vr_state_t;

typedef enum {
    BTHF_CLIENT_VOLUME_TYPE_SPK = 0,
    BTHF_CLIENT_VOLUME_TYPE_MIC
} bthf_client_volume_type_t;

typedef enum
{
    BTHF_CLIENT_NETWORK_STATE_NOT_AVAILABLE = 0,
    BTHF_CLIENT_NETWORK_STATE_AVAILABLE
} bthf_client_network_state_t;

typedef enum
{
    BTHF_CLIENT_SERVICE_TYPE_HOME = 0,
    BTHF_CLIENT_SERVICE_TYPE_ROAMING
} bthf_client_service_type_t;

typedef enum {
    BTHF_CLIENT_CALL_STATE_ACTIVE = 0,
    BTHF_CLIENT_CALL_STATE_HELD,
    BTHF_CLIENT_CALL_STATE_DIALING,
    BTHF_CLIENT_CALL_STATE_ALERTING,
    BTHF_CLIENT_CALL_STATE_INCOMING,
    BTHF_CLIENT_CALL_STATE_WAITING,
    BTHF_CLIENT_CALL_STATE_HELD_BY_RESP_HOLD,
} bthf_client_call_state_t;

typedef enum {
    BTHF_CLIENT_CALL_NO_CALLS_IN_PROGRESS = 0,
    BTHF_CLIENT_CALL_CALLS_IN_PROGRESS
} bthf_client_call_t;

typedef enum {
    BTHF_CLIENT_CALLSETUP_NONE = 0,
    BTHF_CLIENT_CALLSETUP_INCOMING,
    BTHF_CLIENT_CALLSETUP_OUTGOING,
    BTHF_CLIENT_CALLSETUP_ALERTING

} bthf_client_callsetup_t;

typedef enum {
    BTHF_CLIENT_CALLHELD_NONE = 0,
    BTHF_CLIENT_CALLHELD_HOLD_AND_ACTIVE,
    BTHF_CLIENT_CALLHELD_HOLD,
} bthf_client_callheld_t;

typedef enum {
    BTHF_CLIENT_RESP_AND_HOLD_HELD = 0,
    BTRH_CLIENT_RESP_AND_HOLD_ACCEPT,
    BTRH_CLIENT_RESP_AND_HOLD_REJECT,
} bthf_client_resp_and_hold_t;

typedef enum {
    BTHF_CLIENT_CALL_DIRECTION_OUTGOING = 0,
    BTHF_CLIENT_CALL_DIRECTION_INCOMING
} bthf_client_call_direction_t;

typedef enum {
    BTHF_CLIENT_CALL_MPTY_TYPE_SINGLE = 0,
    BTHF_CLIENT_CALL_MPTY_TYPE_MULTI
} bthf_client_call_mpty_type_t;

typedef enum {
    BTHF_CLIENT_CMD_COMPLETE_OK = 0,
    BTHF_CLIENT_CMD_COMPLETE_ERROR,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_NO_CARRIER,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_BUSY,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_NO_ANSWER,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_DELAYED,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_BLACKLISTED,
    BTHF_CLIENT_CMD_COMPLETE_ERROR_CME
} bthf_client_cmd_complete_t;

typedef enum {
    BTHF_CLIENT_CALL_ACTION_CHLD_0 = 0,
    BTHF_CLIENT_CALL_ACTION_CHLD_1,
    BTHF_CLIENT_CALL_ACTION_CHLD_2,
    BTHF_CLIENT_CALL_ACTION_CHLD_3,
    BTHF_CLIENT_CALL_ACTION_CHLD_4,
    BTHF_CLIENT_CALL_ACTION_CHLD_1x,
    BTHF_CLIENT_CALL_ACTION_CHLD_2x,
    BTHF_CLIENT_CALL_ACTION_ATA,
    BTHF_CLIENT_CALL_ACTION_CHUP,
    BTHF_CLIENT_CALL_ACTION_BTRH_0,
    BTHF_CLIENT_CALL_ACTION_BTRH_1,
    BTHF_CLIENT_CALL_ACTION_BTRH_2,
} bthf_client_call_action_t;

typedef enum {
    BTHF_CLIENT_SERVICE_UNKNOWN = 0,
    BTHF_CLIENT_SERVICE_VOICE,
    BTHF_CLIENT_SERVICE_FAX
} bthf_client_subscriber_service_type_t;

typedef enum {
    BTHF_CLIENT_IN_BAND_RINGTONE_NOT_PROVIDED = 0,
    BTHF_CLIENT_IN_BAND_RINGTONE_PROVIDED,
} bthf_client_in_band_ring_state_t;

/* Peer features masks */
#define BTHF_CLIENT_PEER_FEAT_3WAY   0x00000001  /* Three-way calling */
#define BTHF_CLIENT_PEER_FEAT_ECNR   0x00000002  /* Echo cancellation and/or noise reduction */
#define BTHF_CLIENT_PEER_FEAT_VREC   0x00000004  /* Voice recognition */
#define BTHF_CLIENT_PEER_FEAT_INBAND 0x00000008  /* In-band ring tone */
#define BTHF_CLIENT_PEER_FEAT_VTAG   0x00000010  /* Attach a phone number to a voice tag */
#define BTHF_CLIENT_PEER_FEAT_REJECT 0x00000020  /* Ability to reject incoming call */
#define BTHF_CLIENT_PEER_FEAT_ECS    0x00000040  /* Enhanced Call Status */
#define BTHF_CLIENT_PEER_FEAT_ECC    0x00000080  /* Enhanced Call Control */
#define BTHF_CLIENT_PEER_FEAT_EXTERR 0x00000100  /* Extended error codes */
#define BTHF_CLIENT_PEER_FEAT_CODEC  0x00000200  /* Codec Negotiation */

/* Peer call handling features masks */
#define BTHF_CLIENT_CHLD_FEAT_REL           0x00000001  /* 0  Release waiting call or held calls */
#define BTHF_CLIENT_CHLD_FEAT_REL_ACC       0x00000002  /* 1  Release active calls and accept other
                                                              (waiting or held) cal */
#define BTHF_CLIENT_CHLD_FEAT_REL_X         0x00000004  /* 1x Release specified active call only */
#define BTHF_CLIENT_CHLD_FEAT_HOLD_ACC      0x00000008  /* 2  Active calls on hold and accept other
                                                              (waiting or held) call */
#define BTHF_CLIENT_CHLD_FEAT_PRIV_X        0x00000010  /* 2x Request private mode with specified
                                                              call (put the rest on hold) */
#define BTHF_CLIENT_CHLD_FEAT_MERGE         0x00000020  /* 3  Add held call to multiparty */
#define BTHF_CLIENT_CHLD_FEAT_MERGE_DETACH  0x00000040  /* 4  Connect two calls and leave
                                                              (disconnect from) multiparty */

/** Callback for connection state change.
 *  state will have one of the values from BtHfConnectionState
 *  peer/chld_features are valid only for BTHF_CLIENT_CONNECTION_STATE_SLC_CONNECTED state
 */
typedef void (* bthf_client_connection_state_callback)(bthf_client_connection_state_t state,
                                                       unsigned int peer_feat,
                                                       unsigned int chld_feat,
                                                       bt_bdaddr_t *bd_addr);

/** Callback for audio connection state change.
 *  state will have one of the values from BtHfAudioState
 */
typedef void (* bthf_client_audio_state_callback)(bthf_client_audio_state_t state,
                                                  bt_bdaddr_t *bd_addr);

/** Callback for VR connection state change.
 *  state will have one of the values from BtHfVRState
 */
typedef void (* bthf_client_vr_cmd_callback)(bthf_client_vr_state_t state);

/** Callback for network state change
 */
typedef void (* bthf_client_network_state_callback) (bthf_client_network_state_t state);

/** Callback for network roaming status change
 */
typedef void (* bthf_client_network_roaming_callback) (bthf_client_service_type_t type);

/** Callback for signal strength indication
 */
typedef void (* bthf_client_network_signal_callback) (int signal_strength);

/** Callback for battery level indication
 */
typedef void (* bthf_client_battery_level_callback) (int battery_level);

/** Callback for current operator name
 */
typedef void (* bthf_client_current_operator_callback) (const char *name);

/** Callback for call indicator
 */
typedef void (* bthf_client_call_callback) (bthf_client_call_t call);

/** Callback for callsetup indicator
 */
typedef void (* bthf_client_callsetup_callback) (bthf_client_callsetup_t callsetup);

/** Callback for callheld indicator
 */
typedef void (* bthf_client_callheld_callback) (bthf_client_callheld_t callheld);

/** Callback for response and hold
 */
typedef void (* bthf_client_resp_and_hold_callback) (bthf_client_resp_and_hold_t resp_and_hold);

/** Callback for Calling Line Identification notification
 *  Will be called only when there is an incoming call and number is provided.
 */
typedef void (* bthf_client_clip_callback) (const char *number);

/**
 * Callback for Call Waiting notification
 */
typedef void (* bthf_client_call_waiting_callback) (const char *number);

/**
 *  Callback for listing current calls. Can be called multiple time.
 *  If number is unknown NULL is passed.
 */
typedef void (*bthf_client_current_calls) (int index, bthf_client_call_direction_t dir,
                                           bthf_client_call_state_t state,
                                           bthf_client_call_mpty_type_t mpty,
                                           const char *number);

/** Callback for audio volume change
 */
typedef void (*bthf_client_volume_change_callback) (bthf_client_volume_type_t type, int volume);

/** Callback for command complete event
 *  cme is valid only for BTHF_CLIENT_CMD_COMPLETE_ERROR_CME type
 */
typedef void (*bthf_client_cmd_complete_callback) (bthf_client_cmd_complete_t type, int cme);

/** Callback for subscriber information
 */
typedef void (* bthf_client_subscriber_info_callback) (const char *name,
                                                       bthf_client_subscriber_service_type_t type);

/** Callback for in-band ring tone settings
 */
typedef void (* bthf_client_in_band_ring_tone_callback) (bthf_client_in_band_ring_state_t state);

/**
 * Callback for requested number from AG
 */
typedef void (* bthf_client_last_voice_tag_number_callback) (const char *number);

/**
 * Callback for sending ring indication to app
 */
typedef void (* bthf_client_ring_indication_callback) (void);

/**
 * Callback for sending cgmi indication to app
 */
typedef void (* bthf_client_cgmi_indication_callback) (const char *str);

/**
 * Callback for sending cgmm indication to app
 */
typedef void (* bthf_client_cgmm_indication_callback) (const char *str);

/** BT-HF callback structure. */
typedef struct {
    /** set to sizeof(BtHfClientCallbacks) */
    size_t      size;
    bthf_client_connection_state_callback  connection_state_cb;
    bthf_client_audio_state_callback       audio_state_cb;
    bthf_client_vr_cmd_callback            vr_cmd_cb;
    bthf_client_network_state_callback     network_state_cb;
    bthf_client_network_roaming_callback   network_roaming_cb;
    bthf_client_network_signal_callback    network_signal_cb;
    bthf_client_battery_level_callback     battery_level_cb;
    bthf_client_current_operator_callback  current_operator_cb;
    bthf_client_call_callback              call_cb;
    bthf_client_callsetup_callback         callsetup_cb;
    bthf_client_callheld_callback          callheld_cb;
    bthf_client_resp_and_hold_callback     resp_and_hold_cb;
    bthf_client_clip_callback              clip_cb;
    bthf_client_call_waiting_callback      call_waiting_cb;
    bthf_client_current_calls              current_calls_cb;
    bthf_client_volume_change_callback     volume_change_cb;
    bthf_client_cmd_complete_callback      cmd_complete_cb;
    bthf_client_subscriber_info_callback   subscriber_info_cb;
    bthf_client_in_band_ring_tone_callback in_band_ring_tone_cb;
    bthf_client_last_voice_tag_number_callback last_voice_tag_number_callback;
    bthf_client_ring_indication_callback   ring_indication_cb;
    bthf_client_cgmi_indication_callback   cgmi_cb;
    bthf_client_cgmm_indication_callback   cgmm_cb;
} bthf_client_callbacks_t;

/** Represents the standard BT-HF interface. */
typedef struct {

    /** set to sizeof(BtHfClientInterface) */
    size_t size;
    /**
     * Register the BtHf callbacks
     */
    bt_status_t (*init)(bthf_client_callbacks_t* callbacks);

    /** connect to audio gateway */
    bt_status_t (*connect)(bt_bdaddr_t *bd_addr);

    /** disconnect from audio gateway */
    bt_status_t (*disconnect)(bt_bdaddr_t *bd_addr);

    /** create an audio connection */
    bt_status_t (*connect_audio)(bt_bdaddr_t *bd_addr);

    /** close the audio connection */
    bt_status_t (*disconnect_audio)(bt_bdaddr_t *bd_addr);

    /** start voice recognition */
    bt_status_t (*start_voice_recognition)(void);

    /** stop voice recognition */
    bt_status_t (*stop_voice_recognition)(void);

    /** volume control */
    bt_status_t (*volume_control) (bthf_client_volume_type_t type, int volume);

    /** place a call with number a number
     * if number is NULL last called number is called (aka re-dial)*/
    bt_status_t (*dial) (const char *number);

    /** place a call with number specified by location (speed dial) */
    bt_status_t (*dial_memory) (int location);

    /** perform specified call related action
     * idx is limited only for enhanced call control related action
     */
    bt_status_t (*handle_call_action) (bthf_client_call_action_t action, int idx);

    /** query list of current calls */
    bt_status_t (*query_current_calls) (void);

    /** query name of current selected operator */
    bt_status_t (*query_current_operator_name) (void);

    /** Retrieve subscriber information */
    bt_status_t (*retrieve_subscriber_info) (void);

    /** Send DTMF code*/
    bt_status_t (*send_dtmf) (char code);

    /** Request a phone number from AG corresponding to last voice tag recorded */
    bt_status_t (*request_last_voice_tag_number) (void);

    /** Closes the interface. */
    void (*cleanup)(void);

    /** Send AT Command. */
    bt_status_t (*send_at_cmd) (int cmd, int val1, int val2, const char *arg);
} bthf_client_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_HF_CLIENT_H */
