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

#ifndef ANDROID_INCLUDE_BT_HF_H
#define ANDROID_INCLUDE_BT_HF_H

__BEGIN_DECLS

/* AT response code - OK/Error */
typedef enum {
    BTHF_AT_RESPONSE_ERROR = 0,
    BTHF_AT_RESPONSE_OK
} bthf_at_response_t;

typedef enum {
    BTHF_CONNECTION_STATE_DISCONNECTED = 0,
    BTHF_CONNECTION_STATE_CONNECTING,
    BTHF_CONNECTION_STATE_CONNECTED,
    BTHF_CONNECTION_STATE_SLC_CONNECTED,
    BTHF_CONNECTION_STATE_DISCONNECTING
} bthf_connection_state_t;

typedef enum {
    BTHF_AUDIO_STATE_DISCONNECTED = 0,
    BTHF_AUDIO_STATE_CONNECTING,
    BTHF_AUDIO_STATE_CONNECTED,
    BTHF_AUDIO_STATE_DISCONNECTING
} bthf_audio_state_t;

typedef enum {
    BTHF_VR_STATE_STOPPED = 0,
    BTHF_VR_STATE_STARTED
} bthf_vr_state_t;

typedef enum {
    BTHF_VOLUME_TYPE_SPK = 0,
    BTHF_VOLUME_TYPE_MIC
} bthf_volume_type_t;

/* Noise Reduction and Echo Cancellation */
typedef enum
{
    BTHF_NREC_STOP,
    BTHF_NREC_START
} bthf_nrec_t;

/* WBS codec setting */
typedef enum
{
   BTHF_WBS_NONE,
   BTHF_WBS_NO,
   BTHF_WBS_YES
}bthf_wbs_config_t;

/* BIND type*/
typedef enum
{
   BTHF_BIND_SET,
   BTHF_BIND_READ,
   BTHF_BIND_TEST
}bthf_bind_type_t;


/* CHLD - Call held handling */
typedef enum
{
    BTHF_CHLD_TYPE_RELEASEHELD,              // Terminate all held or set UDUB("busy") to a waiting call
    BTHF_CHLD_TYPE_RELEASEACTIVE_ACCEPTHELD, // Terminate all active calls and accepts a waiting/held call
    BTHF_CHLD_TYPE_HOLDACTIVE_ACCEPTHELD,    // Hold all active calls and accepts a waiting/held call
    BTHF_CHLD_TYPE_ADDHELDTOCONF,            // Add all held calls to a conference
} bthf_chld_type_t;

/** Callback for connection state change.
 *  state will have one of the values from BtHfConnectionState
 */
typedef void (* bthf_connection_state_callback)(bthf_connection_state_t state, bt_bdaddr_t *bd_addr);

/** Callback for audio connection state change.
 *  state will have one of the values from BtHfAudioState
 */
typedef void (* bthf_audio_state_callback)(bthf_audio_state_t state, bt_bdaddr_t *bd_addr);

/** Callback for VR connection state change.
 *  state will have one of the values from BtHfVRState
 */
typedef void (* bthf_vr_cmd_callback)(bthf_vr_state_t state, bt_bdaddr_t *bd_addr);

/** Callback for answer incoming call (ATA)
 */
typedef void (* bthf_answer_call_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for disconnect call (AT+CHUP)
 */
typedef void (* bthf_hangup_call_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for disconnect call (AT+CHUP)
 *  type will denote Speaker/Mic gain (BtHfVolumeControl).
 */
typedef void (* bthf_volume_cmd_callback)(bthf_volume_type_t type, int volume, bt_bdaddr_t *bd_addr);

/** Callback for dialing an outgoing call
 *  If number is NULL, redial
 */
typedef void (* bthf_dial_call_cmd_callback)(char *number, bt_bdaddr_t *bd_addr);

/** Callback for sending DTMF tones
 *  tone contains the dtmf character to be sent
 */
typedef void (* bthf_dtmf_cmd_callback)(char tone, bt_bdaddr_t *bd_addr);

/** Callback for enabling/disabling noise reduction/echo cancellation
 *  value will be 1 to enable, 0 to disable
 */
typedef void (* bthf_nrec_cmd_callback)(bthf_nrec_t nrec, bt_bdaddr_t *bd_addr);

/** Callback for AT+BCS and event from BAC
 *  WBS enable, WBS disable
 */
typedef void (* bthf_wbs_callback)(bthf_wbs_config_t wbs, bt_bdaddr_t *bd_addr);

/** Callback for call hold handling (AT+CHLD)
 *  value will contain the call hold command (0, 1, 2, 3)
 */
typedef void (* bthf_chld_cmd_callback)(bthf_chld_type_t chld, bt_bdaddr_t *bd_addr);

/** Callback for CNUM (subscriber number)
 */
typedef void (* bthf_cnum_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for indicators (CIND)
 */
typedef void (* bthf_cind_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for operator selection (COPS)
 */
typedef void (* bthf_cops_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for call list (AT+CLCC)
 */
typedef void (* bthf_clcc_cmd_callback) (bt_bdaddr_t *bd_addr);

/** Callback for unknown AT command recd from HF
 *  at_string will contain the unparsed AT string
 */
typedef void (* bthf_unknown_at_cmd_callback)(char *at_string, bt_bdaddr_t *bd_addr);

/** Callback for keypressed (HSP) event.
 */
typedef void (* bthf_key_pressed_cmd_callback)(bt_bdaddr_t *bd_addr);

/** Callback for HF indicators (BIND)
 */
typedef void (* bthf_bind_cmd_callback)(char* hf_ind, bthf_bind_type_t type, bt_bdaddr_t *bd_addr);

/** Callback for HF indicator value (BIEV)
 */
typedef void (* bthf_biev_cmd_callback)(char* hf_ind_val, bt_bdaddr_t *bd_addr);


/** BT-HF callback structure. */
typedef struct {
    /** set to sizeof(BtHfCallbacks) */
    size_t      size;
    bthf_connection_state_callback  connection_state_cb;
    bthf_audio_state_callback       audio_state_cb;
    bthf_vr_cmd_callback            vr_cmd_cb;
    bthf_answer_call_cmd_callback   answer_call_cmd_cb;
    bthf_hangup_call_cmd_callback   hangup_call_cmd_cb;
    bthf_volume_cmd_callback        volume_cmd_cb;
    bthf_dial_call_cmd_callback     dial_call_cmd_cb;
    bthf_dtmf_cmd_callback          dtmf_cmd_cb;
    bthf_nrec_cmd_callback          nrec_cmd_cb;
    bthf_wbs_callback               wbs_cb;
    bthf_chld_cmd_callback          chld_cmd_cb;
    bthf_cnum_cmd_callback          cnum_cmd_cb;
    bthf_cind_cmd_callback          cind_cmd_cb;
    bthf_cops_cmd_callback          cops_cmd_cb;
    bthf_clcc_cmd_callback          clcc_cmd_cb;
    bthf_unknown_at_cmd_callback    unknown_at_cmd_cb;
    bthf_key_pressed_cmd_callback   key_pressed_cmd_cb;
    bthf_bind_cmd_callback          bind_cmd_cb;
    bthf_biev_cmd_callback          biev_cmd_cb;
} bthf_callbacks_t;

/** Network Status */
typedef enum
{
    BTHF_NETWORK_STATE_NOT_AVAILABLE = 0,
    BTHF_NETWORK_STATE_AVAILABLE
} bthf_network_state_t;

/** Service type */
typedef enum
{
    BTHF_SERVICE_TYPE_HOME = 0,
    BTHF_SERVICE_TYPE_ROAMING
} bthf_service_type_t;

typedef enum {
    BTHF_CALL_STATE_ACTIVE = 0,
    BTHF_CALL_STATE_HELD,
    BTHF_CALL_STATE_DIALING,
    BTHF_CALL_STATE_ALERTING,
    BTHF_CALL_STATE_INCOMING,
    BTHF_CALL_STATE_WAITING,
    BTHF_CALL_STATE_IDLE
} bthf_call_state_t;

typedef enum {
    BTHF_CALL_DIRECTION_OUTGOING = 0,
    BTHF_CALL_DIRECTION_INCOMING
} bthf_call_direction_t;

typedef enum {
    BTHF_CALL_TYPE_VOICE = 0,
    BTHF_CALL_TYPE_DATA,
    BTHF_CALL_TYPE_FAX
} bthf_call_mode_t;

typedef enum {
    BTHF_CALL_MPTY_TYPE_SINGLE = 0,
    BTHF_CALL_MPTY_TYPE_MULTI
} bthf_call_mpty_type_t;

typedef enum {
    BTHF_HF_INDICATOR_STATE_DISABLED = 0,
    BTHF_HF_INDICATOR_STATE_ENABLED
} bthf_hf_indicator_status_t;

typedef enum {
    BTHF_CALL_ADDRTYPE_UNKNOWN = 0x81,
    BTHF_CALL_ADDRTYPE_INTERNATIONAL = 0x91
} bthf_call_addrtype_t;

typedef enum {
    BTHF_VOIP_CALL_NETWORK_TYPE_MOBILE = 0,
    BTHF_VOIP_CALL_NETWORK_TYPE_WIFI
} bthf_voip_call_network_type_t;

typedef enum {
    BTHF_VOIP_STATE_STOPPED = 0,
    BTHF_VOIP_STATE_STARTED
} bthf_voip_state_t;

/** Represents the standard BT-HF interface. */
typedef struct {

    /** set to sizeof(BtHfInterface) */
    size_t          size;
    /**
     * Register the BtHf callbacks
     */
    bt_status_t (*init)( bthf_callbacks_t* callbacks, int max_hf_clients);

    /** connect to headset */
    bt_status_t (*connect)( bt_bdaddr_t *bd_addr );

    /** dis-connect from headset */
    bt_status_t (*disconnect)( bt_bdaddr_t *bd_addr );

    /** create an audio connection */
    bt_status_t (*connect_audio)( bt_bdaddr_t *bd_addr );

    /** close the audio connection */
    bt_status_t (*disconnect_audio)( bt_bdaddr_t *bd_addr );

    /** start voice recognition */
    bt_status_t (*start_voice_recognition)( bt_bdaddr_t *bd_addr );

    /** stop voice recognition */
    bt_status_t (*stop_voice_recognition)( bt_bdaddr_t *bd_addr );

    /** volume control */
    bt_status_t (*volume_control) (bthf_volume_type_t type, int volume, bt_bdaddr_t *bd_addr );

    /** Combined device status change notification */
    bt_status_t (*device_status_notification)(bthf_network_state_t ntk_state, bthf_service_type_t svc_type, int signal,
                           int batt_chg);

    /** Response for COPS command */
    bt_status_t (*cops_response)(const char *cops, bt_bdaddr_t *bd_addr );

    /** Response for CIND command */
    bt_status_t (*cind_response)(int svc, int num_active, int num_held, bthf_call_state_t call_setup_state,
                                 int signal, int roam, int batt_chg, bt_bdaddr_t *bd_addr );

    /** Pre-formatted AT response, typically in response to unknown AT cmd */
    bt_status_t (*formatted_at_response)(const char *rsp, bt_bdaddr_t *bd_addr );

    /** ok/error response
     *  ERROR (0)
     *  OK    (1)
     */
    bt_status_t (*at_response) (bthf_at_response_t response_code, int error_code, bt_bdaddr_t *bd_addr );

    /** response for CLCC command 
     *  Can be iteratively called for each call index
     *  Call index of 0 will be treated as NULL termination (Completes response)
     */
    bt_status_t (*clcc_response) (int index, bthf_call_direction_t dir,
                                bthf_call_state_t state, bthf_call_mode_t mode,
                                bthf_call_mpty_type_t mpty, const char *number,
                                bthf_call_addrtype_t type, bt_bdaddr_t *bd_addr );

    /** notify of a call state change
     *  Each update notifies 
     *    1. Number of active/held/ringing calls
     *    2. call_state: This denotes the state change that triggered this msg
     *                   This will take one of the values from BtHfCallState
     *    3. number & type: valid only for incoming & waiting call
    */
    bt_status_t (*phone_state_change) (int num_active, int num_held, bthf_call_state_t call_setup_state,
                                       const char *number, bthf_call_addrtype_t type);

    /** Closes the interface. */
    void  (*cleanup)( void );

    /** configureation for the SCO codec */
    bt_status_t (*configure_wbs)( bt_bdaddr_t *bd_addr ,bthf_wbs_config_t config );

    /** Response for BIND READ command and activation/deactivation of  HF indicator */
    bt_status_t (*bind_response) (int anum, bthf_hf_indicator_status_t status,
                                  bt_bdaddr_t *bd_addr);

    /** Response for BIND TEST command */
    bt_status_t (*bind_string_response) (const char* result, bt_bdaddr_t *bd_addr);

    /** Sends connectivity network type used by Voip currently to stack */
    bt_status_t (*voip_network_type_wifi) (bthf_voip_state_t is_voip_started,
                                           bthf_voip_call_network_type_t is_network_wifi);
} bthf_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_HF_H */
