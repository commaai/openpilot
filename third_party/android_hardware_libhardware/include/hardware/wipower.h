/*
 * Copyright (c) 2013-2015, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of The Linux Foundation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ANDROID_INCLUDE_WIPOWER_H
#define ANDROID_INCLUDE_WIPOWER_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdbool.h>

#include <hardware/hardware.h>
#include <hardware/bluetooth.h>

__BEGIN_DECLS

typedef enum {
   OFF =0,
   ON
} wipower_state_t;


typedef struct {

unsigned char optional;
unsigned short rect_voltage;
unsigned short rect_current;
unsigned short out_voltage;
unsigned short out_current;
unsigned char temp;
unsigned short rect_voltage_min;
unsigned short rect_voltage_set;
unsigned short rect_voltage_max;
unsigned char alert;
unsigned short rfu1;
unsigned char rfu2;

}__attribute__((packed)) wipower_dyn_data_t;

/** Bluetooth Enable/Disable Callback. */
typedef void (*wipower_state_changed_callback)(wipower_state_t state);


typedef void (*wipower_alerts)(unsigned char alert);


typedef void (*wipower_dynamic_data)(wipower_dyn_data_t* alert_data);


typedef void (*wipower_power_apply)(unsigned char power_flag);

typedef void (*callback_thread_event)(bt_cb_thread_evt evt);

/** Bluetooth DM callback structure. */
typedef struct {
    /** set to sizeof(wipower_callbacks_t) */
    size_t size;
    wipower_state_changed_callback wipower_state_changed_cb;
    wipower_alerts wipower_alert;
    wipower_dynamic_data wipower_data;
    wipower_power_apply wipower_power_event;
    callback_thread_event callback_thread_event;
} wipower_callbacks_t;


/** Represents the standard Wipower interface. */
typedef struct {
    /** set to sizeof(wipower_interface_t) */
    size_t size;

    /** Initialize Wipower modules*/
    int (*init)(wipower_callbacks_t *wp_callbacks);

    /** Enable/Disable Wipower charging */
    int (*enable)(bool enable);

    int (*set_current_limit)(short value);

    unsigned char (*get_current_limit)(void);

    wipower_state_t (*get_state)(void);

    /** Enable/Disable Wipower charging */
    int (*enable_alerts)(bool enable);

    int (*enable_data_notify)(bool enable);
    int (*enable_power_apply)(bool enable, bool on, bool time_flag);
} wipower_interface_t;


__END_DECLS

#endif /* ANDROID_INCLUDE_WIPOWER_H */
