/*
 * Copyright (C) 2008 The Android Open Source Project
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

#ifndef ANDROID_LIGHTS_INTERFACE_H
#define ANDROID_LIGHTS_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

/**
 * The id of this module
 */
#define LIGHTS_HARDWARE_MODULE_ID "lights"

/*
 * These light IDs correspond to logical lights, not physical.
 * So for example, if your INDICATOR light is in line with your
 * BUTTONS, it might make sense to also light the INDICATOR
 * light to a reasonable color when the BUTTONS are lit.
 */
#define LIGHT_ID_BACKLIGHT          "backlight"
#define LIGHT_ID_KEYBOARD           "keyboard"
#define LIGHT_ID_BUTTONS            "buttons"
#define LIGHT_ID_BATTERY            "battery"
#define LIGHT_ID_NOTIFICATIONS      "notifications"
#define LIGHT_ID_ATTENTION          "attention"

/*
 * These lights aren't currently supported by the higher
 * layers, but could be someday, so we have the constants
 * here now.
 */
#define LIGHT_ID_BLUETOOTH          "bluetooth"
#define LIGHT_ID_WIFI               "wifi"

/*
 * Additional hardware-specific lights
 */
#define LIGHT_ID_CAPS               "caps"
#define LIGHT_ID_FUNC               "func"

/* ************************************************************************
 * Flash modes for the flashMode field of light_state_t.
 */

#define LIGHT_FLASH_NONE            0

/**
 * To flash the light at a given rate, set flashMode to LIGHT_FLASH_TIMED,
 * and then flashOnMS should be set to the number of milliseconds to turn
 * the light on, followed by the number of milliseconds to turn the light
 * off.
 */
#define LIGHT_FLASH_TIMED           1

/**
 * To flash the light using hardware assist, set flashMode to
 * the hardware mode.
 */
#define LIGHT_FLASH_HARDWARE        2

/**
 * Light brightness is managed by a user setting.
 */
#define BRIGHTNESS_MODE_USER        0

/**
 * Light brightness is managed by a light sensor.
 */
#define BRIGHTNESS_MODE_SENSOR      1

/**
 * Light mode allows multiple LEDs
 */
#define LIGHT_MODE_MULTIPLE_LEDS    0x01

/**
 * The parameters that can be set for a given light.
 *
 * Not all lights must support all parameters.  If you
 * can do something backward-compatible, you should.
 */
struct light_state_t {
    /**
     * The color of the LED in ARGB.
     *
     * Do your best here.
     *   - If your light can only do red or green, if they ask for blue,
     *     you should do green.
     *   - If you can only do a brightness ramp, then use this formula:
     *      unsigned char brightness = ((77*((color>>16)&0x00ff))
     *              + (150*((color>>8)&0x00ff)) + (29*(color&0x00ff))) >> 8;
     *   - If you can only do on or off, 0 is off, anything else is on.
     *
     * The high byte should be ignored.  Callers will set it to 0xff (which
     * would correspond to 255 alpha).
     *
     * CyanogenMod: The high byte value can be implemented to control the LEDs
     * Brightness from the Lights settings. The value goes from 0x01 to 0xFF.
     */
    unsigned int color;

    /**
     * See the LIGHT_FLASH_* constants
     */
    int flashMode;
    int flashOnMS;
    int flashOffMS;

    /**
     * Policy used by the framework to manage the light's brightness.
     * Currently the values are BRIGHTNESS_MODE_USER and BRIGHTNESS_MODE_SENSOR.
     */
    int brightnessMode;

    /**
     * Define the LEDs modes (multiple, ...).
     * See the LIGHTS_MODE_* mask constants.
     */
    unsigned int ledsModes;
};

struct light_device_t {
    struct hw_device_t common;

    /**
     * Set the provided lights to the provided values.
     *
     * Returns: 0 on succes, error code on failure.
     */
    int (*set_light)(struct light_device_t* dev,
            struct light_state_t const* state);
};


__END_DECLS

#endif  // ANDROID_LIGHTS_INTERFACE_H

