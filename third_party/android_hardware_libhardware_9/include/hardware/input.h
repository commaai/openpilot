/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_INPUT_H
#define ANDROID_INCLUDE_HARDWARE_INPUT_H

#include <hardware/hardware.h>
#include <stdint.h>

__BEGIN_DECLS

#define INPUT_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define INPUT_HARDWARE_MODULE_ID "input"

#define INPUT_INSTANCE_EVDEV "evdev"

typedef enum input_bus {
    INPUT_BUS_BT,
    INPUT_BUS_USB,
    INPUT_BUS_SERIAL,
    INPUT_BUS_BUILTIN
} input_bus_t;

typedef struct input_host input_host_t;

typedef struct input_device_handle input_device_handle_t;

typedef struct input_device_identifier input_device_identifier_t;

typedef struct input_device_definition input_device_definition_t;

typedef struct input_report_definition input_report_definition_t;

typedef struct input_report input_report_t;

typedef struct input_collection input_collection_t;

typedef struct input_property_map input_property_map_t;

typedef struct input_property input_property_t;

typedef enum {
    // keycodes
    INPUT_USAGE_KEYCODE_UNKNOWN,
    INPUT_USAGE_KEYCODE_SOFT_LEFT,
    INPUT_USAGE_KEYCODE_SOFT_RIGHT,
    INPUT_USAGE_KEYCODE_HOME,
    INPUT_USAGE_KEYCODE_BACK,
    INPUT_USAGE_KEYCODE_CALL,
    INPUT_USAGE_KEYCODE_ENDCALL,
    INPUT_USAGE_KEYCODE_0,
    INPUT_USAGE_KEYCODE_1,
    INPUT_USAGE_KEYCODE_2,
    INPUT_USAGE_KEYCODE_3,
    INPUT_USAGE_KEYCODE_4,
    INPUT_USAGE_KEYCODE_5,
    INPUT_USAGE_KEYCODE_6,
    INPUT_USAGE_KEYCODE_7,
    INPUT_USAGE_KEYCODE_8,
    INPUT_USAGE_KEYCODE_9,
    INPUT_USAGE_KEYCODE_STAR,
    INPUT_USAGE_KEYCODE_POUND,
    INPUT_USAGE_KEYCODE_DPAD_UP,
    INPUT_USAGE_KEYCODE_DPAD_DOWN,
    INPUT_USAGE_KEYCODE_DPAD_LEFT,
    INPUT_USAGE_KEYCODE_DPAD_RIGHT,
    INPUT_USAGE_KEYCODE_DPAD_CENTER,
    INPUT_USAGE_KEYCODE_VOLUME_UP,
    INPUT_USAGE_KEYCODE_VOLUME_DOWN,
    INPUT_USAGE_KEYCODE_POWER,
    INPUT_USAGE_KEYCODE_CAMERA,
    INPUT_USAGE_KEYCODE_CLEAR,
    INPUT_USAGE_KEYCODE_A,
    INPUT_USAGE_KEYCODE_B,
    INPUT_USAGE_KEYCODE_C,
    INPUT_USAGE_KEYCODE_D,
    INPUT_USAGE_KEYCODE_E,
    INPUT_USAGE_KEYCODE_F,
    INPUT_USAGE_KEYCODE_G,
    INPUT_USAGE_KEYCODE_H,
    INPUT_USAGE_KEYCODE_I,
    INPUT_USAGE_KEYCODE_J,
    INPUT_USAGE_KEYCODE_K,
    INPUT_USAGE_KEYCODE_L,
    INPUT_USAGE_KEYCODE_M,
    INPUT_USAGE_KEYCODE_N,
    INPUT_USAGE_KEYCODE_O,
    INPUT_USAGE_KEYCODE_P,
    INPUT_USAGE_KEYCODE_Q,
    INPUT_USAGE_KEYCODE_R,
    INPUT_USAGE_KEYCODE_S,
    INPUT_USAGE_KEYCODE_T,
    INPUT_USAGE_KEYCODE_U,
    INPUT_USAGE_KEYCODE_V,
    INPUT_USAGE_KEYCODE_W,
    INPUT_USAGE_KEYCODE_X,
    INPUT_USAGE_KEYCODE_Y,
    INPUT_USAGE_KEYCODE_Z,
    INPUT_USAGE_KEYCODE_COMMA,
    INPUT_USAGE_KEYCODE_PERIOD,
    INPUT_USAGE_KEYCODE_ALT_LEFT,
    INPUT_USAGE_KEYCODE_ALT_RIGHT,
    INPUT_USAGE_KEYCODE_SHIFT_LEFT,
    INPUT_USAGE_KEYCODE_SHIFT_RIGHT,
    INPUT_USAGE_KEYCODE_TAB,
    INPUT_USAGE_KEYCODE_SPACE,
    INPUT_USAGE_KEYCODE_SYM,
    INPUT_USAGE_KEYCODE_EXPLORER,
    INPUT_USAGE_KEYCODE_ENVELOPE,
    INPUT_USAGE_KEYCODE_ENTER,
    INPUT_USAGE_KEYCODE_DEL,
    INPUT_USAGE_KEYCODE_GRAVE,
    INPUT_USAGE_KEYCODE_MINUS,
    INPUT_USAGE_KEYCODE_EQUALS,
    INPUT_USAGE_KEYCODE_LEFT_BRACKET,
    INPUT_USAGE_KEYCODE_RIGHT_BRACKET,
    INPUT_USAGE_KEYCODE_BACKSLASH,
    INPUT_USAGE_KEYCODE_SEMICOLON,
    INPUT_USAGE_KEYCODE_APOSTROPHE,
    INPUT_USAGE_KEYCODE_SLASH,
    INPUT_USAGE_KEYCODE_AT,
    INPUT_USAGE_KEYCODE_NUM,
    INPUT_USAGE_KEYCODE_HEADSETHOOK,
    INPUT_USAGE_KEYCODE_FOCUS,   // *Camera* focus
    INPUT_USAGE_KEYCODE_PLUS,
    INPUT_USAGE_KEYCODE_MENU,
    INPUT_USAGE_KEYCODE_NOTIFICATION,
    INPUT_USAGE_KEYCODE_SEARCH,
    INPUT_USAGE_KEYCODE_MEDIA_PLAY_PAUSE,
    INPUT_USAGE_KEYCODE_MEDIA_STOP,
    INPUT_USAGE_KEYCODE_MEDIA_NEXT,
    INPUT_USAGE_KEYCODE_MEDIA_PREVIOUS,
    INPUT_USAGE_KEYCODE_MEDIA_REWIND,
    INPUT_USAGE_KEYCODE_MEDIA_FAST_FORWARD,
    INPUT_USAGE_KEYCODE_MUTE,
    INPUT_USAGE_KEYCODE_PAGE_UP,
    INPUT_USAGE_KEYCODE_PAGE_DOWN,
    INPUT_USAGE_KEYCODE_PICTSYMBOLS,
    INPUT_USAGE_KEYCODE_SWITCH_CHARSET,
    INPUT_USAGE_KEYCODE_BUTTON_A,
    INPUT_USAGE_KEYCODE_BUTTON_B,
    INPUT_USAGE_KEYCODE_BUTTON_C,
    INPUT_USAGE_KEYCODE_BUTTON_X,
    INPUT_USAGE_KEYCODE_BUTTON_Y,
    INPUT_USAGE_KEYCODE_BUTTON_Z,
    INPUT_USAGE_KEYCODE_BUTTON_L1,
    INPUT_USAGE_KEYCODE_BUTTON_R1,
    INPUT_USAGE_KEYCODE_BUTTON_L2,
    INPUT_USAGE_KEYCODE_BUTTON_R2,
    INPUT_USAGE_KEYCODE_BUTTON_THUMBL,
    INPUT_USAGE_KEYCODE_BUTTON_THUMBR,
    INPUT_USAGE_KEYCODE_BUTTON_START,
    INPUT_USAGE_KEYCODE_BUTTON_SELECT,
    INPUT_USAGE_KEYCODE_BUTTON_MODE,
    INPUT_USAGE_KEYCODE_ESCAPE,
    INPUT_USAGE_KEYCODE_FORWARD_DEL,
    INPUT_USAGE_KEYCODE_CTRL_LEFT,
    INPUT_USAGE_KEYCODE_CTRL_RIGHT,
    INPUT_USAGE_KEYCODE_CAPS_LOCK,
    INPUT_USAGE_KEYCODE_SCROLL_LOCK,
    INPUT_USAGE_KEYCODE_META_LEFT,
    INPUT_USAGE_KEYCODE_META_RIGHT,
    INPUT_USAGE_KEYCODE_FUNCTION,
    INPUT_USAGE_KEYCODE_SYSRQ,
    INPUT_USAGE_KEYCODE_BREAK,
    INPUT_USAGE_KEYCODE_MOVE_HOME,
    INPUT_USAGE_KEYCODE_MOVE_END,
    INPUT_USAGE_KEYCODE_INSERT,
    INPUT_USAGE_KEYCODE_FORWARD,
    INPUT_USAGE_KEYCODE_MEDIA_PLAY,
    INPUT_USAGE_KEYCODE_MEDIA_PAUSE,
    INPUT_USAGE_KEYCODE_MEDIA_CLOSE,
    INPUT_USAGE_KEYCODE_MEDIA_EJECT,
    INPUT_USAGE_KEYCODE_MEDIA_RECORD,
    INPUT_USAGE_KEYCODE_F1,
    INPUT_USAGE_KEYCODE_F2,
    INPUT_USAGE_KEYCODE_F3,
    INPUT_USAGE_KEYCODE_F4,
    INPUT_USAGE_KEYCODE_F5,
    INPUT_USAGE_KEYCODE_F6,
    INPUT_USAGE_KEYCODE_F7,
    INPUT_USAGE_KEYCODE_F8,
    INPUT_USAGE_KEYCODE_F9,
    INPUT_USAGE_KEYCODE_F10,
    INPUT_USAGE_KEYCODE_F11,
    INPUT_USAGE_KEYCODE_F12,
    INPUT_USAGE_KEYCODE_NUM_LOCK,
    INPUT_USAGE_KEYCODE_NUMPAD_0,
    INPUT_USAGE_KEYCODE_NUMPAD_1,
    INPUT_USAGE_KEYCODE_NUMPAD_2,
    INPUT_USAGE_KEYCODE_NUMPAD_3,
    INPUT_USAGE_KEYCODE_NUMPAD_4,
    INPUT_USAGE_KEYCODE_NUMPAD_5,
    INPUT_USAGE_KEYCODE_NUMPAD_6,
    INPUT_USAGE_KEYCODE_NUMPAD_7,
    INPUT_USAGE_KEYCODE_NUMPAD_8,
    INPUT_USAGE_KEYCODE_NUMPAD_9,
    INPUT_USAGE_KEYCODE_NUMPAD_DIVIDE,
    INPUT_USAGE_KEYCODE_NUMPAD_MULTIPLY,
    INPUT_USAGE_KEYCODE_NUMPAD_SUBTRACT,
    INPUT_USAGE_KEYCODE_NUMPAD_ADD,
    INPUT_USAGE_KEYCODE_NUMPAD_DOT,
    INPUT_USAGE_KEYCODE_NUMPAD_COMMA,
    INPUT_USAGE_KEYCODE_NUMPAD_ENTER,
    INPUT_USAGE_KEYCODE_NUMPAD_EQUALS,
    INPUT_USAGE_KEYCODE_NUMPAD_LEFT_PAREN,
    INPUT_USAGE_KEYCODE_NUMPAD_RIGHT_PAREN,
    INPUT_USAGE_KEYCODE_VOLUME_MUTE,
    INPUT_USAGE_KEYCODE_INFO,
    INPUT_USAGE_KEYCODE_CHANNEL_UP,
    INPUT_USAGE_KEYCODE_CHANNEL_DOWN,
    INPUT_USAGE_KEYCODE_ZOOM_IN,
    INPUT_USAGE_KEYCODE_ZOOM_OUT,
    INPUT_USAGE_KEYCODE_TV,
    INPUT_USAGE_KEYCODE_WINDOW,
    INPUT_USAGE_KEYCODE_GUIDE,
    INPUT_USAGE_KEYCODE_DVR,
    INPUT_USAGE_KEYCODE_BOOKMARK,
    INPUT_USAGE_KEYCODE_CAPTIONS,
    INPUT_USAGE_KEYCODE_SETTINGS,
    INPUT_USAGE_KEYCODE_TV_POWER,
    INPUT_USAGE_KEYCODE_TV_INPUT,
    INPUT_USAGE_KEYCODE_STB_POWER,
    INPUT_USAGE_KEYCODE_STB_INPUT,
    INPUT_USAGE_KEYCODE_AVR_POWER,
    INPUT_USAGE_KEYCODE_AVR_INPUT,
    INPUT_USAGE_KEYCODE_PROG_RED,
    INPUT_USAGE_KEYCODE_PROG_GREEN,
    INPUT_USAGE_KEYCODE_PROG_YELLOW,
    INPUT_USAGE_KEYCODE_PROG_BLUE,
    INPUT_USAGE_KEYCODE_APP_SWITCH,
    INPUT_USAGE_KEYCODE_BUTTON_1,
    INPUT_USAGE_KEYCODE_BUTTON_2,
    INPUT_USAGE_KEYCODE_BUTTON_3,
    INPUT_USAGE_KEYCODE_BUTTON_4,
    INPUT_USAGE_KEYCODE_BUTTON_5,
    INPUT_USAGE_KEYCODE_BUTTON_6,
    INPUT_USAGE_KEYCODE_BUTTON_7,
    INPUT_USAGE_KEYCODE_BUTTON_8,
    INPUT_USAGE_KEYCODE_BUTTON_9,
    INPUT_USAGE_KEYCODE_BUTTON_10,
    INPUT_USAGE_KEYCODE_BUTTON_11,
    INPUT_USAGE_KEYCODE_BUTTON_12,
    INPUT_USAGE_KEYCODE_BUTTON_13,
    INPUT_USAGE_KEYCODE_BUTTON_14,
    INPUT_USAGE_KEYCODE_BUTTON_15,
    INPUT_USAGE_KEYCODE_BUTTON_16,
    INPUT_USAGE_KEYCODE_LANGUAGE_SWITCH,
    INPUT_USAGE_KEYCODE_MANNER_MODE,
    INPUT_USAGE_KEYCODE_3D_MODE,
    INPUT_USAGE_KEYCODE_CONTACTS,
    INPUT_USAGE_KEYCODE_CALENDAR,
    INPUT_USAGE_KEYCODE_MUSIC,
    INPUT_USAGE_KEYCODE_CALCULATOR,
    INPUT_USAGE_KEYCODE_ZENKAKU_HANKAKU,
    INPUT_USAGE_KEYCODE_EISU,
    INPUT_USAGE_KEYCODE_MUHENKAN,
    INPUT_USAGE_KEYCODE_HENKAN,
    INPUT_USAGE_KEYCODE_KATAKANA_HIRAGANA,
    INPUT_USAGE_KEYCODE_YEN,
    INPUT_USAGE_KEYCODE_RO,
    INPUT_USAGE_KEYCODE_KANA,
    INPUT_USAGE_KEYCODE_ASSIST,
    INPUT_USAGE_KEYCODE_BRIGHTNESS_DOWN,
    INPUT_USAGE_KEYCODE_BRIGHTNESS_UP,
    INPUT_USAGE_KEYCODE_MEDIA_AUDIO_TRACK,
    INPUT_USAGE_KEYCODE_SLEEP,
    INPUT_USAGE_KEYCODE_WAKEUP,
    INPUT_USAGE_KEYCODE_PAIRING,
    INPUT_USAGE_KEYCODE_MEDIA_TOP_MENU,
    INPUT_USAGE_KEYCODE_11,
    INPUT_USAGE_KEYCODE_12,
    INPUT_USAGE_KEYCODE_LAST_CHANNEL,
    INPUT_USAGE_KEYCODE_TV_DATA_SERVICE,
    INPUT_USAGE_KEYCODE_VOICE_ASSIST,
    INPUT_USAGE_KEYCODE_TV_RADIO_SERVICE,
    INPUT_USAGE_KEYCODE_TV_TELETEXT,
    INPUT_USAGE_KEYCODE_TV_NUMBER_ENTRY,
    INPUT_USAGE_KEYCODE_TV_TERRESTRIAL_ANALOG,
    INPUT_USAGE_KEYCODE_TV_TERRESTRIAL_DIGITAL,
    INPUT_USAGE_KEYCODE_TV_SATELLITE,
    INPUT_USAGE_KEYCODE_TV_SATELLITE_BS,
    INPUT_USAGE_KEYCODE_TV_SATELLITE_CS,
    INPUT_USAGE_KEYCODE_TV_SATELLITE_SERVICE,
    INPUT_USAGE_KEYCODE_TV_NETWORK,
    INPUT_USAGE_KEYCODE_TV_ANTENNA_CABLE,
    INPUT_USAGE_KEYCODE_TV_INPUT_HDMI_1,
    INPUT_USAGE_KEYCODE_TV_INPUT_HDMI_2,
    INPUT_USAGE_KEYCODE_TV_INPUT_HDMI_3,
    INPUT_USAGE_KEYCODE_TV_INPUT_HDMI_4,
    INPUT_USAGE_KEYCODE_TV_INPUT_COMPOSITE_1,
    INPUT_USAGE_KEYCODE_TV_INPUT_COMPOSITE_2,
    INPUT_USAGE_KEYCODE_TV_INPUT_COMPONENT_1,
    INPUT_USAGE_KEYCODE_TV_INPUT_COMPONENT_2,
    INPUT_USAGE_KEYCODE_TV_INPUT_VGA_1,
    INPUT_USAGE_KEYCODE_TV_AUDIO_DESCRIPTION,
    INPUT_USAGE_KEYCODE_TV_AUDIO_DESCRIPTION_MIX_UP,
    INPUT_USAGE_KEYCODE_TV_AUDIO_DESCRIPTION_MIX_DOWN,
    INPUT_USAGE_KEYCODE_TV_ZOOM_MODE,
    INPUT_USAGE_KEYCODE_TV_CONTENTS_MENU,
    INPUT_USAGE_KEYCODE_TV_MEDIA_CONTEXT_MENU,
    INPUT_USAGE_KEYCODE_TV_TIMER_PROGRAMMING,
    INPUT_USAGE_KEYCODE_HELP,

    // axes
    INPUT_USAGE_AXIS_X,
    INPUT_USAGE_AXIS_Y,
    INPUT_USAGE_AXIS_Z,
    INPUT_USAGE_AXIS_RX,
    INPUT_USAGE_AXIS_RY,
    INPUT_USAGE_AXIS_RZ,
    INPUT_USAGE_AXIS_HAT_X,
    INPUT_USAGE_AXIS_HAT_Y,
    INPUT_USAGE_AXIS_PRESSURE,
    INPUT_USAGE_AXIS_SIZE,
    INPUT_USAGE_AXIS_TOUCH_MAJOR,
    INPUT_USAGE_AXIS_TOUCH_MINOR,
    INPUT_USAGE_AXIS_TOOL_MAJOR,
    INPUT_USAGE_AXIS_TOOL_MINOR,
    INPUT_USAGE_AXIS_ORIENTATION,
    INPUT_USAGE_AXIS_VSCROLL,
    INPUT_USAGE_AXIS_HSCROLL,
    INPUT_USAGE_AXIS_LTRIGGER,
    INPUT_USAGE_AXIS_RTRIGGER,
    INPUT_USAGE_AXIS_THROTTLE,
    INPUT_USAGE_AXIS_RUDDER,
    INPUT_USAGE_AXIS_WHEEL,
    INPUT_USAGE_AXIS_GAS,
    INPUT_USAGE_AXIS_BRAKE,
    INPUT_USAGE_AXIS_DISTANCE,
    INPUT_USAGE_AXIS_TILT,
    INPUT_USAGE_AXIS_GENERIC_1,
    INPUT_USAGE_AXIS_GENERIC_2,
    INPUT_USAGE_AXIS_GENERIC_3,
    INPUT_USAGE_AXIS_GENERIC_4,
    INPUT_USAGE_AXIS_GENERIC_5,
    INPUT_USAGE_AXIS_GENERIC_6,
    INPUT_USAGE_AXIS_GENERIC_7,
    INPUT_USAGE_AXIS_GENERIC_8,
    INPUT_USAGE_AXIS_GENERIC_9,
    INPUT_USAGE_AXIS_GENERIC_10,
    INPUT_USAGE_AXIS_GENERIC_11,
    INPUT_USAGE_AXIS_GENERIC_12,
    INPUT_USAGE_AXIS_GENERIC_13,
    INPUT_USAGE_AXIS_GENERIC_14,
    INPUT_USAGE_AXIS_GENERIC_15,
    INPUT_USAGE_AXIS_GENERIC_16,

    // leds
    INPUT_USAGE_LED_NUM_LOCK,
    INPUT_USAGE_LED_CAPS_LOCK,
    INPUT_USAGE_LED_SCROLL_LOCK,
    INPUT_USAGE_LED_COMPOSE,
    INPUT_USAGE_LED_KANA,
    INPUT_USAGE_LED_SLEEP,
    INPUT_USAGE_LED_SUSPEND,
    INPUT_USAGE_LED_MUTE,
    INPUT_USAGE_LED_MISC,
    INPUT_USAGE_LED_MAIL,
    INPUT_USAGE_LED_CHARGING,
    INPUT_USAGE_LED_CONTROLLER_1,
    INPUT_USAGE_LED_CONTROLLER_2,
    INPUT_USAGE_LED_CONTROLLER_3,
    INPUT_USAGE_LED_CONTROLLER_4,

    // switches
    INPUT_USAGE_SWITCH_UNKNOWN,
    INPUT_USAGE_SWITCH_LID,
    INPUT_USAGE_SWITCH_KEYPAD_SLIDE,
    INPUT_USAGE_SWITCH_HEADPHONE_INSERT,
    INPUT_USAGE_SWITCH_MICROPHONE_INSERT,
    INPUT_USAGE_SWITCH_LINEOUT_INSERT,
    INPUT_USAGE_SWITCH_CAMERA_LENS_COVER,

    // mouse buttons
    // (see android.view.MotionEvent)
    INPUT_USAGE_BUTTON_UNKNOWN,
    INPUT_USAGE_BUTTON_PRIMARY,   // left
    INPUT_USAGE_BUTTON_SECONDARY, // right
    INPUT_USAGE_BUTTON_TERTIARY,  // middle
    INPUT_USAGE_BUTTON_FORWARD,
    INPUT_USAGE_BUTTON_BACK,
} input_usage_t;

typedef enum input_collection_id {
    INPUT_COLLECTION_ID_TOUCH,
    INPUT_COLLECTION_ID_KEYBOARD,
    INPUT_COLLECTION_ID_MOUSE,
    INPUT_COLLECTION_ID_TOUCHPAD,
    INPUT_COLLECTION_ID_SWITCH,
    // etc
} input_collection_id_t;

typedef struct input_message input_message_t;

typedef struct input_host_callbacks {

    /**
     * Creates a device identifier with the given properties.
     * The unique ID should be a string that precisely identifies a given piece of hardware. For
     * example, an input device connected via Bluetooth could use its MAC address as its unique ID.
     */
    input_device_identifier_t* (*create_device_identifier)(input_host_t* host,
            const char* name, int32_t product_id, int32_t vendor_id,
            input_bus_t bus, const char* unique_id);

    /**
     * Allocates the device definition which will describe the input capabilities of a device. A
     * device definition may be used to register as many devices as desired.
     */
    input_device_definition_t* (*create_device_definition)(input_host_t* host);

    /**
     * Allocate either an input report, which the HAL will use to tell the host of incoming input
     * events, or an output report, which the host will use to tell the HAL of desired state
     * changes (e.g. setting an LED).
     */
    input_report_definition_t* (*create_input_report_definition)(input_host_t* host);
    input_report_definition_t* (*create_output_report_definition)(input_host_t* host);

    /**
     * Frees the report definition.
     */
    void (*free_report_definition)(input_host_t* host, input_report_definition_t* report_def);

    /**
     * Append the report to the given input device.
     */
    void (*input_device_definition_add_report)(input_host_t* host,
            input_device_definition_t* d, input_report_definition_t* r);

    /**
     * Add a collection with the given arity and ID. A collection describes a set
     * of logically grouped properties such as the X and Y coordinates of a single finger touch or
     * the set of keys on a keyboard. The arity declares how many repeated instances of this
     * collection will appear in whatever report it is attached to. The ID describes the type of
     * grouping being represented by the collection. For example, a touchscreen capable of
     * reporting up to 2 fingers simultaneously might have a collection with the X and Y
     * coordinates, an arity of 2, and an ID of INPUT_COLLECTION_USAGE_TOUCHSCREEN. Any given ID
     * may only be present once for a given report.
     */
    void (*input_report_definition_add_collection)(input_host_t* host,
            input_report_definition_t* report, input_collection_id_t id, int32_t arity);

    /**
     * Declare an int usage with the given properties. The report and collection defines where the
     * usage is being declared.
     */
    void (*input_report_definition_declare_usage_int)(input_host_t* host,
            input_report_definition_t* report, input_collection_id_t id,
            input_usage_t usage, int32_t min, int32_t max, float resolution);

    /**
     * Declare a set of boolean usages with the given properties.  The report and collection
     * defines where the usages are being declared.
     */
    void (*input_report_definition_declare_usages_bool)(input_host_t* host,
            input_report_definition_t* report, input_collection_id_t id,
            input_usage_t* usage, size_t usage_count);


    /**
     * Register a given input device definition. This notifies the host that an input device has
     * been connected and gives a description of all its capabilities.
     */
    input_device_handle_t* (*register_device)(input_host_t* host,
            input_device_identifier_t* id, input_device_definition_t* d);

    /** Unregister the given device */
    void (*unregister_device)(input_host_t* host, input_device_handle_t* handle);

    /**
     * Allocate a report that will contain all of the state as described by the given report.
     */
    input_report_t* (*input_allocate_report)(input_host_t* host, input_report_definition_t* r);

    /**
     * Add an int usage value to a report.
     */
    void (*input_report_set_usage_int)(input_host_t* host, input_report_t* r,
            input_collection_id_t id, input_usage_t usage, int32_t value, int32_t arity_index);

    /**
     * Add a boolean usage value to a report.
     */
    void (*input_report_set_usage_bool)(input_host_t* host, input_report_t* r,
            input_collection_id_t id, input_usage_t usage, bool value, int32_t arity_index);

    void (*report_event)(input_host_t* host, input_device_handle_t* d, input_report_t* report);

    /**
     * Retrieve the set of properties for the device. The returned
     * input_property_map_t* may be used to query specific properties via the
     * input_get_device_property callback.
     */
    input_property_map_t* (*input_get_device_property_map)(input_host_t* host,
            input_device_identifier_t* id);
    /**
     * Retrieve a property for the device with the given key. Returns NULL if
     * the key does not exist, or an input_property_t* that must be freed using
     * input_free_device_property(). Using an input_property_t after the
     * corresponding input_property_map_t is freed is undefined.
     */
    input_property_t* (*input_get_device_property)(input_host_t* host,
            input_property_map_t* map, const char* key);

    /**
     * Get the key for the input property. Returns NULL if the property is NULL.
     * The returned const char* is owned by the input_property_t.
     */
    const char* (*input_get_property_key)(input_host_t* host, input_property_t* property);

    /**
     * Get the value for the input property. Returns NULL if the property is
     * NULL. The returned const char* is owned by the input_property_t.
     */
    const char* (*input_get_property_value)(input_host_t* host, input_property_t* property);

    /**
     * Frees the input_property_t*.
     */
    void (*input_free_device_property)(input_host_t* host, input_property_t* property);

    /**
     * Frees the input_property_map_t*.
     */
    void (*input_free_device_property_map)(input_host_t* host, input_property_map_t* map);
} input_host_callbacks_t;

typedef struct input_module input_module_t;

struct input_module {
    /**
     * Common methods of the input module. This *must* be the first member
     * of input_module as users of this structure will cast a hw_module_t
     * to input_module pointer in contexts where it's known
     * the hw_module_t references a input_module.
     */
    struct hw_module_t common;

    /**
     * Initialize the module with host callbacks. At this point the HAL should start up whatever
     * infrastructure it needs to in order to process input events.
     */
    void (*init)(const input_module_t* module, input_host_t* host, input_host_callbacks_t cb);

    /**
     * Sends an output report with a new set of state the host would like the given device to
     * assume.
     */
    void (*notify_report)(const input_module_t* module, input_report_t* report);
};

static inline int input_open(const struct hw_module_t** module, const char* type) {
    return hw_get_module_by_class(INPUT_HARDWARE_MODULE_ID, type, module);
}

__END_DECLS

#endif  /* ANDROID_INCLUDE_HARDWARE_INPUT_H */
