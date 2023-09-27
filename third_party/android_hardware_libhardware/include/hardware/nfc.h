/*
 * Copyright (C) 2011, 2012 The Android Open Source Project
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

#ifndef ANDROID_NFC_HAL_INTERFACE_H
#define ANDROID_NFC_HAL_INTERFACE_H

#include <stdint.h>
#include <strings.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS


/* NFC device HAL for NCI-based NFC controllers.
 *
 * This HAL allows NCI silicon vendors to make use
 * of the core NCI stack in Android for their own silicon.
 *
 * The responibilities of the NCI HAL implementation
 * are as follows:
 *
 * - Implement the transport to the NFC controller
 * - Implement each of the HAL methods specified below as applicable to their silicon
 * - Pass up received NCI messages from the controller to the stack
 *
 * A simplified timeline of NCI HAL method calls:
 * 1) Core NCI stack calls open()
 * 2) Core NCI stack executes CORE_RESET and CORE_INIT through calls to write()
 * 3) Core NCI stack calls core_initialized() to allow HAL to do post-init configuration
 * 4) Core NCI stack calls pre_discover() to allow HAL to prepare for RF discovery
 * 5) Core NCI stack starts discovery through calls to write()
 * 6) Core NCI stack stops discovery through calls to write() (e.g. screen turns off)
 * 7) Core NCI stack calls pre_discover() to prepare for RF discovery (e.g. screen turned back on)
 * 8) Core NCI stack starts discovery through calls to write()
 * ...
 * ...
 * 9) Core NCI stack calls close()
 */
#define NFC_NCI_HARDWARE_MODULE_ID "nfc_nci"
#define NFC_NCI_BCM2079X_HARDWARE_MODULE_ID "nfc_nci.bcm2079x"
#define NFC_NCI_NXP_PN54X_HARDWARE_MODULE_ID "nfc_nci.pn54x"
#define NFC_NCI_CONTROLLER "nci"

/*
 *  nfc_nci_module_t should contain module-specific parameters
 */
typedef struct nfc_nci_module_t {
    /**
     * Common methods of the NFC NCI module.  This *must* be the first member of
     * nfc_nci_module_t as users of this structure will cast a hw_module_t to
     * nfc_nci_module_t pointer in contexts where it's known the hw_module_t references a
     * nfc_nci_module_t.
     */
    struct hw_module_t common;
} nfc_nci_module_t;

/*
 * HAL events that can be passed back to the stack
 */
typedef uint8_t nfc_event_t;

enum {
    HAL_NFC_OPEN_CPLT_EVT           = 0x00,
    HAL_NFC_CLOSE_CPLT_EVT          = 0x01,
    HAL_NFC_POST_INIT_CPLT_EVT      = 0x02,
    HAL_NFC_PRE_DISCOVER_CPLT_EVT   = 0x03,
    HAL_NFC_REQUEST_CONTROL_EVT     = 0x04,
    HAL_NFC_RELEASE_CONTROL_EVT     = 0x05,
    HAL_NFC_ERROR_EVT               = 0x06
};

/*
 * Allowed status return values for each of the HAL methods
 */
typedef uint8_t nfc_status_t;

enum {
    HAL_NFC_STATUS_OK               = 0x00,
    HAL_NFC_STATUS_FAILED           = 0x01,
    HAL_NFC_STATUS_ERR_TRANSPORT    = 0x02,
    HAL_NFC_STATUS_ERR_CMD_TIMEOUT  = 0x03,
    HAL_NFC_STATUS_REFUSED          = 0x04
};

/*
 * The callback passed in from the NFC stack that the HAL
 * can use to pass events back to the stack.
 */
typedef void (nfc_stack_callback_t) (nfc_event_t event, nfc_status_t event_status);

/*
 * The callback passed in from the NFC stack that the HAL
 * can use to pass incomming data to the stack.
 */
typedef void (nfc_stack_data_callback_t) (uint16_t data_len, uint8_t* p_data);

/* nfc_nci_device_t starts with a hw_device_t struct,
 * followed by device-specific methods and members.
 *
 * All methods in the NCI HAL are asynchronous.
 */
typedef struct nfc_nci_device {
    /**
     * Common methods of the NFC NCI device.  This *must* be the first member of
     * nfc_nci_device_t as users of this structure will cast a hw_device_t to
     * nfc_nci_device_t pointer in contexts where it's known the hw_device_t references a
     * nfc_nci_device_t.
     */
    struct hw_device_t common;
    /*
     * (*open)() Opens the NFC controller device and performs initialization.
     * This may include patch download and other vendor-specific initialization.
     *
     * If open completes successfully, the controller should be ready to perform
     * NCI initialization - ie accept CORE_RESET and subsequent commands through
     * the write() call.
     *
     * If open() returns 0, the NCI stack will wait for a HAL_NFC_OPEN_CPLT_EVT
     * before continuing.
     *
     * If open() returns any other value, the NCI stack will stop.
     *
     */
    int (*open)(const struct nfc_nci_device *p_dev, nfc_stack_callback_t *p_cback,
            nfc_stack_data_callback_t *p_data_cback);

    /*
     * (*write)() Performs an NCI write.
     *
     * This method may queue writes and return immediately. The only
     * requirement is that the writes are executed in order.
     */
    int (*write)(const struct nfc_nci_device *p_dev, uint16_t data_len, const uint8_t *p_data);

    /*
     * (*core_initialized)() is called after the CORE_INIT_RSP is received from the NFCC.
     * At this time, the HAL can do any chip-specific configuration.
     *
     * If core_initialized() returns 0, the NCI stack will wait for a HAL_NFC_POST_INIT_CPLT_EVT
     * before continuing.
     *
     * If core_initialized() returns any other value, the NCI stack will continue
     * immediately.
     */
    int (*core_initialized)(const struct nfc_nci_device *p_dev, uint8_t* p_core_init_rsp_params);

    /*
     * (*pre_discover)() Is called every time before starting RF discovery.
     * It is a good place to do vendor-specific configuration that must be
     * performed every time RF discovery is about to be started.
     *
     * If pre_discover() returns 0, the NCI stack will wait for a HAL_NFC_PRE_DISCOVER_CPLT_EVT
     * before continuing.
     *
     * If pre_discover() returns any other value, the NCI stack will start
     * RF discovery immediately.
     */
    int (*pre_discover)(const struct nfc_nci_device *p_dev);

    /*
     * (*close)() Closed the NFC controller. Should free all resources.
     */
    int (*close)(const struct nfc_nci_device *p_dev);

    /*
     * (*control_granted)() Grant HAL the exclusive control to send NCI commands.
     * Called in response to HAL_REQUEST_CONTROL_EVT.
     * Must only be called when there are no NCI commands pending.
     * HAL_RELEASE_CONTROL_EVT will notify when HAL no longer needs exclusive control.
     */
    int (*control_granted)(const struct nfc_nci_device *p_dev);

    /*
     * (*power_cycle)() Restart controller by power cyle;
     * HAL_OPEN_CPLT_EVT will notify when operation is complete.
     */
    int (*power_cycle)(const struct nfc_nci_device *p_dev);
} nfc_nci_device_t;

/*
 * Convenience methods that the NFC stack can use to open
 * and close an NCI device
 */
static inline int nfc_nci_open(const struct hw_module_t* module,
        nfc_nci_device_t** dev) {
    return module->methods->open(module, NFC_NCI_CONTROLLER,
        (struct hw_device_t**) dev);
}

static inline int nfc_nci_close(nfc_nci_device_t* dev) {
    return dev->common.close(&dev->common);
}
/*
 * End NFC NCI HAL
 */

/*
 * This is a limited NFC HAL for NXP PN544-based devices.
 * This HAL as Android is moving to
 * an NCI-based NFC stack.
 *
 * All NCI-based NFC controllers should use the NFC-NCI
 * HAL instead.
 * Begin PN544 specific HAL
 */
#define NFC_HARDWARE_MODULE_ID "nfc"

#define NFC_PN544_CONTROLLER "pn544"

typedef struct nfc_module_t {
    /**
     * Common methods of the NFC NXP PN544 module.  This *must* be the first member of
     * nfc_module_t as users of this structure will cast a hw_module_t to
     * nfc_module_t pointer in contexts where it's known the hw_module_t references a
     * nfc_module_t.
     */
    struct hw_module_t common;
} nfc_module_t;

/*
 * PN544 linktypes.
 * UART
 * I2C
 * USB (uses UART DAL)
 */
typedef enum {
    PN544_LINK_TYPE_UART,
    PN544_LINK_TYPE_I2C,
    PN544_LINK_TYPE_USB,
    PN544_LINK_TYPE_INVALID,
} nfc_pn544_linktype;

typedef struct {
    /**
     * Common methods of the NFC NXP PN544 device.  This *must* be the first member of
     * nfc_pn544_device_t as users of this structure will cast a hw_device_t to
     * nfc_pn544_device_t pointer in contexts where it's known the hw_device_t references a
     * nfc_pn544_device_t.
     */
    struct hw_device_t common;

    /* The number of EEPROM registers to write */
    uint32_t num_eeprom_settings;

    /* The actual EEPROM settings
     * For PN544, each EEPROM setting is a 4-byte entry,
     * of the format [0x00, addr_msb, addr_lsb, value].
     */
    uint8_t* eeprom_settings;

    /* The link type to which the PN544 is connected */
    nfc_pn544_linktype linktype;

    /* The device node to which the PN544 is connected */
    const char* device_node;

    /* On Crespo we had an I2C issue that would cause us to sometimes read
     * the I2C slave address (0x57) over the bus. libnfc contains
     * a hack to ignore this byte and try to read the length byte
     * again.
     * Set to 0 to disable the workaround, 1 to enable it.
     */
    uint8_t enable_i2c_workaround;
    /* I2C slave address. Multiple I2C addresses are
     * possible for PN544 module. Configure address according to
     * board design.
     */
    uint8_t i2c_device_address;
} nfc_pn544_device_t;

static inline int nfc_pn544_open(const struct hw_module_t* module,
        nfc_pn544_device_t** dev) {
    return module->methods->open(module, NFC_PN544_CONTROLLER,
        (struct hw_device_t**) dev);
}

static inline int nfc_pn544_close(nfc_pn544_device_t* dev) {
    return dev->common.close(&dev->common);
}
/*
 * End PN544 specific HAL
 */

__END_DECLS

#endif // ANDROID_NFC_HAL_INTERFACE_H
