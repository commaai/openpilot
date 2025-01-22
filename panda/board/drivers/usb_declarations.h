#pragma once

// IRQs: OTG_FS

typedef union {
  uint16_t w;
  struct BW {
    uint8_t msb;
    uint8_t lsb;
  }
  bw;
} uint16_t_uint8_t;

typedef union _USB_Setup {
  uint32_t d8[2];
  struct _SetupPkt_Struc
  {
    uint8_t           bmRequestType;
    uint8_t           bRequest;
    uint16_t_uint8_t  wValue;
    uint16_t_uint8_t  wIndex;
    uint16_t_uint8_t  wLength;
  } b;
} USB_Setup_TypeDef;

extern bool usb_enumerated;

void usb_init(void);
void refresh_can_tx_slots_available(void);

// **** supporting defines ****
#define  USB_REQ_GET_STATUS                             0x00
#define  USB_REQ_CLEAR_FEATURE                          0x01
#define  USB_REQ_SET_FEATURE                            0x03
#define  USB_REQ_SET_ADDRESS                            0x05
#define  USB_REQ_GET_DESCRIPTOR                         0x06
#define  USB_REQ_SET_DESCRIPTOR                         0x07
#define  USB_REQ_GET_CONFIGURATION                      0x08
#define  USB_REQ_SET_CONFIGURATION                      0x09
#define  USB_REQ_GET_INTERFACE                          0x0A
#define  USB_REQ_SET_INTERFACE                          0x0B
#define  USB_REQ_SYNCH_FRAME                            0x0C

#define  USB_DESC_TYPE_DEVICE                           0x01
#define  USB_DESC_TYPE_CONFIGURATION                    0x02
#define  USB_DESC_TYPE_STRING                           0x03
#define  USB_DESC_TYPE_INTERFACE                        0x04
#define  USB_DESC_TYPE_ENDPOINT                         0x05
#define  USB_DESC_TYPE_DEVICE_QUALIFIER                 0x06
#define  USB_DESC_TYPE_OTHER_SPEED_CONFIGURATION        0x07
#define  USB_DESC_TYPE_BINARY_OBJECT_STORE              0x0f

// offsets for configuration strings
#define  STRING_OFFSET_LANGID                           0x00
#define  STRING_OFFSET_IMANUFACTURER                    0x01
#define  STRING_OFFSET_IPRODUCT                         0x02
#define  STRING_OFFSET_ISERIAL                          0x03
#define  STRING_OFFSET_ICONFIGURATION                   0x04
#define  STRING_OFFSET_IINTERFACE                       0x05

// WebUSB requests
#define  WEBUSB_REQ_GET_URL                             0x02

// WebUSB types
#define  WEBUSB_DESC_TYPE_URL                           0x03
#define  WEBUSB_URL_SCHEME_HTTPS                        0x01
#define  WEBUSB_URL_SCHEME_HTTP                         0x00

// WinUSB requests
#define  WINUSB_REQ_GET_COMPATID_DESCRIPTOR             0x04
#define  WINUSB_REQ_GET_EXT_PROPS_OS                    0x05
#define  WINUSB_REQ_GET_DESCRIPTOR                      0x07

#define STS_GOUT_NAK                           1
#define STS_DATA_UPDT                          2
#define STS_XFER_COMP                          3
#define STS_SETUP_COMP                         4
#define STS_SETUP_UPDT                         6

// for the repeating interfaces
#define DSCR_INTERFACE_LEN 9
#define DSCR_ENDPOINT_LEN 7
#define DSCR_CONFIG_LEN 9
#define DSCR_DEVICE_LEN 18

// endpoint types
#define ENDPOINT_TYPE_CONTROL 0
#define ENDPOINT_TYPE_ISO 1
#define ENDPOINT_TYPE_BULK 2
#define ENDPOINT_TYPE_INT 3

// These are arbitrary values used in bRequest
#define  MS_VENDOR_CODE 0x20
#define  WEBUSB_VENDOR_CODE 0x30

// BOS constants
#define BINARY_OBJECT_STORE_DESCRIPTOR_LENGTH   0x05
#define BINARY_OBJECT_STORE_DESCRIPTOR          0x0F
#define WINUSB_PLATFORM_DESCRIPTOR_LENGTH       0x9E

// Convert machine byte order to USB byte order
#define TOUSBORDER(num)\
  ((num) & 0xFFU), (((uint16_t)(num) >> 8) & 0xFFU)

// take in string length and return the first 2 bytes of a string descriptor
#define STRING_DESCRIPTOR_HEADER(size)\
  (((((size) * 2) + 2) & 0xFF) | 0x0300)

#define ENDPOINT_RCV 0x80
#define ENDPOINT_SND 0x00

// packet read and write
void usb_tick(void);
// ***************************** USB port *****************************
void can_tx_comms_resume_usb(void);
