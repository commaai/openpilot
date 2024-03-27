// IRQs: OTG_FS

typedef union {
  uint16_t w;
  struct BW {
    uint8_t msb;
    uint8_t lsb;
  }
  bw;
}
uint16_t_uint8_t;

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
}
USB_Setup_TypeDef;

bool usb_enumerated = false;
uint16_t usb_last_frame_num = 0U;

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

uint8_t response[USBPACKET_MAX_SIZE];

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

uint8_t device_desc[] = {
  DSCR_DEVICE_LEN, USB_DESC_TYPE_DEVICE, //Length, Type
  0x10, 0x02, // bcdUSB max version of USB supported (2.1)
  0xFF, 0xFF, 0xFF, 0x40, // Class, Subclass, Protocol, Max Packet Size
  TOUSBORDER(USB_VID), // idVendor
  TOUSBORDER(USB_PID), // idProduct
  0x00, 0x00, // bcdDevice
  0x01, 0x02, // Manufacturer, Product
  0x03, 0x01 // Serial Number, Num Configurations
};

uint8_t device_qualifier[] = {
  0x0a, USB_DESC_TYPE_DEVICE_QUALIFIER, //Length, Type
  0x10, 0x02, // bcdUSB max version of USB supported (2.1)
  0xFF, 0xFF, 0xFF, 0x40, // bDeviceClass, bDeviceSubClass, bDeviceProtocol, bMaxPacketSize0
  0x01, 0x00 // bNumConfigurations, bReserved
};

#define ENDPOINT_RCV 0x80
#define ENDPOINT_SND 0x00

uint8_t configuration_desc[] = {
  DSCR_CONFIG_LEN, USB_DESC_TYPE_CONFIGURATION, // Length, Type,
  TOUSBORDER(0x0045U), // Total Len (uint16)
  0x01, 0x01, STRING_OFFSET_ICONFIGURATION, // Num Interface, Config Value, Configuration
  0xc0, 0x32, // Attributes, Max Power
  // interface 0 ALT 0
  DSCR_INTERFACE_LEN, USB_DESC_TYPE_INTERFACE, // Length, Type
  0x00, 0x00, 0x03, // Index, Alt Index idx, Endpoint count
  0XFF, 0xFF, 0xFF, // Class, Subclass, Protocol
  0x00, // Interface
    // endpoint 1, read CAN
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_RCV | 1, ENDPOINT_TYPE_BULK, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x00, // Polling Interval (NA)
    // endpoint 2, send serial
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_SND | 2, ENDPOINT_TYPE_BULK, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x00, // Polling Interval
    // endpoint 3, send CAN
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_SND | 3, ENDPOINT_TYPE_BULK, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x00, // Polling Interval
  // interface 0 ALT 1
  DSCR_INTERFACE_LEN, USB_DESC_TYPE_INTERFACE, // Length, Type
  0x00, 0x01, 0x03, // Index, Alt Index idx, Endpoint count
  0XFF, 0xFF, 0xFF, // Class, Subclass, Protocol
  0x00, // Interface
    // endpoint 1, read CAN
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_RCV | 1, ENDPOINT_TYPE_INT, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x05, // Polling Interval (5 frames)
    // endpoint 2, send serial
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_SND | 2, ENDPOINT_TYPE_BULK, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x00, // Polling Interval
    // endpoint 3, send CAN
    DSCR_ENDPOINT_LEN, USB_DESC_TYPE_ENDPOINT, // Length, Type
    ENDPOINT_SND | 3, ENDPOINT_TYPE_BULK, // Endpoint Num/Direction, Type
    TOUSBORDER(0x0040U), // Max Packet (0x0040)
    0x00, // Polling Interval
};

// STRING_DESCRIPTOR_HEADER is for uint16 string descriptors
// it takes in a string length, which is bytes/2 because unicode
uint16_t string_language_desc[] = {
  STRING_DESCRIPTOR_HEADER(1),
  0x0409 // american english
};

// these strings are all uint16's so that we don't need to spam ,0 after every character
uint16_t string_manufacturer_desc[] = {
  STRING_DESCRIPTOR_HEADER(8),
  'c', 'o', 'm', 'm', 'a', '.', 'a', 'i'
};

uint16_t string_product_desc[] = {
  STRING_DESCRIPTOR_HEADER(5),
  'p', 'a', 'n', 'd', 'a'
};

// default serial number when we're not a panda
uint16_t string_serial_desc[] = {
  STRING_DESCRIPTOR_HEADER(4),
  'n', 'o', 'n', 'e'
};

// a string containing the default configuration index
uint16_t string_configuration_desc[] = {
  STRING_DESCRIPTOR_HEADER(2),
  '0', '1' // "01"
};

// WCID (auto install WinUSB driver)
// https://github.com/pbatard/libwdi/wiki/WCID-Devices
// https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/winusb-installation#automatic-installation-of--winusb-without-an-inf-file
// WinUSB 1.0 descriptors, this is mostly used by Windows XP
uint8_t string_238_desc[] = {
  0x12, USB_DESC_TYPE_STRING, // bLength, bDescriptorType
  'M',0, 'S',0, 'F',0, 'T',0, '1',0, '0',0, '0',0, // qwSignature (MSFT100)
  MS_VENDOR_CODE, 0x00 // bMS_VendorCode, bPad
};
uint8_t winusb_ext_compatid_os_desc[] = {
  0x28, 0x00, 0x00, 0x00, // dwLength
  0x00, 0x01, // bcdVersion
  0x04, 0x00, // wIndex
  0x01, // bCount
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Reserved
  0x00, // bFirstInterfaceNumber
  0x00, // Reserved
  'W', 'I', 'N', 'U', 'S', 'B', 0x00, 0x00, // compatible ID (WINUSB)
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // subcompatible ID (none)
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00 // Reserved
};
uint8_t winusb_ext_prop_os_desc[] = {
  0x8e, 0x00, 0x00, 0x00, // dwLength
  0x00, 0x01, // bcdVersion
  0x05, 0x00, // wIndex
  0x01, 0x00, // wCount
  // first property
  0x84, 0x00, 0x00, 0x00, // dwSize
  0x01, 0x00, 0x00, 0x00, // dwPropertyDataType
  0x28, 0x00, // wPropertyNameLength
  'D',0, 'e',0, 'v',0, 'i',0, 'c',0, 'e',0, 'I',0, 'n',0, 't',0, 'e',0, 'r',0, 'f',0, 'a',0, 'c',0, 'e',0, 'G',0, 'U',0, 'I',0, 'D',0, 0, 0, // bPropertyName (DeviceInterfaceGUID)
  0x4e, 0x00, 0x00, 0x00, // dwPropertyDataLength
  '{',0, 'c',0, 'c',0, 'e',0, '5',0, '2',0, '9',0, '1',0, 'c',0, '-',0, 'a',0, '6',0, '9',0, 'f',0, '-',0, '4',0 ,'9',0 ,'9',0 ,'5',0 ,'-',0, 'a',0, '4',0, 'c',0, '2',0, '-',0, '2',0, 'a',0, 'e',0, '5',0, '7',0, 'a',0, '5',0, '1',0, 'a',0, 'd',0, 'e',0, '9',0, '}',0, 0, 0, // bPropertyData ({CCE5291C-A69F-4995-A4C2-2AE57A51ADE9})
};

/*
Binary Object Store descriptor used to expose WebUSB (and more WinUSB) metadata
comments are from the wicg spec
References used:
  https://wicg.github.io/webusb/#webusb-platform-capability-descriptor
  https://github.com/sowbug/weblight/blob/192ad7a0e903542e2aa28c607d98254a12a6399d/firmware/webusb.c
  https://os.mbed.com/users/larsgk/code/USBDevice_WebUSB/file/1d8a6665d607/WebUSBDevice/

*/
uint8_t binary_object_store_desc[] = {
  // BOS header
  BINARY_OBJECT_STORE_DESCRIPTOR_LENGTH, // bLength, this is only the length of the header
  BINARY_OBJECT_STORE_DESCRIPTOR, // bDescriptorType
  0x39, 0x00, // wTotalLength (LSB, MSB)
  0x02, // bNumDeviceCaps (WebUSB + WinUSB)

  // -------------------------------------------------
  // WebUSB descriptor
  // header
    0x18, // bLength, Size of this descriptor. Must be set to 24.
    0x10, // bDescriptorType, DEVICE CAPABILITY descriptor
    0x05, // bDevCapabilityType, PLATFORM capability
    0x00, // bReserved, This field is reserved and shall be set to zero.

  // PlatformCapabilityUUID, Must be set to {3408b638-09a9-47a0-8bfd-a0768815b665}.
    0x38, 0xB6, 0x08, 0x34,
    0xA9, 0x09, 0xA0, 0x47,
    0x8B, 0xFD, 0xA0, 0x76,
    0x88, 0x15, 0xB6, 0x65,
  // </PlatformCapabilityUUID>

  0x00, 0x01, // bcdVersion, Protocol version supported. Must be set to 0x0100.
  WEBUSB_VENDOR_CODE, // bVendorCode, bRequest value used for issuing WebUSB requests.
  // there used to be a concept of "allowed origins", but it was removed from the spec
  // it was intended to be a security feature, but then the entire security model relies on domain ownership
  // https://github.com/WICG/webusb/issues/49
  // other implementations use various other indexed to leverate this no-longer-valid feature. we wont.
  // the spec says we *must* reply to index 0x03 with the url, so we'll hint that that's the right index
  0x03, // iLandingPage, URL descriptor index of the device’s landing page.

  // -------------------------------------------------
  // WinUSB descriptor
  // header
    0x1C, // Descriptor size (28 bytes)
    0x10, // Descriptor type (Device Capability)
    0x05, // Capability type (Platform)
    0x00, // Reserved

  // MS OS 2.0 Platform Capability ID (D8DD60DF-4589-4CC7-9CD2-659D9E648A9F)
  // Indicates the device supports the Microsoft OS 2.0 descriptor
    0xDF, 0x60, 0xDD, 0xD8,
    0x89, 0x45, 0xC7, 0x4C,
    0x9C, 0xD2, 0x65, 0x9D,
    0x9E, 0x64, 0x8A, 0x9F,

  0x00, 0x00, 0x03, 0x06, // Windows version, currently set to 8.1 (0x06030000)

  WINUSB_PLATFORM_DESCRIPTOR_LENGTH, 0x00, // MS OS 2.0 descriptor size (word)
  MS_VENDOR_CODE, 0x00 // vendor code, no alternate enumeration
};

uint8_t webusb_url_descriptor[] = {
  0x14,                  /* bLength */
  WEBUSB_DESC_TYPE_URL, // bDescriptorType
  WEBUSB_URL_SCHEME_HTTPS, // bScheme
  'u', 's', 'b', 'p', 'a', 'n', 'd', 'a', '.', 'c', 'o', 'm', 'm', 'a', '.', 'a', 'i'
};

// WinUSB 2.0 descriptor. This is what modern systems use
// https://github.com/sowbug/weblight/blob/192ad7a0e903542e2aa28c607d98254a12a6399d/firmware/webusb.c
// http://janaxelson.com/files/ms_os_20_descriptors.c
// https://books.google.com/books?id=pkefBgAAQBAJ&pg=PA353&lpg=PA353
uint8_t winusb_20_desc[WINUSB_PLATFORM_DESCRIPTOR_LENGTH] = {
  // Microsoft OS 2.0 descriptor set header (table 10)
  0x0A, 0x00, // Descriptor size (10 bytes)
  0x00, 0x00, // MS OS 2.0 descriptor set header

  0x00, 0x00, 0x03, 0x06, // Windows version (8.1) (0x06030000)
  WINUSB_PLATFORM_DESCRIPTOR_LENGTH, 0x00, // Total size of MS OS 2.0 descriptor set

  // Microsoft OS 2.0 compatible ID descriptor
    0x14, 0x00, // Descriptor size (20 bytes)
    0x03, 0x00, // MS OS 2.0 compatible ID descriptor
    'W', 'I', 'N', 'U', 'S', 'B', 0x00, 0x00, // compatible ID (WINUSB)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // Sub-compatible ID

  // Registry property descriptor
  0x80, 0x00, // Descriptor size (130 bytes)
  0x04, 0x00, // Registry Property descriptor
  0x01, 0x00, // Strings are null-terminated Unicode
  0x28, 0x00, // Size of Property Name (40 bytes) "DeviceInterfaceGUID"

  // bPropertyName (DeviceInterfaceGUID)
    'D', 0x00, 'e', 0x00, 'v', 0x00, 'i', 0x00, 'c', 0x00, 'e', 0x00, 'I', 0x00, 'n', 0x00,
    't', 0x00, 'e', 0x00, 'r', 0x00, 'f', 0x00, 'a', 0x00, 'c', 0x00, 'e', 0x00, 'G', 0x00,
    'U', 0x00, 'I', 0x00, 'D', 0x00, 0x00, 0x00,

  0x4E, 0x00, // Size of Property Data (78 bytes)

  // Vendor-defined property data: {CCE5291C-A69F-4995-A4C2-2AE57A51ADE9}
    '{', 0x00, 'c', 0x00, 'c', 0x00, 'e', 0x00, '5', 0x00, '2', 0x00, '9', 0x00, '1', 0x00, // 16
    'c', 0x00, '-', 0x00, 'a', 0x00, '6', 0x00, '9', 0x00, 'f', 0x00, '-', 0x00, '4', 0x00, // 32
    '9', 0x00, '9', 0x00, '5', 0x00, '-', 0x00, 'a', 0x00, '4', 0x00, 'c', 0x00, '2', 0x00, // 48
    '-', 0x00, '2', 0x00, 'a', 0x00, 'e', 0x00, '5', 0x00, '7', 0x00, 'a', 0x00, '5', 0x00, // 64
    '1', 0x00, 'a', 0x00, 'd', 0x00, 'e', 0x00, '9', 0x00, '}', 0x00, 0x00, 0x00 // 78 bytes
};

// current packet
USB_Setup_TypeDef setup;
uint8_t usbdata[0x100] __attribute__((aligned(4)));
uint8_t* ep0_txdata = NULL;
uint16_t ep0_txlen = 0;
bool outep3_processing = false;

// Store the current interface alt setting.
int current_int0_alt_setting = 0;

// packet read and write

void *USB_ReadPacket(void *dest, uint16_t len) {
  uint32_t *dest_copy = (uint32_t *)dest;
  uint32_t count32b = ((uint32_t)len + 3U) / 4U;

  for (uint32_t i = 0; i < count32b; i++) {
    *dest_copy = USBx_DFIFO(0U);
    dest_copy++;
  }
  return ((void *)dest_copy);
}

void USB_WritePacket(const void *src, uint16_t len, uint32_t ep) {
  #ifdef DEBUG_USB
  print("writing ");
  hexdump(src, len);
  #endif

  uint32_t numpacket = ((uint32_t)len + (USBPACKET_MAX_SIZE - 1U)) / USBPACKET_MAX_SIZE;
  uint32_t count32b = 0;
  count32b = ((uint32_t)len + 3U) / 4U;

  // TODO: revisit this
  USBx_INEP(ep)->DIEPTSIZ = ((numpacket << 19) & USB_OTG_DIEPTSIZ_PKTCNT) |
                            (len               & USB_OTG_DIEPTSIZ_XFRSIZ);
  USBx_INEP(ep)->DIEPCTL |= (USB_OTG_DIEPCTL_CNAK | USB_OTG_DIEPCTL_EPENA);

  // load the FIFO
  if (src != NULL) {
    const uint32_t *src_copy = (const uint32_t *)src;
    for (uint32_t i = 0; i < count32b; i++) {
      USBx_DFIFO(ep) = *src_copy;
      src_copy++;
    }
  }
}

// IN EP 0 TX FIFO has a max size of 127 bytes (much smaller than the rest)
// so use TX FIFO empty interrupt to send larger amounts of data
void USB_WritePacket_EP0(uint8_t *src, uint16_t len) {
  #ifdef DEBUG_USB
  print("writing ");
  hexdump(src, len);
  #endif

  uint16_t wplen = MIN(len, 0x40);
  USB_WritePacket(src, wplen, 0);

  if (wplen < len) {
    ep0_txdata = &src[wplen];
    ep0_txlen = len - wplen;
    USBx_DEVICE->DIEPEMPMSK |= 1;
  } else {
    USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
  }
}

void usb_reset(void) {
  // unmask endpoint interrupts, so many sets
  USBx_DEVICE->DAINT = 0xFFFFFFFFU;
  USBx_DEVICE->DAINTMSK = 0xFFFFFFFFU;
  //USBx_DEVICE->DOEPMSK = (USB_OTG_DOEPMSK_STUPM | USB_OTG_DOEPMSK_XFRCM | USB_OTG_DOEPMSK_EPDM);
  //USBx_DEVICE->DIEPMSK = (USB_OTG_DIEPMSK_TOM | USB_OTG_DIEPMSK_XFRCM | USB_OTG_DIEPMSK_EPDM | USB_OTG_DIEPMSK_ITTXFEMSK);
  //USBx_DEVICE->DIEPMSK = (USB_OTG_DIEPMSK_TOM | USB_OTG_DIEPMSK_XFRCM | USB_OTG_DIEPMSK_EPDM);

  // all interrupts for debugging
  USBx_DEVICE->DIEPMSK = 0xFFFFFFFFU;
  USBx_DEVICE->DOEPMSK = 0xFFFFFFFFU;

  // clear interrupts
  USBx_INEP(0U)->DIEPINT = 0xFF;
  USBx_OUTEP(0U)->DOEPINT = 0xFF;

  // unset the address
  USBx_DEVICE->DCFG &= ~USB_OTG_DCFG_DAD;

  // set up USB FIFOs
  // RX start address is fixed to 0
  USBx->GRXFSIZ = 0x40;

  // 0x100 to offset past GRXFSIZ
  USBx->DIEPTXF0_HNPTXFSIZ = (0x40UL << 16) | 0x40U;

  // EP1, massive
  USBx->DIEPTXF[0] = (0x40UL << 16) | 0x80U;

  // flush TX fifo
  USBx->GRSTCTL = USB_OTG_GRSTCTL_TXFFLSH | USB_OTG_GRSTCTL_TXFNUM_4;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_TXFFLSH) == USB_OTG_GRSTCTL_TXFFLSH);
  // flush RX FIFO
  USBx->GRSTCTL = USB_OTG_GRSTCTL_RXFFLSH;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_RXFFLSH) == USB_OTG_GRSTCTL_RXFFLSH);

  // no global NAK
  USBx_DEVICE->DCTL |= USB_OTG_DCTL_CGINAK;

  // ready to receive setup packets
  USBx_OUTEP(0U)->DOEPTSIZ = USB_OTG_DOEPTSIZ_STUPCNT | (USB_OTG_DOEPTSIZ_PKTCNT & (1UL << 19)) | (3U << 3);
}

char to_hex_char(uint8_t a) {
  char ret;
  if (a < 10U) {
    ret = '0' + a;
  } else {
    ret = 'a' + (a - 10U);
  }
  return ret;
}

void usb_tick(void) {
  uint16_t current_frame_num = (USBx_DEVICE->DSTS & USB_OTG_DSTS_FNSOF_Msk) >> USB_OTG_DSTS_FNSOF_Pos;
  usb_enumerated = (current_frame_num != usb_last_frame_num);
  usb_last_frame_num = current_frame_num;
}

void usb_setup(void) {
  int resp_len;
  ControlPacket_t control_req;

  // setup packet is ready
  switch (setup.b.bRequest) {
    case USB_REQ_SET_CONFIGURATION:
      // enable other endpoints, has to be here?
      USBx_INEP(1U)->DIEPCTL = (0x40U & USB_OTG_DIEPCTL_MPSIZ) | (2UL << 18) | (1UL << 22) |
                              USB_OTG_DIEPCTL_SD0PID_SEVNFRM | USB_OTG_DIEPCTL_USBAEP;
      USBx_INEP(1U)->DIEPINT = 0xFF;

      USBx_OUTEP(2U)->DOEPTSIZ = (1UL << 19) | 0x40U;
      USBx_OUTEP(2U)->DOEPCTL = (0x40U & USB_OTG_DOEPCTL_MPSIZ) | (2UL << 18) |
                               USB_OTG_DOEPCTL_SD0PID_SEVNFRM | USB_OTG_DOEPCTL_USBAEP;
      USBx_OUTEP(2U)->DOEPINT = 0xFF;

      USBx_OUTEP(3U)->DOEPTSIZ = (32UL << 19) | 0x800U;
      USBx_OUTEP(3U)->DOEPCTL = (0x40U & USB_OTG_DOEPCTL_MPSIZ) | (2UL << 18) |
                               USB_OTG_DOEPCTL_SD0PID_SEVNFRM | USB_OTG_DOEPCTL_USBAEP;
      USBx_OUTEP(3U)->DOEPINT = 0xFF;

      // mark ready to receive
      USBx_OUTEP(2U)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
      USBx_OUTEP(3U)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;

      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case USB_REQ_SET_ADDRESS:
      // set now?
      USBx_DEVICE->DCFG |= ((setup.b.wValue.w & 0x7fU) << 4);

      #ifdef DEBUG_USB
        print(" set address\n");
      #endif

      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;

      break;
    case USB_REQ_GET_DESCRIPTOR:
      switch (setup.b.wValue.bw.lsb) {
        case USB_DESC_TYPE_DEVICE:
          //print("    writing device descriptor\n");

          // set bcdDevice to hardware type
          device_desc[13] = hw_type;
          // setup transfer
          USB_WritePacket(device_desc, MIN(sizeof(device_desc), setup.b.wLength.w), 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;

          //print("D");
          break;
        case USB_DESC_TYPE_CONFIGURATION:
          USB_WritePacket(configuration_desc, MIN(sizeof(configuration_desc), setup.b.wLength.w), 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        case USB_DESC_TYPE_DEVICE_QUALIFIER:
          USB_WritePacket(device_qualifier, MIN(sizeof(device_qualifier), setup.b.wLength.w), 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        case USB_DESC_TYPE_STRING:
          switch (setup.b.wValue.bw.msb) {
            case STRING_OFFSET_LANGID:
              USB_WritePacket((uint8_t*)string_language_desc, MIN(sizeof(string_language_desc), setup.b.wLength.w), 0);
              break;
            case STRING_OFFSET_IMANUFACTURER:
              USB_WritePacket((uint8_t*)string_manufacturer_desc, MIN(sizeof(string_manufacturer_desc), setup.b.wLength.w), 0);
              break;
            case STRING_OFFSET_IPRODUCT:
              USB_WritePacket((uint8_t*)string_product_desc, MIN(sizeof(string_product_desc), setup.b.wLength.w), 0);
              break;
            case STRING_OFFSET_ISERIAL:
              response[0] = 0x02 + (12 * 4);
              response[1] = 0x03;

              // 96 bits = 12 bytes
              for (int i = 0; i < 12; i++){
                uint8_t cc = ((uint8_t *)UID_BASE)[i];
                response[2 + (i * 4)] = to_hex_char((cc >> 4) & 0xFU);
                response[2 + (i * 4) + 1] = '\0';
                response[2 + (i * 4) + 2] = to_hex_char((cc >> 0) & 0xFU);
                response[2 + (i * 4) + 3] = '\0';
              }

              USB_WritePacket(response, MIN(response[0], setup.b.wLength.w), 0);
              break;
            case STRING_OFFSET_ICONFIGURATION:
              USB_WritePacket((uint8_t*)string_configuration_desc, MIN(sizeof(string_configuration_desc), setup.b.wLength.w), 0);
              break;
            case 238:
              USB_WritePacket((uint8_t*)string_238_desc, MIN(sizeof(string_238_desc), setup.b.wLength.w), 0);
              break;
            default:
              // nothing
              USB_WritePacket(0, 0, 0);
              break;
          }
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        case USB_DESC_TYPE_BINARY_OBJECT_STORE:
          USB_WritePacket(binary_object_store_desc, MIN(sizeof(binary_object_store_desc), setup.b.wLength.w), 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        default:
          // nothing here?
          USB_WritePacket(0, 0, 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
      }
      break;
    case USB_REQ_GET_STATUS:
      // empty response?
      response[0] = 0;
      response[1] = 0;
      USB_WritePacket((void*)&response, 2, 0);
      USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case USB_REQ_SET_INTERFACE:
      // Store the alt setting number for IN EP behavior.
      current_int0_alt_setting = setup.b.wValue.w;
      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case WEBUSB_VENDOR_CODE:
      switch (setup.b.wIndex.w) {
        case WEBUSB_REQ_GET_URL:
          USB_WritePacket(webusb_url_descriptor, MIN(sizeof(webusb_url_descriptor), setup.b.wLength.w), 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        default:
          // probably asking for allowed origins, which was removed from the spec
          USB_WritePacket(0, 0, 0);
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
      }
      break;
    case MS_VENDOR_CODE:
      switch (setup.b.wIndex.w) {
        // winusb 2.0 descriptor from BOS
        case WINUSB_REQ_GET_DESCRIPTOR:
          USB_WritePacket_EP0((uint8_t*)winusb_20_desc, MIN(sizeof(winusb_20_desc), setup.b.wLength.w));
          break;
        // Extended Compat ID OS Descriptor
        case WINUSB_REQ_GET_COMPATID_DESCRIPTOR:
          USB_WritePacket_EP0((uint8_t*)winusb_ext_compatid_os_desc, MIN(sizeof(winusb_ext_compatid_os_desc), setup.b.wLength.w));
          break;
        // Extended Properties OS Descriptor
        case WINUSB_REQ_GET_EXT_PROPS_OS:
          USB_WritePacket_EP0((uint8_t*)winusb_ext_prop_os_desc, MIN(sizeof(winusb_ext_prop_os_desc), setup.b.wLength.w));
          break;
        default:
          USB_WritePacket_EP0(0, 0);
      }
      break;
    default:
      control_req.request = setup.b.bRequest;
      control_req.param1 = setup.b.wValue.w;
      control_req.param2 = setup.b.wIndex.w;
      control_req.length = setup.b.wLength.w;

      resp_len = comms_control_handler(&control_req, response);
      // response pending if -1 was returned
      if (resp_len != -1) {
        USB_WritePacket(response, MIN(resp_len, setup.b.wLength.w), 0);
        USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      }
  }
}



// ***************************** USB port *****************************

void usb_irqhandler(void) {
  //USBx->GINTMSK = 0;

  unsigned int gintsts = USBx->GINTSTS;
  unsigned int gotgint = USBx->GOTGINT;
  unsigned int daint = USBx_DEVICE->DAINT;

  // gintsts SUSPEND? 04008428
  #ifdef DEBUG_USB
    puth(gintsts);
    print(" ");
    /*puth(USBx->GCCFG);
    print(" ");*/
    puth(gotgint);
    print(" ep ");
    puth(daint);
    print(" USB interrupt!\n");
  #endif

  if ((gintsts & USB_OTG_GINTSTS_CIDSCHG) != 0U) {
    print("connector ID status change\n");
  }

  if ((gintsts & USB_OTG_GINTSTS_USBRST) != 0U) {
    print("USB reset\n");
    usb_reset();
  }

  if ((gintsts & USB_OTG_GINTSTS_ENUMDNE) != 0U) {
    print("enumeration done");
    // Full speed, ENUMSPD
    //puth(USBx_DEVICE->DSTS);
    print("\n");
  }

  if ((gintsts & USB_OTG_GINTSTS_OTGINT) != 0U) {
    print("OTG int:");
    puth(USBx->GOTGINT);
    print("\n");

    // getting ADTOCHG
    //USBx->GOTGINT = USBx->GOTGINT;
  }

  // RX FIFO first
  if ((gintsts & USB_OTG_GINTSTS_RXFLVL) != 0U) {
    // 1. Read the Receive status pop register
    volatile unsigned int rxst = USBx->GRXSTSP;
    int status = (rxst & USB_OTG_GRXSTSP_PKTSTS) >> 17;

    #ifdef DEBUG_USB
      print(" RX FIFO:");
      puth(rxst);
      print(" status: ");
      puth(status);
      print(" len: ");
      puth((rxst & USB_OTG_GRXSTSP_BCNT) >> 4);
      print("\n");
    #endif

    if (status == STS_DATA_UPDT) {
      int endpoint = (rxst & USB_OTG_GRXSTSP_EPNUM);
      int len = (rxst & USB_OTG_GRXSTSP_BCNT) >> 4;
      (void)USB_ReadPacket(&usbdata, len);
      #ifdef DEBUG_USB
        print("  data ");
        puth(len);
        print("\n");
        hexdump(&usbdata, len);
      #endif

      if (endpoint == 2) {
        comms_endpoint2_write((uint8_t *) usbdata, len);
      }

      if (endpoint == 3) {
        outep3_processing = true;
        comms_can_write(usbdata, len);
      }
    } else if (status == STS_SETUP_UPDT) {
      (void)USB_ReadPacket(&setup, 8);
      #ifdef DEBUG_USB
        print("  setup ");
        hexdump(&setup, 8);
        print("\n");
      #endif
    } else {
      // status is neither STS_DATA_UPDT or STS_SETUP_UPDT, skip
    }
  }

  /*if (gintsts & USB_OTG_GINTSTS_HPRTINT) {
    // host
    print("HPRT:");
    puth(USBx_HOST_PORT->HPRT);
    print("\n");
    if (USBx_HOST_PORT->HPRT & USB_OTG_HPRT_PCDET) {
      USBx_HOST_PORT->HPRT |= USB_OTG_HPRT_PRST;
      USBx_HOST_PORT->HPRT |= USB_OTG_HPRT_PCDET;
    }

  }*/

  if ((gintsts & USB_OTG_GINTSTS_BOUTNAKEFF) || (gintsts & USB_OTG_GINTSTS_GINAKEFF)) {
    // no global NAK, why is this getting set?
    #ifdef DEBUG_USB
      print("GLOBAL NAK\n");
    #endif
    USBx_DEVICE->DCTL |= USB_OTG_DCTL_CGONAK | USB_OTG_DCTL_CGINAK;
  }

  if ((gintsts & USB_OTG_GINTSTS_SRQINT) != 0U) {
    // we want to do "A-device host negotiation protocol" since we are the A-device
    /*print("start request\n");
    puth(USBx->GOTGCTL);
    print("\n");*/
    //USBx->GUSBCFG |= USB_OTG_GUSBCFG_FDMOD;
    //USBx_HOST_PORT->HPRT = USB_OTG_HPRT_PPWR | USB_OTG_HPRT_PENA;
    //USBx->GOTGCTL |= USB_OTG_GOTGCTL_SRQ;
  }

  // out endpoint hit
  if ((gintsts & USB_OTG_GINTSTS_OEPINT) != 0U) {
    #ifdef DEBUG_USB
      print("  0:");
      puth(USBx_OUTEP(0U)->DOEPINT);
      print(" 2:");
      puth(USBx_OUTEP(2U)->DOEPINT);
      print(" 3:");
      puth(USBx_OUTEP(3U)->DOEPINT);
      print(" ");
      puth(USBx_OUTEP(3U)->DOEPCTL);
      print(" 4:");
      puth(USBx_OUTEP(4)->DOEPINT);
      print(" OUT ENDPOINT\n");
    #endif

    if ((USBx_OUTEP(2U)->DOEPINT & USB_OTG_DOEPINT_XFRC) != 0U) {
      #ifdef DEBUG_USB
        print("  OUT2 PACKET XFRC\n");
      #endif
      USBx_OUTEP(2U)->DOEPTSIZ = (1UL << 19) | 0x40U;
      USBx_OUTEP(2U)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
    }

    if ((USBx_OUTEP(3U)->DOEPINT & USB_OTG_DOEPINT_XFRC) != 0U) {
      #ifdef DEBUG_USB
        print("  OUT3 PACKET XFRC\n");
      #endif
      // NAK cleared by process_can (if tx buffers have room)
      outep3_processing = false;
      refresh_can_tx_slots_available();
    } else if ((USBx_OUTEP(3U)->DOEPINT & 0x2000U) != 0U) {
      #ifdef DEBUG_USB
        print("  OUT3 PACKET WTF\n");
      #endif
      // if NAK was set trigger this, unknown interrupt
      // TODO: why was this here? fires when TX buffers when we can't clear NAK
      // USBx_OUTEP(3U)->DOEPTSIZ = (1U << 19) | 0x40U;
      // USBx_OUTEP(3U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
    } else if ((USBx_OUTEP(3U)->DOEPINT) != 0U) {
      #ifdef DEBUG_USB
        print("OUTEP3 error ");
        puth(USBx_OUTEP(3U)->DOEPINT);
        print("\n");
      #endif
    } else {
      // USBx_OUTEP(3U)->DOEPINT is 0, ok to skip
    }

    if ((USBx_OUTEP(0U)->DOEPINT & USB_OTG_DIEPINT_XFRC) != 0U) {
      // ready for next packet
      USBx_OUTEP(0U)->DOEPTSIZ = USB_OTG_DOEPTSIZ_STUPCNT | (USB_OTG_DOEPTSIZ_PKTCNT & (1UL << 19)) | (1U << 3);
    }

    // respond to setup packets
    if ((USBx_OUTEP(0U)->DOEPINT & USB_OTG_DOEPINT_STUP) != 0U) {
      usb_setup();
    }

    USBx_OUTEP(0U)->DOEPINT = USBx_OUTEP(0U)->DOEPINT;
    USBx_OUTEP(2U)->DOEPINT = USBx_OUTEP(2U)->DOEPINT;
    USBx_OUTEP(3U)->DOEPINT = USBx_OUTEP(3U)->DOEPINT;
  }

  // interrupt endpoint hit (Page 1221)
  if ((gintsts & USB_OTG_GINTSTS_IEPINT) != 0U) {
    #ifdef DEBUG_USB
      print("  ");
      puth(USBx_INEP(0U)->DIEPINT);
      print(" ");
      puth(USBx_INEP(1U)->DIEPINT);
      print(" IN ENDPOINT\n");
    #endif

    // Should likely check the EP of the IN request even if there is
    // only one IN endpoint.

    // No need to set NAK in OTG_DIEPCTL0 when nothing to send,
    // Appears USB core automatically sets NAK. WritePacket clears it.

    // Handle the two interface alternate settings. Setting 0 has EP1
    // as bulk. Setting 1 has EP1 as interrupt. The code to handle
    // these two EP variations are very similar and can be
    // restructured for smaller code footprint. Keeping split out for
    // now for clarity.

    //TODO add default case. Should it NAK?
    switch (current_int0_alt_setting) {
      case 0: ////// Bulk config
        // *** IN token received when TxFIFO is empty
        if ((USBx_INEP(1U)->DIEPINT & USB_OTG_DIEPMSK_ITTXFEMSK) != 0U) {
          #ifdef DEBUG_USB
          print("  IN PACKET QUEUE\n");
          #endif
          // TODO: always assuming max len, can we get the length?
          USB_WritePacket((void *)response, comms_can_read(response, 0x40), 1);
        }
        break;

      case 1: ////// Interrupt config
        // *** IN token received when TxFIFO is empty
        if ((USBx_INEP(1U)->DIEPINT & USB_OTG_DIEPMSK_ITTXFEMSK) != 0U) {
          #ifdef DEBUG_USB
          print("  IN PACKET QUEUE\n");
          #endif
          // TODO: always assuming max len, can we get the length?
          int len = comms_can_read(response, 0x40);
          if (len > 0) {
            USB_WritePacket((void *)response, len, 1);
          }
        }
        break;
      default:
        print("current_int0_alt_setting value invalid\n");
        break;
    }

    if ((USBx_INEP(0U)->DIEPINT & USB_OTG_DIEPMSK_ITTXFEMSK) != 0U) {
      #ifdef DEBUG_USB
      print("  IN PACKET QUEUE\n");
      #endif

      if ((ep0_txlen != 0U) && ((USBx_INEP(0U)->DTXFSTS & USB_OTG_DTXFSTS_INEPTFSAV) >= 0x40U)) {
        uint16_t len = MIN(ep0_txlen, 0x40);
        USB_WritePacket(ep0_txdata, len, 0);
        ep0_txdata = &ep0_txdata[len];
        ep0_txlen -= len;
        if (ep0_txlen == 0U) {
          ep0_txdata = NULL;
          USBx_DEVICE->DIEPEMPMSK &= ~1;
          USBx_OUTEP(0U)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
        }
      }
    }

    // clear interrupts
    USBx_INEP(0U)->DIEPINT = USBx_INEP(0U)->DIEPINT; // Why ep0?
    USBx_INEP(1U)->DIEPINT = USBx_INEP(1U)->DIEPINT;
  }

  // clear all interrupts we handled
  USBx_DEVICE->DAINT = daint;
  USBx->GOTGINT = gotgint;
  USBx->GINTSTS = gintsts;

  //USBx->GINTMSK = 0xFFFFFFFF & ~(USB_OTG_GINTMSK_NPTXFEM | USB_OTG_GINTMSK_PTXFEM | USB_OTG_GINTSTS_SOF | USB_OTG_GINTSTS_EOPF);
}

void can_tx_comms_resume_usb(void) {
  ENTER_CRITICAL();
  if (!outep3_processing && (USBx_OUTEP(3U)->DOEPCTL & USB_OTG_DOEPCTL_NAKSTS) != 0U) {
    USBx_OUTEP(3U)->DOEPTSIZ = (32UL << 19) | 0x800U;
    USBx_OUTEP(3U)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
  }
  EXIT_CRITICAL();
}

void usb_soft_disconnect(bool enable) {
  if (enable) {
    USBx_DEVICE->DCTL |= USB_OTG_DCTL_SDIS;
  } else {
    USBx_DEVICE->DCTL &= ~USB_OTG_DCTL_SDIS;
  }
}
