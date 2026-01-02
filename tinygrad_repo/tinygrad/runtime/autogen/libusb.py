# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('libusb', 'usb-1.0')
enum_libusb_class_code = CEnum(ctypes.c_uint32)
LIBUSB_CLASS_PER_INTERFACE = enum_libusb_class_code.define('LIBUSB_CLASS_PER_INTERFACE', 0)
LIBUSB_CLASS_AUDIO = enum_libusb_class_code.define('LIBUSB_CLASS_AUDIO', 1)
LIBUSB_CLASS_COMM = enum_libusb_class_code.define('LIBUSB_CLASS_COMM', 2)
LIBUSB_CLASS_HID = enum_libusb_class_code.define('LIBUSB_CLASS_HID', 3)
LIBUSB_CLASS_PHYSICAL = enum_libusb_class_code.define('LIBUSB_CLASS_PHYSICAL', 5)
LIBUSB_CLASS_IMAGE = enum_libusb_class_code.define('LIBUSB_CLASS_IMAGE', 6)
LIBUSB_CLASS_PTP = enum_libusb_class_code.define('LIBUSB_CLASS_PTP', 6)
LIBUSB_CLASS_PRINTER = enum_libusb_class_code.define('LIBUSB_CLASS_PRINTER', 7)
LIBUSB_CLASS_MASS_STORAGE = enum_libusb_class_code.define('LIBUSB_CLASS_MASS_STORAGE', 8)
LIBUSB_CLASS_HUB = enum_libusb_class_code.define('LIBUSB_CLASS_HUB', 9)
LIBUSB_CLASS_DATA = enum_libusb_class_code.define('LIBUSB_CLASS_DATA', 10)
LIBUSB_CLASS_SMART_CARD = enum_libusb_class_code.define('LIBUSB_CLASS_SMART_CARD', 11)
LIBUSB_CLASS_CONTENT_SECURITY = enum_libusb_class_code.define('LIBUSB_CLASS_CONTENT_SECURITY', 13)
LIBUSB_CLASS_VIDEO = enum_libusb_class_code.define('LIBUSB_CLASS_VIDEO', 14)
LIBUSB_CLASS_PERSONAL_HEALTHCARE = enum_libusb_class_code.define('LIBUSB_CLASS_PERSONAL_HEALTHCARE', 15)
LIBUSB_CLASS_DIAGNOSTIC_DEVICE = enum_libusb_class_code.define('LIBUSB_CLASS_DIAGNOSTIC_DEVICE', 220)
LIBUSB_CLASS_WIRELESS = enum_libusb_class_code.define('LIBUSB_CLASS_WIRELESS', 224)
LIBUSB_CLASS_MISCELLANEOUS = enum_libusb_class_code.define('LIBUSB_CLASS_MISCELLANEOUS', 239)
LIBUSB_CLASS_APPLICATION = enum_libusb_class_code.define('LIBUSB_CLASS_APPLICATION', 254)
LIBUSB_CLASS_VENDOR_SPEC = enum_libusb_class_code.define('LIBUSB_CLASS_VENDOR_SPEC', 255)

enum_libusb_descriptor_type = CEnum(ctypes.c_uint32)
LIBUSB_DT_DEVICE = enum_libusb_descriptor_type.define('LIBUSB_DT_DEVICE', 1)
LIBUSB_DT_CONFIG = enum_libusb_descriptor_type.define('LIBUSB_DT_CONFIG', 2)
LIBUSB_DT_STRING = enum_libusb_descriptor_type.define('LIBUSB_DT_STRING', 3)
LIBUSB_DT_INTERFACE = enum_libusb_descriptor_type.define('LIBUSB_DT_INTERFACE', 4)
LIBUSB_DT_ENDPOINT = enum_libusb_descriptor_type.define('LIBUSB_DT_ENDPOINT', 5)
LIBUSB_DT_INTERFACE_ASSOCIATION = enum_libusb_descriptor_type.define('LIBUSB_DT_INTERFACE_ASSOCIATION', 11)
LIBUSB_DT_BOS = enum_libusb_descriptor_type.define('LIBUSB_DT_BOS', 15)
LIBUSB_DT_DEVICE_CAPABILITY = enum_libusb_descriptor_type.define('LIBUSB_DT_DEVICE_CAPABILITY', 16)
LIBUSB_DT_HID = enum_libusb_descriptor_type.define('LIBUSB_DT_HID', 33)
LIBUSB_DT_REPORT = enum_libusb_descriptor_type.define('LIBUSB_DT_REPORT', 34)
LIBUSB_DT_PHYSICAL = enum_libusb_descriptor_type.define('LIBUSB_DT_PHYSICAL', 35)
LIBUSB_DT_HUB = enum_libusb_descriptor_type.define('LIBUSB_DT_HUB', 41)
LIBUSB_DT_SUPERSPEED_HUB = enum_libusb_descriptor_type.define('LIBUSB_DT_SUPERSPEED_HUB', 42)
LIBUSB_DT_SS_ENDPOINT_COMPANION = enum_libusb_descriptor_type.define('LIBUSB_DT_SS_ENDPOINT_COMPANION', 48)

enum_libusb_endpoint_direction = CEnum(ctypes.c_uint32)
LIBUSB_ENDPOINT_OUT = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_OUT', 0)
LIBUSB_ENDPOINT_IN = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_IN', 128)

enum_libusb_endpoint_transfer_type = CEnum(ctypes.c_uint32)
LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK', 2)
LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT', 3)

enum_libusb_standard_request = CEnum(ctypes.c_uint32)
LIBUSB_REQUEST_GET_STATUS = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_STATUS', 0)
LIBUSB_REQUEST_CLEAR_FEATURE = enum_libusb_standard_request.define('LIBUSB_REQUEST_CLEAR_FEATURE', 1)
LIBUSB_REQUEST_SET_FEATURE = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_FEATURE', 3)
LIBUSB_REQUEST_SET_ADDRESS = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_ADDRESS', 5)
LIBUSB_REQUEST_GET_DESCRIPTOR = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_DESCRIPTOR', 6)
LIBUSB_REQUEST_SET_DESCRIPTOR = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_DESCRIPTOR', 7)
LIBUSB_REQUEST_GET_CONFIGURATION = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_CONFIGURATION', 8)
LIBUSB_REQUEST_SET_CONFIGURATION = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_CONFIGURATION', 9)
LIBUSB_REQUEST_GET_INTERFACE = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_INTERFACE', 10)
LIBUSB_REQUEST_SET_INTERFACE = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_INTERFACE', 11)
LIBUSB_REQUEST_SYNCH_FRAME = enum_libusb_standard_request.define('LIBUSB_REQUEST_SYNCH_FRAME', 12)
LIBUSB_REQUEST_SET_SEL = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_SEL', 48)
LIBUSB_SET_ISOCH_DELAY = enum_libusb_standard_request.define('LIBUSB_SET_ISOCH_DELAY', 49)

enum_libusb_request_type = CEnum(ctypes.c_uint32)
LIBUSB_REQUEST_TYPE_STANDARD = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_STANDARD', 0)
LIBUSB_REQUEST_TYPE_CLASS = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_CLASS', 32)
LIBUSB_REQUEST_TYPE_VENDOR = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_VENDOR', 64)
LIBUSB_REQUEST_TYPE_RESERVED = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_RESERVED', 96)

enum_libusb_request_recipient = CEnum(ctypes.c_uint32)
LIBUSB_RECIPIENT_DEVICE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_DEVICE', 0)
LIBUSB_RECIPIENT_INTERFACE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_INTERFACE', 1)
LIBUSB_RECIPIENT_ENDPOINT = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_ENDPOINT', 2)
LIBUSB_RECIPIENT_OTHER = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_OTHER', 3)

enum_libusb_iso_sync_type = CEnum(ctypes.c_uint32)
LIBUSB_ISO_SYNC_TYPE_NONE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_NONE', 0)
LIBUSB_ISO_SYNC_TYPE_ASYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ASYNC', 1)
LIBUSB_ISO_SYNC_TYPE_ADAPTIVE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ADAPTIVE', 2)
LIBUSB_ISO_SYNC_TYPE_SYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_SYNC', 3)

enum_libusb_iso_usage_type = CEnum(ctypes.c_uint32)
LIBUSB_ISO_USAGE_TYPE_DATA = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_DATA', 0)
LIBUSB_ISO_USAGE_TYPE_FEEDBACK = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_FEEDBACK', 1)
LIBUSB_ISO_USAGE_TYPE_IMPLICIT = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_IMPLICIT', 2)

enum_libusb_supported_speed = CEnum(ctypes.c_uint32)
LIBUSB_LOW_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_LOW_SPEED_OPERATION', 1)
LIBUSB_FULL_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_FULL_SPEED_OPERATION', 2)
LIBUSB_HIGH_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_HIGH_SPEED_OPERATION', 4)
LIBUSB_SUPER_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_SUPER_SPEED_OPERATION', 8)

enum_libusb_usb_2_0_extension_attributes = CEnum(ctypes.c_uint32)
LIBUSB_BM_LPM_SUPPORT = enum_libusb_usb_2_0_extension_attributes.define('LIBUSB_BM_LPM_SUPPORT', 2)

enum_libusb_ss_usb_device_capability_attributes = CEnum(ctypes.c_uint32)
LIBUSB_BM_LTM_SUPPORT = enum_libusb_ss_usb_device_capability_attributes.define('LIBUSB_BM_LTM_SUPPORT', 2)

enum_libusb_bos_type = CEnum(ctypes.c_uint32)
LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY', 1)
LIBUSB_BT_USB_2_0_EXTENSION = enum_libusb_bos_type.define('LIBUSB_BT_USB_2_0_EXTENSION', 2)
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_SS_USB_DEVICE_CAPABILITY', 3)
LIBUSB_BT_CONTAINER_ID = enum_libusb_bos_type.define('LIBUSB_BT_CONTAINER_ID', 4)
LIBUSB_BT_PLATFORM_DESCRIPTOR = enum_libusb_bos_type.define('LIBUSB_BT_PLATFORM_DESCRIPTOR', 5)

class struct_libusb_device_descriptor(Struct): pass
uint8_t = ctypes.c_ubyte
uint16_t = ctypes.c_uint16
struct_libusb_device_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bcdUSB', uint16_t),
  ('bDeviceClass', uint8_t),
  ('bDeviceSubClass', uint8_t),
  ('bDeviceProtocol', uint8_t),
  ('bMaxPacketSize0', uint8_t),
  ('idVendor', uint16_t),
  ('idProduct', uint16_t),
  ('bcdDevice', uint16_t),
  ('iManufacturer', uint8_t),
  ('iProduct', uint8_t),
  ('iSerialNumber', uint8_t),
  ('bNumConfigurations', uint8_t),
]
class struct_libusb_endpoint_descriptor(Struct): pass
struct_libusb_endpoint_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bEndpointAddress', uint8_t),
  ('bmAttributes', uint8_t),
  ('wMaxPacketSize', uint16_t),
  ('bInterval', uint8_t),
  ('bRefresh', uint8_t),
  ('bSynchAddress', uint8_t),
  ('extra', ctypes.POINTER(ctypes.c_ubyte)),
  ('extra_length', ctypes.c_int32),
]
class struct_libusb_interface_association_descriptor(Struct): pass
struct_libusb_interface_association_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bFirstInterface', uint8_t),
  ('bInterfaceCount', uint8_t),
  ('bFunctionClass', uint8_t),
  ('bFunctionSubClass', uint8_t),
  ('bFunctionProtocol', uint8_t),
  ('iFunction', uint8_t),
]
class struct_libusb_interface_association_descriptor_array(Struct): pass
struct_libusb_interface_association_descriptor_array._fields_ = [
  ('iad', ctypes.POINTER(struct_libusb_interface_association_descriptor)),
  ('length', ctypes.c_int32),
]
class struct_libusb_interface_descriptor(Struct): pass
struct_libusb_interface_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bInterfaceNumber', uint8_t),
  ('bAlternateSetting', uint8_t),
  ('bNumEndpoints', uint8_t),
  ('bInterfaceClass', uint8_t),
  ('bInterfaceSubClass', uint8_t),
  ('bInterfaceProtocol', uint8_t),
  ('iInterface', uint8_t),
  ('endpoint', ctypes.POINTER(struct_libusb_endpoint_descriptor)),
  ('extra', ctypes.POINTER(ctypes.c_ubyte)),
  ('extra_length', ctypes.c_int32),
]
class struct_libusb_interface(Struct): pass
struct_libusb_interface._fields_ = [
  ('altsetting', ctypes.POINTER(struct_libusb_interface_descriptor)),
  ('num_altsetting', ctypes.c_int32),
]
class struct_libusb_config_descriptor(Struct): pass
struct_libusb_config_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('wTotalLength', uint16_t),
  ('bNumInterfaces', uint8_t),
  ('bConfigurationValue', uint8_t),
  ('iConfiguration', uint8_t),
  ('bmAttributes', uint8_t),
  ('MaxPower', uint8_t),
  ('interface', ctypes.POINTER(struct_libusb_interface)),
  ('extra', ctypes.POINTER(ctypes.c_ubyte)),
  ('extra_length', ctypes.c_int32),
]
class struct_libusb_ss_endpoint_companion_descriptor(Struct): pass
struct_libusb_ss_endpoint_companion_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bMaxBurst', uint8_t),
  ('bmAttributes', uint8_t),
  ('wBytesPerInterval', uint16_t),
]
class struct_libusb_bos_dev_capability_descriptor(Struct): pass
struct_libusb_bos_dev_capability_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bDevCapabilityType', uint8_t),
  ('dev_capability_data', (uint8_t * 0)),
]
class struct_libusb_bos_descriptor(Struct): pass
struct_libusb_bos_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('wTotalLength', uint16_t),
  ('bNumDeviceCaps', uint8_t),
  ('dev_capability', (ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor) * 0)),
]
class struct_libusb_usb_2_0_extension_descriptor(Struct): pass
uint32_t = ctypes.c_uint32
struct_libusb_usb_2_0_extension_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bDevCapabilityType', uint8_t),
  ('bmAttributes', uint32_t),
]
class struct_libusb_ss_usb_device_capability_descriptor(Struct): pass
struct_libusb_ss_usb_device_capability_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bDevCapabilityType', uint8_t),
  ('bmAttributes', uint8_t),
  ('wSpeedSupported', uint16_t),
  ('bFunctionalitySupport', uint8_t),
  ('bU1DevExitLat', uint8_t),
  ('bU2DevExitLat', uint16_t),
]
class struct_libusb_container_id_descriptor(Struct): pass
struct_libusb_container_id_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bDevCapabilityType', uint8_t),
  ('bReserved', uint8_t),
  ('ContainerID', (uint8_t * 16)),
]
class struct_libusb_platform_descriptor(Struct): pass
struct_libusb_platform_descriptor._fields_ = [
  ('bLength', uint8_t),
  ('bDescriptorType', uint8_t),
  ('bDevCapabilityType', uint8_t),
  ('bReserved', uint8_t),
  ('PlatformCapabilityUUID', (uint8_t * 16)),
  ('CapabilityData', (uint8_t * 0)),
]
class struct_libusb_control_setup(Struct): pass
struct_libusb_control_setup._packed_ = True
struct_libusb_control_setup._fields_ = [
  ('bmRequestType', uint8_t),
  ('bRequest', uint8_t),
  ('wValue', uint16_t),
  ('wIndex', uint16_t),
  ('wLength', uint16_t),
]
class struct_libusb_context(Struct): pass
class struct_libusb_device(Struct): pass
class struct_libusb_device_handle(Struct): pass
class struct_libusb_version(Struct): pass
struct_libusb_version._fields_ = [
  ('major', uint16_t),
  ('minor', uint16_t),
  ('micro', uint16_t),
  ('nano', uint16_t),
  ('rc', ctypes.POINTER(ctypes.c_char)),
  ('describe', ctypes.POINTER(ctypes.c_char)),
]
libusb_context = struct_libusb_context
libusb_device = struct_libusb_device
libusb_device_handle = struct_libusb_device_handle
enum_libusb_speed = CEnum(ctypes.c_uint32)
LIBUSB_SPEED_UNKNOWN = enum_libusb_speed.define('LIBUSB_SPEED_UNKNOWN', 0)
LIBUSB_SPEED_LOW = enum_libusb_speed.define('LIBUSB_SPEED_LOW', 1)
LIBUSB_SPEED_FULL = enum_libusb_speed.define('LIBUSB_SPEED_FULL', 2)
LIBUSB_SPEED_HIGH = enum_libusb_speed.define('LIBUSB_SPEED_HIGH', 3)
LIBUSB_SPEED_SUPER = enum_libusb_speed.define('LIBUSB_SPEED_SUPER', 4)
LIBUSB_SPEED_SUPER_PLUS = enum_libusb_speed.define('LIBUSB_SPEED_SUPER_PLUS', 5)

enum_libusb_error = CEnum(ctypes.c_int32)
LIBUSB_SUCCESS = enum_libusb_error.define('LIBUSB_SUCCESS', 0)
LIBUSB_ERROR_IO = enum_libusb_error.define('LIBUSB_ERROR_IO', -1)
LIBUSB_ERROR_INVALID_PARAM = enum_libusb_error.define('LIBUSB_ERROR_INVALID_PARAM', -2)
LIBUSB_ERROR_ACCESS = enum_libusb_error.define('LIBUSB_ERROR_ACCESS', -3)
LIBUSB_ERROR_NO_DEVICE = enum_libusb_error.define('LIBUSB_ERROR_NO_DEVICE', -4)
LIBUSB_ERROR_NOT_FOUND = enum_libusb_error.define('LIBUSB_ERROR_NOT_FOUND', -5)
LIBUSB_ERROR_BUSY = enum_libusb_error.define('LIBUSB_ERROR_BUSY', -6)
LIBUSB_ERROR_TIMEOUT = enum_libusb_error.define('LIBUSB_ERROR_TIMEOUT', -7)
LIBUSB_ERROR_OVERFLOW = enum_libusb_error.define('LIBUSB_ERROR_OVERFLOW', -8)
LIBUSB_ERROR_PIPE = enum_libusb_error.define('LIBUSB_ERROR_PIPE', -9)
LIBUSB_ERROR_INTERRUPTED = enum_libusb_error.define('LIBUSB_ERROR_INTERRUPTED', -10)
LIBUSB_ERROR_NO_MEM = enum_libusb_error.define('LIBUSB_ERROR_NO_MEM', -11)
LIBUSB_ERROR_NOT_SUPPORTED = enum_libusb_error.define('LIBUSB_ERROR_NOT_SUPPORTED', -12)
LIBUSB_ERROR_OTHER = enum_libusb_error.define('LIBUSB_ERROR_OTHER', -99)

enum_libusb_transfer_type = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_TYPE_CONTROL = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_TRANSFER_TYPE_BULK = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK', 2)
LIBUSB_TRANSFER_TYPE_INTERRUPT = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_INTERRUPT', 3)
LIBUSB_TRANSFER_TYPE_BULK_STREAM = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK_STREAM', 4)

enum_libusb_transfer_status = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_COMPLETED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_COMPLETED', 0)
LIBUSB_TRANSFER_ERROR = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_ERROR', 1)
LIBUSB_TRANSFER_TIMED_OUT = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_TIMED_OUT', 2)
LIBUSB_TRANSFER_CANCELLED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_CANCELLED', 3)
LIBUSB_TRANSFER_STALL = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_STALL', 4)
LIBUSB_TRANSFER_NO_DEVICE = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_NO_DEVICE', 5)
LIBUSB_TRANSFER_OVERFLOW = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_OVERFLOW', 6)

enum_libusb_transfer_flags = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_SHORT_NOT_OK = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_SHORT_NOT_OK', 1)
LIBUSB_TRANSFER_FREE_BUFFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_BUFFER', 2)
LIBUSB_TRANSFER_FREE_TRANSFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_TRANSFER', 4)
LIBUSB_TRANSFER_ADD_ZERO_PACKET = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_ADD_ZERO_PACKET', 8)

class struct_libusb_iso_packet_descriptor(Struct): pass
struct_libusb_iso_packet_descriptor._fields_ = [
  ('length', ctypes.c_uint32),
  ('actual_length', ctypes.c_uint32),
  ('status', enum_libusb_transfer_status),
]
class struct_libusb_transfer(Struct): pass
libusb_transfer_cb_fn = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_transfer))
struct_libusb_transfer._fields_ = [
  ('dev_handle', ctypes.POINTER(libusb_device_handle)),
  ('flags', uint8_t),
  ('endpoint', ctypes.c_ubyte),
  ('type', ctypes.c_ubyte),
  ('timeout', ctypes.c_uint32),
  ('status', enum_libusb_transfer_status),
  ('length', ctypes.c_int32),
  ('actual_length', ctypes.c_int32),
  ('callback', libusb_transfer_cb_fn),
  ('user_data', ctypes.c_void_p),
  ('buffer', ctypes.POINTER(ctypes.c_ubyte)),
  ('num_iso_packets', ctypes.c_int32),
  ('iso_packet_desc', (struct_libusb_iso_packet_descriptor * 0)),
]
enum_libusb_capability = CEnum(ctypes.c_uint32)
LIBUSB_CAP_HAS_CAPABILITY = enum_libusb_capability.define('LIBUSB_CAP_HAS_CAPABILITY', 0)
LIBUSB_CAP_HAS_HOTPLUG = enum_libusb_capability.define('LIBUSB_CAP_HAS_HOTPLUG', 1)
LIBUSB_CAP_HAS_HID_ACCESS = enum_libusb_capability.define('LIBUSB_CAP_HAS_HID_ACCESS', 256)
LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER = enum_libusb_capability.define('LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER', 257)

enum_libusb_log_level = CEnum(ctypes.c_uint32)
LIBUSB_LOG_LEVEL_NONE = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_NONE', 0)
LIBUSB_LOG_LEVEL_ERROR = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_ERROR', 1)
LIBUSB_LOG_LEVEL_WARNING = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_WARNING', 2)
LIBUSB_LOG_LEVEL_INFO = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_INFO', 3)
LIBUSB_LOG_LEVEL_DEBUG = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_DEBUG', 4)

enum_libusb_log_cb_mode = CEnum(ctypes.c_uint32)
LIBUSB_LOG_CB_GLOBAL = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_GLOBAL', 1)
LIBUSB_LOG_CB_CONTEXT = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_CONTEXT', 2)

enum_libusb_option = CEnum(ctypes.c_uint32)
LIBUSB_OPTION_LOG_LEVEL = enum_libusb_option.define('LIBUSB_OPTION_LOG_LEVEL', 0)
LIBUSB_OPTION_USE_USBDK = enum_libusb_option.define('LIBUSB_OPTION_USE_USBDK', 1)
LIBUSB_OPTION_NO_DEVICE_DISCOVERY = enum_libusb_option.define('LIBUSB_OPTION_NO_DEVICE_DISCOVERY', 2)
LIBUSB_OPTION_LOG_CB = enum_libusb_option.define('LIBUSB_OPTION_LOG_CB', 3)
LIBUSB_OPTION_MAX = enum_libusb_option.define('LIBUSB_OPTION_MAX', 4)

libusb_log_cb = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_context), enum_libusb_log_level, ctypes.POINTER(ctypes.c_char))
class struct_libusb_init_option(Struct): pass
class struct_libusb_init_option_value(ctypes.Union): pass
struct_libusb_init_option_value._fields_ = [
  ('ival', ctypes.c_int32),
  ('log_cbval', libusb_log_cb),
]
struct_libusb_init_option._fields_ = [
  ('option', enum_libusb_option),
  ('value', struct_libusb_init_option_value),
]
try: (libusb_init:=dll.libusb_init).restype, libusb_init.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.POINTER(libusb_context))]
except AttributeError: pass

try: (libusb_init_context:=dll.libusb_init_context).restype, libusb_init_context.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.POINTER(libusb_context)), (struct_libusb_init_option * 0), ctypes.c_int32]
except AttributeError: pass

try: (libusb_exit:=dll.libusb_exit).restype, libusb_exit.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_set_debug:=dll.libusb_set_debug).restype, libusb_set_debug.argtypes = None, [ctypes.POINTER(libusb_context), ctypes.c_int32]
except AttributeError: pass

try: (libusb_set_log_cb:=dll.libusb_set_log_cb).restype, libusb_set_log_cb.argtypes = None, [ctypes.POINTER(libusb_context), libusb_log_cb, ctypes.c_int32]
except AttributeError: pass

try: (libusb_get_version:=dll.libusb_get_version).restype, libusb_get_version.argtypes = ctypes.POINTER(struct_libusb_version), []
except AttributeError: pass

try: (libusb_has_capability:=dll.libusb_has_capability).restype, libusb_has_capability.argtypes = ctypes.c_int32, [uint32_t]
except AttributeError: pass

try: (libusb_error_name:=dll.libusb_error_name).restype, libusb_error_name.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32]
except AttributeError: pass

try: (libusb_setlocale:=dll.libusb_setlocale).restype, libusb_setlocale.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (libusb_strerror:=dll.libusb_strerror).restype, libusb_strerror.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32]
except AttributeError: pass

ssize_t = ctypes.c_int64
try: (libusb_get_device_list:=dll.libusb_get_device_list).restype, libusb_get_device_list.argtypes = ssize_t, [ctypes.POINTER(libusb_context), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(libusb_device)))]
except AttributeError: pass

try: (libusb_free_device_list:=dll.libusb_free_device_list).restype, libusb_free_device_list.argtypes = None, [ctypes.POINTER(ctypes.POINTER(libusb_device)), ctypes.c_int32]
except AttributeError: pass

try: (libusb_ref_device:=dll.libusb_ref_device).restype, libusb_ref_device.argtypes = ctypes.POINTER(libusb_device), [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_unref_device:=dll.libusb_unref_device).restype, libusb_unref_device.argtypes = None, [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_configuration:=dll.libusb_get_configuration).restype, libusb_get_configuration.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (libusb_get_device_descriptor:=dll.libusb_get_device_descriptor).restype, libusb_get_device_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.POINTER(struct_libusb_device_descriptor)]
except AttributeError: pass

try: (libusb_get_active_config_descriptor:=dll.libusb_get_active_config_descriptor).restype, libusb_get_active_config_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError: pass

try: (libusb_get_config_descriptor:=dll.libusb_get_config_descriptor).restype, libusb_get_config_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError: pass

try: (libusb_get_config_descriptor_by_value:=dll.libusb_get_config_descriptor_by_value).restype, libusb_get_config_descriptor_by_value.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError: pass

try: (libusb_free_config_descriptor:=dll.libusb_free_config_descriptor).restype, libusb_free_config_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_config_descriptor)]
except AttributeError: pass

try: (libusb_get_ss_endpoint_companion_descriptor:=dll.libusb_get_ss_endpoint_companion_descriptor).restype, libusb_get_ss_endpoint_companion_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_endpoint_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor))]
except AttributeError: pass

try: (libusb_free_ss_endpoint_companion_descriptor:=dll.libusb_free_ss_endpoint_companion_descriptor).restype, libusb_free_ss_endpoint_companion_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor)]
except AttributeError: pass

try: (libusb_get_bos_descriptor:=dll.libusb_get_bos_descriptor).restype, libusb_get_bos_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.POINTER(struct_libusb_bos_descriptor))]
except AttributeError: pass

try: (libusb_free_bos_descriptor:=dll.libusb_free_bos_descriptor).restype, libusb_free_bos_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_bos_descriptor)]
except AttributeError: pass

try: (libusb_get_usb_2_0_extension_descriptor:=dll.libusb_get_usb_2_0_extension_descriptor).restype, libusb_get_usb_2_0_extension_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor))]
except AttributeError: pass

try: (libusb_free_usb_2_0_extension_descriptor:=dll.libusb_free_usb_2_0_extension_descriptor).restype, libusb_free_usb_2_0_extension_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor)]
except AttributeError: pass

try: (libusb_get_ss_usb_device_capability_descriptor:=dll.libusb_get_ss_usb_device_capability_descriptor).restype, libusb_get_ss_usb_device_capability_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor))]
except AttributeError: pass

try: (libusb_free_ss_usb_device_capability_descriptor:=dll.libusb_free_ss_usb_device_capability_descriptor).restype, libusb_free_ss_usb_device_capability_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor)]
except AttributeError: pass

try: (libusb_get_container_id_descriptor:=dll.libusb_get_container_id_descriptor).restype, libusb_get_container_id_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_container_id_descriptor))]
except AttributeError: pass

try: (libusb_free_container_id_descriptor:=dll.libusb_free_container_id_descriptor).restype, libusb_free_container_id_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_container_id_descriptor)]
except AttributeError: pass

try: (libusb_get_platform_descriptor:=dll.libusb_get_platform_descriptor).restype, libusb_get_platform_descriptor.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_platform_descriptor))]
except AttributeError: pass

try: (libusb_free_platform_descriptor:=dll.libusb_free_platform_descriptor).restype, libusb_free_platform_descriptor.argtypes = None, [ctypes.POINTER(struct_libusb_platform_descriptor)]
except AttributeError: pass

try: (libusb_get_bus_number:=dll.libusb_get_bus_number).restype, libusb_get_bus_number.argtypes = uint8_t, [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_port_number:=dll.libusb_get_port_number).restype, libusb_get_port_number.argtypes = uint8_t, [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_port_numbers:=dll.libusb_get_port_numbers).restype, libusb_get_port_numbers.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.POINTER(uint8_t), ctypes.c_int32]
except AttributeError: pass

try: (libusb_get_port_path:=dll.libusb_get_port_path).restype, libusb_get_port_path.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(libusb_device), ctypes.POINTER(uint8_t), uint8_t]
except AttributeError: pass

try: (libusb_get_parent:=dll.libusb_get_parent).restype, libusb_get_parent.argtypes = ctypes.POINTER(libusb_device), [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_device_address:=dll.libusb_get_device_address).restype, libusb_get_device_address.argtypes = uint8_t, [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_device_speed:=dll.libusb_get_device_speed).restype, libusb_get_device_speed.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device)]
except AttributeError: pass

try: (libusb_get_max_packet_size:=dll.libusb_get_max_packet_size).restype, libusb_get_max_packet_size.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.c_ubyte]
except AttributeError: pass

try: (libusb_get_max_iso_packet_size:=dll.libusb_get_max_iso_packet_size).restype, libusb_get_max_iso_packet_size.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.c_ubyte]
except AttributeError: pass

try: (libusb_get_max_alt_packet_size:=dll.libusb_get_max_alt_packet_size).restype, libusb_get_max_alt_packet_size.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.c_int32, ctypes.c_int32, ctypes.c_ubyte]
except AttributeError: pass

try: (libusb_get_interface_association_descriptors:=dll.libusb_get_interface_association_descriptors).restype, libusb_get_interface_association_descriptors.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array))]
except AttributeError: pass

try: (libusb_get_active_interface_association_descriptors:=dll.libusb_get_active_interface_association_descriptors).restype, libusb_get_active_interface_association_descriptors.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array))]
except AttributeError: pass

try: (libusb_free_interface_association_descriptors:=dll.libusb_free_interface_association_descriptors).restype, libusb_free_interface_association_descriptors.argtypes = None, [ctypes.POINTER(struct_libusb_interface_association_descriptor_array)]
except AttributeError: pass

intptr_t = ctypes.c_int64
try: (libusb_wrap_sys_device:=dll.libusb_wrap_sys_device).restype, libusb_wrap_sys_device.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), intptr_t, ctypes.POINTER(ctypes.POINTER(libusb_device_handle))]
except AttributeError: pass

try: (libusb_open:=dll.libusb_open).restype, libusb_open.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(libusb_device_handle))]
except AttributeError: pass

try: (libusb_close:=dll.libusb_close).restype, libusb_close.argtypes = None, [ctypes.POINTER(libusb_device_handle)]
except AttributeError: pass

try: (libusb_get_device:=dll.libusb_get_device).restype, libusb_get_device.argtypes = ctypes.POINTER(libusb_device), [ctypes.POINTER(libusb_device_handle)]
except AttributeError: pass

try: (libusb_set_configuration:=dll.libusb_set_configuration).restype, libusb_set_configuration.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_claim_interface:=dll.libusb_claim_interface).restype, libusb_claim_interface.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_release_interface:=dll.libusb_release_interface).restype, libusb_release_interface.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_open_device_with_vid_pid:=dll.libusb_open_device_with_vid_pid).restype, libusb_open_device_with_vid_pid.argtypes = ctypes.POINTER(libusb_device_handle), [ctypes.POINTER(libusb_context), uint16_t, uint16_t]
except AttributeError: pass

try: (libusb_set_interface_alt_setting:=dll.libusb_set_interface_alt_setting).restype, libusb_set_interface_alt_setting.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (libusb_clear_halt:=dll.libusb_clear_halt).restype, libusb_clear_halt.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte]
except AttributeError: pass

try: (libusb_reset_device:=dll.libusb_reset_device).restype, libusb_reset_device.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle)]
except AttributeError: pass

try: (libusb_alloc_streams:=dll.libusb_alloc_streams).restype, libusb_alloc_streams.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), uint32_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError: pass

try: (libusb_free_streams:=dll.libusb_free_streams).restype, libusb_free_streams.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (libusb_dev_mem_alloc:=dll.libusb_dev_mem_alloc).restype, libusb_dev_mem_alloc.argtypes = ctypes.POINTER(ctypes.c_ubyte), [ctypes.POINTER(libusb_device_handle), size_t]
except AttributeError: pass

try: (libusb_dev_mem_free:=dll.libusb_dev_mem_free).restype, libusb_dev_mem_free.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), size_t]
except AttributeError: pass

try: (libusb_kernel_driver_active:=dll.libusb_kernel_driver_active).restype, libusb_kernel_driver_active.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_detach_kernel_driver:=dll.libusb_detach_kernel_driver).restype, libusb_detach_kernel_driver.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_attach_kernel_driver:=dll.libusb_attach_kernel_driver).restype, libusb_attach_kernel_driver.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_set_auto_detach_kernel_driver:=dll.libusb_set_auto_detach_kernel_driver).restype, libusb_set_auto_detach_kernel_driver.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_int32]
except AttributeError: pass

try: (libusb_alloc_transfer:=dll.libusb_alloc_transfer).restype, libusb_alloc_transfer.argtypes = ctypes.POINTER(struct_libusb_transfer), [ctypes.c_int32]
except AttributeError: pass

try: (libusb_submit_transfer:=dll.libusb_submit_transfer).restype, libusb_submit_transfer.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError: pass

try: (libusb_cancel_transfer:=dll.libusb_cancel_transfer).restype, libusb_cancel_transfer.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError: pass

try: (libusb_free_transfer:=dll.libusb_free_transfer).restype, libusb_free_transfer.argtypes = None, [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError: pass

try: (libusb_transfer_set_stream_id:=dll.libusb_transfer_set_stream_id).restype, libusb_transfer_set_stream_id.argtypes = None, [ctypes.POINTER(struct_libusb_transfer), uint32_t]
except AttributeError: pass

try: (libusb_transfer_get_stream_id:=dll.libusb_transfer_get_stream_id).restype, libusb_transfer_get_stream_id.argtypes = uint32_t, [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError: pass

try: (libusb_control_transfer:=dll.libusb_control_transfer).restype, libusb_control_transfer.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), uint8_t, uint8_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_ubyte), uint16_t, ctypes.c_uint32]
except AttributeError: pass

try: (libusb_bulk_transfer:=dll.libusb_bulk_transfer).restype, libusb_bulk_transfer.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError: pass

try: (libusb_interrupt_transfer:=dll.libusb_interrupt_transfer).restype, libusb_interrupt_transfer.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError: pass

try: (libusb_get_string_descriptor_ascii:=dll.libusb_get_string_descriptor_ascii).restype, libusb_get_string_descriptor_ascii.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_device_handle), uint8_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError: pass

try: (libusb_try_lock_events:=dll.libusb_try_lock_events).restype, libusb_try_lock_events.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_lock_events:=dll.libusb_lock_events).restype, libusb_lock_events.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_unlock_events:=dll.libusb_unlock_events).restype, libusb_unlock_events.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_event_handling_ok:=dll.libusb_event_handling_ok).restype, libusb_event_handling_ok.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_event_handler_active:=dll.libusb_event_handler_active).restype, libusb_event_handler_active.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_interrupt_event_handler:=dll.libusb_interrupt_event_handler).restype, libusb_interrupt_event_handler.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_lock_event_waiters:=dll.libusb_lock_event_waiters).restype, libusb_lock_event_waiters.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_unlock_event_waiters:=dll.libusb_unlock_event_waiters).restype, libusb_unlock_event_waiters.argtypes = None, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

class struct_timeval(Struct): pass
__time_t = ctypes.c_int64
__suseconds_t = ctypes.c_int64
struct_timeval._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_usec', ctypes.c_int64),
]
try: (libusb_wait_for_event:=dll.libusb_wait_for_event).restype, libusb_wait_for_event.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (libusb_handle_events_timeout:=dll.libusb_handle_events_timeout).restype, libusb_handle_events_timeout.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (libusb_handle_events_timeout_completed:=dll.libusb_handle_events_timeout_completed).restype, libusb_handle_events_timeout_completed.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (libusb_handle_events:=dll.libusb_handle_events).restype, libusb_handle_events.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_handle_events_completed:=dll.libusb_handle_events_completed).restype, libusb_handle_events_completed.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (libusb_handle_events_locked:=dll.libusb_handle_events_locked).restype, libusb_handle_events_locked.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (libusb_pollfds_handle_timeouts:=dll.libusb_pollfds_handle_timeouts).restype, libusb_pollfds_handle_timeouts.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_get_next_timeout:=dll.libusb_get_next_timeout).restype, libusb_get_next_timeout.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

class struct_libusb_pollfd(Struct): pass
struct_libusb_pollfd._fields_ = [
  ('fd', ctypes.c_int32),
  ('events', ctypes.c_int16),
]
libusb_pollfd_added_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int16, ctypes.c_void_p)
libusb_pollfd_removed_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_void_p)
try: (libusb_get_pollfds:=dll.libusb_get_pollfds).restype, libusb_get_pollfds.argtypes = ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd)), [ctypes.POINTER(libusb_context)]
except AttributeError: pass

try: (libusb_free_pollfds:=dll.libusb_free_pollfds).restype, libusb_free_pollfds.argtypes = None, [ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd))]
except AttributeError: pass

try: (libusb_set_pollfd_notifiers:=dll.libusb_set_pollfd_notifiers).restype, libusb_set_pollfd_notifiers.argtypes = None, [ctypes.POINTER(libusb_context), libusb_pollfd_added_cb, libusb_pollfd_removed_cb, ctypes.c_void_p]
except AttributeError: pass

libusb_hotplug_callback_handle = ctypes.c_int32
libusb_hotplug_event = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED', 1)
LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT', 2)

libusb_hotplug_flag = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_ENUMERATE = libusb_hotplug_flag.define('LIBUSB_HOTPLUG_ENUMERATE', 1)

libusb_hotplug_callback_fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_device), libusb_hotplug_event, ctypes.c_void_p)
try: (libusb_hotplug_register_callback:=dll.libusb_hotplug_register_callback).restype, libusb_hotplug_register_callback.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, libusb_hotplug_callback_fn, ctypes.c_void_p, ctypes.POINTER(libusb_hotplug_callback_handle)]
except AttributeError: pass

try: (libusb_hotplug_deregister_callback:=dll.libusb_hotplug_deregister_callback).restype, libusb_hotplug_deregister_callback.argtypes = None, [ctypes.POINTER(libusb_context), libusb_hotplug_callback_handle]
except AttributeError: pass

try: (libusb_hotplug_get_user_data:=dll.libusb_hotplug_get_user_data).restype, libusb_hotplug_get_user_data.argtypes = ctypes.c_void_p, [ctypes.POINTER(libusb_context), libusb_hotplug_callback_handle]
except AttributeError: pass

try: (libusb_set_option:=dll.libusb_set_option).restype, libusb_set_option.argtypes = ctypes.c_int32, [ctypes.POINTER(libusb_context), enum_libusb_option]
except AttributeError: pass

LIBUSB_DEPRECATED_FOR = lambda f: __attribute__ ((deprecated))
LIBUSB_API_VERSION = 0x0100010A
LIBUSBX_API_VERSION = LIBUSB_API_VERSION
LIBUSB_DT_DEVICE_SIZE = 18
LIBUSB_DT_CONFIG_SIZE = 9
LIBUSB_DT_INTERFACE_SIZE = 9
LIBUSB_DT_ENDPOINT_SIZE = 7
LIBUSB_DT_ENDPOINT_AUDIO_SIZE = 9
LIBUSB_DT_HUB_NONVAR_SIZE = 7
LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE = 6
LIBUSB_DT_BOS_SIZE = 5
LIBUSB_DT_DEVICE_CAPABILITY_SIZE = 3
LIBUSB_BT_USB_2_0_EXTENSION_SIZE = 7
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE = 10
LIBUSB_BT_CONTAINER_ID_SIZE = 20
LIBUSB_BT_PLATFORM_DESCRIPTOR_MIN_SIZE = 20
LIBUSB_DT_BOS_MAX_SIZE = (LIBUSB_DT_BOS_SIZE + LIBUSB_BT_USB_2_0_EXTENSION_SIZE + LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE + LIBUSB_BT_CONTAINER_ID_SIZE)
LIBUSB_ENDPOINT_ADDRESS_MASK = 0x0f
LIBUSB_ENDPOINT_DIR_MASK = 0x80
LIBUSB_TRANSFER_TYPE_MASK = 0x03
LIBUSB_ISO_SYNC_TYPE_MASK = 0x0c
LIBUSB_ISO_USAGE_TYPE_MASK = 0x30
LIBUSB_ERROR_COUNT = 14
LIBUSB_OPTION_WEAK_AUTHORITY = LIBUSB_OPTION_NO_DEVICE_DISCOVERY
LIBUSB_HOTPLUG_NO_FLAGS = 0
LIBUSB_HOTPLUG_MATCH_ANY = -1