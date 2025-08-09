# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os


class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libusb'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libusb'] = None if (lib_path:=os.getenv('LIBUSB_PATH', ctypes.util.find_library('usb-1.0'))) is None else ctypes.CDLL(lib_path) #  ctypes.CDLL('libusb')
class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



LIBUSB_H = True # macro
ZERO_SIZED_ARRAY = True # macro
# def LIBUSB_DEPRECATED_FOR(f):  # macro
#    return ((deprecated))
# LIBUSB_PACKED = ((packed)) # macro
LIBUSB_CALL = True # macro
LIBUSB_API_VERSION = 0x01000109 # macro
LIBUSBX_API_VERSION = 0x01000109 # macro
LIBUSB_DT_DEVICE_SIZE = 18 # macro
LIBUSB_DT_CONFIG_SIZE = 9 # macro
LIBUSB_DT_INTERFACE_SIZE = 9 # macro
LIBUSB_DT_ENDPOINT_SIZE = 7 # macro
LIBUSB_DT_ENDPOINT_AUDIO_SIZE = 9 # macro
LIBUSB_DT_HUB_NONVAR_SIZE = 7 # macro
LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE = 6 # macro
LIBUSB_DT_BOS_SIZE = 5 # macro
LIBUSB_DT_DEVICE_CAPABILITY_SIZE = 3 # macro
LIBUSB_BT_USB_2_0_EXTENSION_SIZE = 7 # macro
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE = 10 # macro
LIBUSB_BT_CONTAINER_ID_SIZE = 20 # macro
LIBUSB_DT_BOS_MAX_SIZE = (5+7+10+20) # macro
LIBUSB_ENDPOINT_ADDRESS_MASK = 0x0f # macro
LIBUSB_ENDPOINT_DIR_MASK = 0x80 # macro
LIBUSB_TRANSFER_TYPE_MASK = 0x03 # macro
LIBUSB_ISO_SYNC_TYPE_MASK = 0x0c # macro
LIBUSB_ISO_USAGE_TYPE_MASK = 0x30 # macro
LIBUSB_ERROR_COUNT = 14 # macro
LIBUSB_HOTPLUG_NO_FLAGS = 0 # macro
LIBUSB_HOTPLUG_MATCH_ANY = -1 # macro
uint16_t = ctypes.c_uint16
try:
    libusb_cpu_to_le16 = _libraries['libusb'].libusb_cpu_to_le16
    libusb_cpu_to_le16.restype = uint16_t
    libusb_cpu_to_le16.argtypes = [uint16_t]
except AttributeError:
    pass
 # macro

# values for enumeration 'libusb_class_code'
libusb_class_code__enumvalues = {
    0: 'LIBUSB_CLASS_PER_INTERFACE',
    1: 'LIBUSB_CLASS_AUDIO',
    2: 'LIBUSB_CLASS_COMM',
    3: 'LIBUSB_CLASS_HID',
    5: 'LIBUSB_CLASS_PHYSICAL',
    6: 'LIBUSB_CLASS_IMAGE',
    6: 'LIBUSB_CLASS_PTP',
    7: 'LIBUSB_CLASS_PRINTER',
    8: 'LIBUSB_CLASS_MASS_STORAGE',
    9: 'LIBUSB_CLASS_HUB',
    10: 'LIBUSB_CLASS_DATA',
    11: 'LIBUSB_CLASS_SMART_CARD',
    13: 'LIBUSB_CLASS_CONTENT_SECURITY',
    14: 'LIBUSB_CLASS_VIDEO',
    15: 'LIBUSB_CLASS_PERSONAL_HEALTHCARE',
    220: 'LIBUSB_CLASS_DIAGNOSTIC_DEVICE',
    224: 'LIBUSB_CLASS_WIRELESS',
    239: 'LIBUSB_CLASS_MISCELLANEOUS',
    254: 'LIBUSB_CLASS_APPLICATION',
    255: 'LIBUSB_CLASS_VENDOR_SPEC',
}
LIBUSB_CLASS_PER_INTERFACE = 0
LIBUSB_CLASS_AUDIO = 1
LIBUSB_CLASS_COMM = 2
LIBUSB_CLASS_HID = 3
LIBUSB_CLASS_PHYSICAL = 5
LIBUSB_CLASS_IMAGE = 6
LIBUSB_CLASS_PTP = 6
LIBUSB_CLASS_PRINTER = 7
LIBUSB_CLASS_MASS_STORAGE = 8
LIBUSB_CLASS_HUB = 9
LIBUSB_CLASS_DATA = 10
LIBUSB_CLASS_SMART_CARD = 11
LIBUSB_CLASS_CONTENT_SECURITY = 13
LIBUSB_CLASS_VIDEO = 14
LIBUSB_CLASS_PERSONAL_HEALTHCARE = 15
LIBUSB_CLASS_DIAGNOSTIC_DEVICE = 220
LIBUSB_CLASS_WIRELESS = 224
LIBUSB_CLASS_MISCELLANEOUS = 239
LIBUSB_CLASS_APPLICATION = 254
LIBUSB_CLASS_VENDOR_SPEC = 255
libusb_class_code = ctypes.c_uint32 # enum

# values for enumeration 'libusb_descriptor_type'
libusb_descriptor_type__enumvalues = {
    1: 'LIBUSB_DT_DEVICE',
    2: 'LIBUSB_DT_CONFIG',
    3: 'LIBUSB_DT_STRING',
    4: 'LIBUSB_DT_INTERFACE',
    5: 'LIBUSB_DT_ENDPOINT',
    15: 'LIBUSB_DT_BOS',
    16: 'LIBUSB_DT_DEVICE_CAPABILITY',
    33: 'LIBUSB_DT_HID',
    34: 'LIBUSB_DT_REPORT',
    35: 'LIBUSB_DT_PHYSICAL',
    41: 'LIBUSB_DT_HUB',
    42: 'LIBUSB_DT_SUPERSPEED_HUB',
    48: 'LIBUSB_DT_SS_ENDPOINT_COMPANION',
}
LIBUSB_DT_DEVICE = 1
LIBUSB_DT_CONFIG = 2
LIBUSB_DT_STRING = 3
LIBUSB_DT_INTERFACE = 4
LIBUSB_DT_ENDPOINT = 5
LIBUSB_DT_BOS = 15
LIBUSB_DT_DEVICE_CAPABILITY = 16
LIBUSB_DT_HID = 33
LIBUSB_DT_REPORT = 34
LIBUSB_DT_PHYSICAL = 35
LIBUSB_DT_HUB = 41
LIBUSB_DT_SUPERSPEED_HUB = 42
LIBUSB_DT_SS_ENDPOINT_COMPANION = 48
libusb_descriptor_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_endpoint_direction'
libusb_endpoint_direction__enumvalues = {
    0: 'LIBUSB_ENDPOINT_OUT',
    128: 'LIBUSB_ENDPOINT_IN',
}
LIBUSB_ENDPOINT_OUT = 0
LIBUSB_ENDPOINT_IN = 128
libusb_endpoint_direction = ctypes.c_uint32 # enum

# values for enumeration 'libusb_endpoint_transfer_type'
libusb_endpoint_transfer_type__enumvalues = {
    0: 'LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL',
    1: 'LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS',
    2: 'LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK',
    3: 'LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT',
}
LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL = 0
LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS = 1
LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK = 2
LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT = 3
libusb_endpoint_transfer_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_standard_request'
libusb_standard_request__enumvalues = {
    0: 'LIBUSB_REQUEST_GET_STATUS',
    1: 'LIBUSB_REQUEST_CLEAR_FEATURE',
    3: 'LIBUSB_REQUEST_SET_FEATURE',
    5: 'LIBUSB_REQUEST_SET_ADDRESS',
    6: 'LIBUSB_REQUEST_GET_DESCRIPTOR',
    7: 'LIBUSB_REQUEST_SET_DESCRIPTOR',
    8: 'LIBUSB_REQUEST_GET_CONFIGURATION',
    9: 'LIBUSB_REQUEST_SET_CONFIGURATION',
    10: 'LIBUSB_REQUEST_GET_INTERFACE',
    11: 'LIBUSB_REQUEST_SET_INTERFACE',
    12: 'LIBUSB_REQUEST_SYNCH_FRAME',
    48: 'LIBUSB_REQUEST_SET_SEL',
    49: 'LIBUSB_SET_ISOCH_DELAY',
}
LIBUSB_REQUEST_GET_STATUS = 0
LIBUSB_REQUEST_CLEAR_FEATURE = 1
LIBUSB_REQUEST_SET_FEATURE = 3
LIBUSB_REQUEST_SET_ADDRESS = 5
LIBUSB_REQUEST_GET_DESCRIPTOR = 6
LIBUSB_REQUEST_SET_DESCRIPTOR = 7
LIBUSB_REQUEST_GET_CONFIGURATION = 8
LIBUSB_REQUEST_SET_CONFIGURATION = 9
LIBUSB_REQUEST_GET_INTERFACE = 10
LIBUSB_REQUEST_SET_INTERFACE = 11
LIBUSB_REQUEST_SYNCH_FRAME = 12
LIBUSB_REQUEST_SET_SEL = 48
LIBUSB_SET_ISOCH_DELAY = 49
libusb_standard_request = ctypes.c_uint32 # enum

# values for enumeration 'libusb_request_type'
libusb_request_type__enumvalues = {
    0: 'LIBUSB_REQUEST_TYPE_STANDARD',
    32: 'LIBUSB_REQUEST_TYPE_CLASS',
    64: 'LIBUSB_REQUEST_TYPE_VENDOR',
    96: 'LIBUSB_REQUEST_TYPE_RESERVED',
}
LIBUSB_REQUEST_TYPE_STANDARD = 0
LIBUSB_REQUEST_TYPE_CLASS = 32
LIBUSB_REQUEST_TYPE_VENDOR = 64
LIBUSB_REQUEST_TYPE_RESERVED = 96
libusb_request_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_request_recipient'
libusb_request_recipient__enumvalues = {
    0: 'LIBUSB_RECIPIENT_DEVICE',
    1: 'LIBUSB_RECIPIENT_INTERFACE',
    2: 'LIBUSB_RECIPIENT_ENDPOINT',
    3: 'LIBUSB_RECIPIENT_OTHER',
}
LIBUSB_RECIPIENT_DEVICE = 0
LIBUSB_RECIPIENT_INTERFACE = 1
LIBUSB_RECIPIENT_ENDPOINT = 2
LIBUSB_RECIPIENT_OTHER = 3
libusb_request_recipient = ctypes.c_uint32 # enum

# values for enumeration 'libusb_iso_sync_type'
libusb_iso_sync_type__enumvalues = {
    0: 'LIBUSB_ISO_SYNC_TYPE_NONE',
    1: 'LIBUSB_ISO_SYNC_TYPE_ASYNC',
    2: 'LIBUSB_ISO_SYNC_TYPE_ADAPTIVE',
    3: 'LIBUSB_ISO_SYNC_TYPE_SYNC',
}
LIBUSB_ISO_SYNC_TYPE_NONE = 0
LIBUSB_ISO_SYNC_TYPE_ASYNC = 1
LIBUSB_ISO_SYNC_TYPE_ADAPTIVE = 2
LIBUSB_ISO_SYNC_TYPE_SYNC = 3
libusb_iso_sync_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_iso_usage_type'
libusb_iso_usage_type__enumvalues = {
    0: 'LIBUSB_ISO_USAGE_TYPE_DATA',
    1: 'LIBUSB_ISO_USAGE_TYPE_FEEDBACK',
    2: 'LIBUSB_ISO_USAGE_TYPE_IMPLICIT',
}
LIBUSB_ISO_USAGE_TYPE_DATA = 0
LIBUSB_ISO_USAGE_TYPE_FEEDBACK = 1
LIBUSB_ISO_USAGE_TYPE_IMPLICIT = 2
libusb_iso_usage_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_supported_speed'
libusb_supported_speed__enumvalues = {
    1: 'LIBUSB_LOW_SPEED_OPERATION',
    2: 'LIBUSB_FULL_SPEED_OPERATION',
    4: 'LIBUSB_HIGH_SPEED_OPERATION',
    8: 'LIBUSB_SUPER_SPEED_OPERATION',
}
LIBUSB_LOW_SPEED_OPERATION = 1
LIBUSB_FULL_SPEED_OPERATION = 2
LIBUSB_HIGH_SPEED_OPERATION = 4
LIBUSB_SUPER_SPEED_OPERATION = 8
libusb_supported_speed = ctypes.c_uint32 # enum

# values for enumeration 'libusb_usb_2_0_extension_attributes'
libusb_usb_2_0_extension_attributes__enumvalues = {
    2: 'LIBUSB_BM_LPM_SUPPORT',
}
LIBUSB_BM_LPM_SUPPORT = 2
libusb_usb_2_0_extension_attributes = ctypes.c_uint32 # enum

# values for enumeration 'libusb_ss_usb_device_capability_attributes'
libusb_ss_usb_device_capability_attributes__enumvalues = {
    2: 'LIBUSB_BM_LTM_SUPPORT',
}
LIBUSB_BM_LTM_SUPPORT = 2
libusb_ss_usb_device_capability_attributes = ctypes.c_uint32 # enum

# values for enumeration 'libusb_bos_type'
libusb_bos_type__enumvalues = {
    1: 'LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY',
    2: 'LIBUSB_BT_USB_2_0_EXTENSION',
    3: 'LIBUSB_BT_SS_USB_DEVICE_CAPABILITY',
    4: 'LIBUSB_BT_CONTAINER_ID',
}
LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY = 1
LIBUSB_BT_USB_2_0_EXTENSION = 2
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY = 3
LIBUSB_BT_CONTAINER_ID = 4
libusb_bos_type = ctypes.c_uint32 # enum
class struct_libusb_device_descriptor(Structure):
    pass

struct_libusb_device_descriptor._pack_ = 1 # source:False
struct_libusb_device_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bcdUSB', ctypes.c_uint16),
    ('bDeviceClass', ctypes.c_ubyte),
    ('bDeviceSubClass', ctypes.c_ubyte),
    ('bDeviceProtocol', ctypes.c_ubyte),
    ('bMaxPacketSize0', ctypes.c_ubyte),
    ('idVendor', ctypes.c_uint16),
    ('idProduct', ctypes.c_uint16),
    ('bcdDevice', ctypes.c_uint16),
    ('iManufacturer', ctypes.c_ubyte),
    ('iProduct', ctypes.c_ubyte),
    ('iSerialNumber', ctypes.c_ubyte),
    ('bNumConfigurations', ctypes.c_ubyte),
]

class struct_libusb_endpoint_descriptor(Structure):
    pass

struct_libusb_endpoint_descriptor._pack_ = 1 # source:False
struct_libusb_endpoint_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bEndpointAddress', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('wMaxPacketSize', ctypes.c_uint16),
    ('bInterval', ctypes.c_ubyte),
    ('bRefresh', ctypes.c_ubyte),
    ('bSynchAddress', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extra_length', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_libusb_interface_descriptor(Structure):
    pass

struct_libusb_interface_descriptor._pack_ = 1 # source:False
struct_libusb_interface_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bInterfaceNumber', ctypes.c_ubyte),
    ('bAlternateSetting', ctypes.c_ubyte),
    ('bNumEndpoints', ctypes.c_ubyte),
    ('bInterfaceClass', ctypes.c_ubyte),
    ('bInterfaceSubClass', ctypes.c_ubyte),
    ('bInterfaceProtocol', ctypes.c_ubyte),
    ('iInterface', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('endpoint', ctypes.POINTER(struct_libusb_endpoint_descriptor)),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extra_length', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_libusb_interface(Structure):
    pass

struct_libusb_interface._pack_ = 1 # source:False
struct_libusb_interface._fields_ = [
    ('altsetting', ctypes.POINTER(struct_libusb_interface_descriptor)),
    ('num_altsetting', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_libusb_config_descriptor(Structure):
    pass

struct_libusb_config_descriptor._pack_ = 1 # source:False
struct_libusb_config_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('wTotalLength', ctypes.c_uint16),
    ('bNumInterfaces', ctypes.c_ubyte),
    ('bConfigurationValue', ctypes.c_ubyte),
    ('iConfiguration', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('MaxPower', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('interface', ctypes.POINTER(struct_libusb_interface)),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extra_length', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_libusb_ss_endpoint_companion_descriptor(Structure):
    pass

struct_libusb_ss_endpoint_companion_descriptor._pack_ = 1 # source:False
struct_libusb_ss_endpoint_companion_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bMaxBurst', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('wBytesPerInterval', ctypes.c_uint16),
]

class struct_libusb_bos_dev_capability_descriptor(Structure):
    pass

struct_libusb_bos_dev_capability_descriptor._pack_ = 1 # source:False
struct_libusb_bos_dev_capability_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bDevCapabilityType', ctypes.c_ubyte),
    ('dev_capability_data', ctypes.c_ubyte * 0),
]

class struct_libusb_bos_descriptor(Structure):
    pass

struct_libusb_bos_descriptor._pack_ = 1 # source:False
struct_libusb_bos_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('wTotalLength', ctypes.c_uint16),
    ('bNumDeviceCaps', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('dev_capability', ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor) * 0),
]

class struct_libusb_usb_2_0_extension_descriptor(Structure):
    pass

struct_libusb_usb_2_0_extension_descriptor._pack_ = 1 # source:False
struct_libusb_usb_2_0_extension_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bDevCapabilityType', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_uint32),
]

class struct_libusb_ss_usb_device_capability_descriptor(Structure):
    pass

struct_libusb_ss_usb_device_capability_descriptor._pack_ = 1 # source:False
struct_libusb_ss_usb_device_capability_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bDevCapabilityType', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('wSpeedSupported', ctypes.c_uint16),
    ('bFunctionalitySupport', ctypes.c_ubyte),
    ('bU1DevExitLat', ctypes.c_ubyte),
    ('bU2DevExitLat', ctypes.c_uint16),
]

class struct_libusb_container_id_descriptor(Structure):
    pass

struct_libusb_container_id_descriptor._pack_ = 1 # source:False
struct_libusb_container_id_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bDevCapabilityType', ctypes.c_ubyte),
    ('bReserved', ctypes.c_ubyte),
    ('ContainerID', ctypes.c_ubyte * 16),
]

class struct_libusb_control_setup(Structure):
    pass

struct_libusb_control_setup._pack_ = 1 # source:True
struct_libusb_control_setup._fields_ = [
    ('bmRequestType', ctypes.c_ubyte),
    ('bRequest', ctypes.c_ubyte),
    ('wValue', ctypes.c_uint16),
    ('wIndex', ctypes.c_uint16),
    ('wLength', ctypes.c_uint16),
]

# LIBUSB_CONTROL_SETUP_SIZE = (ctypes.sizeof(struct_libusb_control_setup)) # macro
class struct_libusb_context(Structure):
    pass

class struct_libusb_device(Structure):
    pass

class struct_libusb_device_handle(Structure):
    pass

class struct_libusb_version(Structure):
    pass

struct_libusb_version._pack_ = 1 # source:False
struct_libusb_version._fields_ = [
    ('major', ctypes.c_uint16),
    ('minor', ctypes.c_uint16),
    ('micro', ctypes.c_uint16),
    ('nano', ctypes.c_uint16),
    ('rc', ctypes.POINTER(ctypes.c_char)),
    ('describe', ctypes.POINTER(ctypes.c_char)),
]

libusb_context = struct_libusb_context
libusb_device = struct_libusb_device
libusb_device_handle = struct_libusb_device_handle

# values for enumeration 'libusb_speed'
libusb_speed__enumvalues = {
    0: 'LIBUSB_SPEED_UNKNOWN',
    1: 'LIBUSB_SPEED_LOW',
    2: 'LIBUSB_SPEED_FULL',
    3: 'LIBUSB_SPEED_HIGH',
    4: 'LIBUSB_SPEED_SUPER',
    5: 'LIBUSB_SPEED_SUPER_PLUS',
}
LIBUSB_SPEED_UNKNOWN = 0
LIBUSB_SPEED_LOW = 1
LIBUSB_SPEED_FULL = 2
LIBUSB_SPEED_HIGH = 3
LIBUSB_SPEED_SUPER = 4
LIBUSB_SPEED_SUPER_PLUS = 5
libusb_speed = ctypes.c_uint32 # enum

# values for enumeration 'libusb_error'
libusb_error__enumvalues = {
    0: 'LIBUSB_SUCCESS',
    -1: 'LIBUSB_ERROR_IO',
    -2: 'LIBUSB_ERROR_INVALID_PARAM',
    -3: 'LIBUSB_ERROR_ACCESS',
    -4: 'LIBUSB_ERROR_NO_DEVICE',
    -5: 'LIBUSB_ERROR_NOT_FOUND',
    -6: 'LIBUSB_ERROR_BUSY',
    -7: 'LIBUSB_ERROR_TIMEOUT',
    -8: 'LIBUSB_ERROR_OVERFLOW',
    -9: 'LIBUSB_ERROR_PIPE',
    -10: 'LIBUSB_ERROR_INTERRUPTED',
    -11: 'LIBUSB_ERROR_NO_MEM',
    -12: 'LIBUSB_ERROR_NOT_SUPPORTED',
    -99: 'LIBUSB_ERROR_OTHER',
}
LIBUSB_SUCCESS = 0
LIBUSB_ERROR_IO = -1
LIBUSB_ERROR_INVALID_PARAM = -2
LIBUSB_ERROR_ACCESS = -3
LIBUSB_ERROR_NO_DEVICE = -4
LIBUSB_ERROR_NOT_FOUND = -5
LIBUSB_ERROR_BUSY = -6
LIBUSB_ERROR_TIMEOUT = -7
LIBUSB_ERROR_OVERFLOW = -8
LIBUSB_ERROR_PIPE = -9
LIBUSB_ERROR_INTERRUPTED = -10
LIBUSB_ERROR_NO_MEM = -11
LIBUSB_ERROR_NOT_SUPPORTED = -12
LIBUSB_ERROR_OTHER = -99
libusb_error = ctypes.c_int32 # enum

# values for enumeration 'libusb_transfer_type'
libusb_transfer_type__enumvalues = {
    0: 'LIBUSB_TRANSFER_TYPE_CONTROL',
    1: 'LIBUSB_TRANSFER_TYPE_ISOCHRONOUS',
    2: 'LIBUSB_TRANSFER_TYPE_BULK',
    3: 'LIBUSB_TRANSFER_TYPE_INTERRUPT',
    4: 'LIBUSB_TRANSFER_TYPE_BULK_STREAM',
}
LIBUSB_TRANSFER_TYPE_CONTROL = 0
LIBUSB_TRANSFER_TYPE_ISOCHRONOUS = 1
LIBUSB_TRANSFER_TYPE_BULK = 2
LIBUSB_TRANSFER_TYPE_INTERRUPT = 3
LIBUSB_TRANSFER_TYPE_BULK_STREAM = 4
libusb_transfer_type = ctypes.c_uint32 # enum

# values for enumeration 'libusb_transfer_status'
libusb_transfer_status__enumvalues = {
    0: 'LIBUSB_TRANSFER_COMPLETED',
    1: 'LIBUSB_TRANSFER_ERROR',
    2: 'LIBUSB_TRANSFER_TIMED_OUT',
    3: 'LIBUSB_TRANSFER_CANCELLED',
    4: 'LIBUSB_TRANSFER_STALL',
    5: 'LIBUSB_TRANSFER_NO_DEVICE',
    6: 'LIBUSB_TRANSFER_OVERFLOW',
}
LIBUSB_TRANSFER_COMPLETED = 0
LIBUSB_TRANSFER_ERROR = 1
LIBUSB_TRANSFER_TIMED_OUT = 2
LIBUSB_TRANSFER_CANCELLED = 3
LIBUSB_TRANSFER_STALL = 4
LIBUSB_TRANSFER_NO_DEVICE = 5
LIBUSB_TRANSFER_OVERFLOW = 6
libusb_transfer_status = ctypes.c_uint32 # enum

# values for enumeration 'libusb_transfer_flags'
libusb_transfer_flags__enumvalues = {
    1: 'LIBUSB_TRANSFER_SHORT_NOT_OK',
    2: 'LIBUSB_TRANSFER_FREE_BUFFER',
    4: 'LIBUSB_TRANSFER_FREE_TRANSFER',
    8: 'LIBUSB_TRANSFER_ADD_ZERO_PACKET',
}
LIBUSB_TRANSFER_SHORT_NOT_OK = 1
LIBUSB_TRANSFER_FREE_BUFFER = 2
LIBUSB_TRANSFER_FREE_TRANSFER = 4
LIBUSB_TRANSFER_ADD_ZERO_PACKET = 8
libusb_transfer_flags = ctypes.c_uint32 # enum
class struct_libusb_iso_packet_descriptor(Structure):
    pass

struct_libusb_iso_packet_descriptor._pack_ = 1 # source:False
struct_libusb_iso_packet_descriptor._fields_ = [
    ('length', ctypes.c_uint32),
    ('actual_length', ctypes.c_uint32),
    ('status', libusb_transfer_status),
]

class struct_libusb_transfer(Structure):
    pass

libusb_transfer_cb_fn = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_transfer))

# values for enumeration 'libusb_capability'
libusb_capability__enumvalues = {
    0: 'LIBUSB_CAP_HAS_CAPABILITY',
    1: 'LIBUSB_CAP_HAS_HOTPLUG',
    256: 'LIBUSB_CAP_HAS_HID_ACCESS',
    257: 'LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER',
}
LIBUSB_CAP_HAS_CAPABILITY = 0
LIBUSB_CAP_HAS_HOTPLUG = 1
LIBUSB_CAP_HAS_HID_ACCESS = 256
LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER = 257
libusb_capability = ctypes.c_uint32 # enum

# values for enumeration 'libusb_log_level'
libusb_log_level__enumvalues = {
    0: 'LIBUSB_LOG_LEVEL_NONE',
    1: 'LIBUSB_LOG_LEVEL_ERROR',
    2: 'LIBUSB_LOG_LEVEL_WARNING',
    3: 'LIBUSB_LOG_LEVEL_INFO',
    4: 'LIBUSB_LOG_LEVEL_DEBUG',
}
LIBUSB_LOG_LEVEL_NONE = 0
LIBUSB_LOG_LEVEL_ERROR = 1
LIBUSB_LOG_LEVEL_WARNING = 2
LIBUSB_LOG_LEVEL_INFO = 3
LIBUSB_LOG_LEVEL_DEBUG = 4
libusb_log_level = ctypes.c_uint32 # enum

# values for enumeration 'libusb_log_cb_mode'
libusb_log_cb_mode__enumvalues = {
    1: 'LIBUSB_LOG_CB_GLOBAL',
    2: 'LIBUSB_LOG_CB_CONTEXT',
}
LIBUSB_LOG_CB_GLOBAL = 1
LIBUSB_LOG_CB_CONTEXT = 2
libusb_log_cb_mode = ctypes.c_uint32 # enum
libusb_log_cb = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_context), libusb_log_level, ctypes.POINTER(ctypes.c_char))
try:
    libusb_init = _libraries['libusb'].libusb_init
    libusb_init.restype = ctypes.c_int32
    libusb_init.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_libusb_context))]
except AttributeError:
    pass
try:
    libusb_exit = _libraries['libusb'].libusb_exit
    libusb_exit.restype = None
    libusb_exit.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_set_debug = _libraries['libusb'].libusb_set_debug
    libusb_set_debug.restype = None
    libusb_set_debug.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_set_log_cb = _libraries['libusb'].libusb_set_log_cb
    libusb_set_log_cb.restype = None
    libusb_set_log_cb.argtypes = [ctypes.POINTER(struct_libusb_context), libusb_log_cb, ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_get_version = _libraries['libusb'].libusb_get_version
    libusb_get_version.restype = ctypes.POINTER(struct_libusb_version)
    libusb_get_version.argtypes = []
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    libusb_has_capability = _libraries['libusb'].libusb_has_capability
    libusb_has_capability.restype = ctypes.c_int32
    libusb_has_capability.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    libusb_error_name = _libraries['libusb'].libusb_error_name
    libusb_error_name.restype = ctypes.POINTER(ctypes.c_char)
    libusb_error_name.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_setlocale = _libraries['libusb'].libusb_setlocale
    libusb_setlocale.restype = ctypes.c_int32
    libusb_setlocale.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    libusb_strerror = _libraries['libusb'].libusb_strerror
    libusb_strerror.restype = ctypes.POINTER(ctypes.c_char)
    libusb_strerror.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
ssize_t = ctypes.c_int64
try:
    libusb_get_device_list = _libraries['libusb'].libusb_get_device_list
    libusb_get_device_list.restype = ssize_t
    libusb_get_device_list.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_libusb_device)))]
except AttributeError:
    pass
try:
    libusb_free_device_list = _libraries['libusb'].libusb_free_device_list
    libusb_free_device_list.restype = None
    libusb_free_device_list.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_libusb_device)), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_ref_device = _libraries['libusb'].libusb_ref_device
    libusb_ref_device.restype = ctypes.POINTER(struct_libusb_device)
    libusb_ref_device.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_unref_device = _libraries['libusb'].libusb_unref_device
    libusb_unref_device.restype = None
    libusb_unref_device.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_configuration = _libraries['libusb'].libusb_get_configuration
    libusb_get_configuration.restype = ctypes.c_int32
    libusb_get_configuration.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    libusb_get_device_descriptor = _libraries['libusb'].libusb_get_device_descriptor
    libusb_get_device_descriptor.restype = ctypes.c_int32
    libusb_get_device_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.POINTER(struct_libusb_device_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_active_config_descriptor = _libraries['libusb'].libusb_get_active_config_descriptor
    libusb_get_active_config_descriptor.restype = ctypes.c_int32
    libusb_get_active_config_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    libusb_get_config_descriptor = _libraries['libusb'].libusb_get_config_descriptor
    libusb_get_config_descriptor.restype = ctypes.c_int32
    libusb_get_config_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError:
    pass
try:
    libusb_get_config_descriptor_by_value = _libraries['libusb'].libusb_get_config_descriptor_by_value
    libusb_get_config_descriptor_by_value.restype = ctypes.c_int32
    libusb_get_config_descriptor_by_value.argtypes = [ctypes.POINTER(struct_libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_config_descriptor = _libraries['libusb'].libusb_free_config_descriptor
    libusb_free_config_descriptor.restype = None
    libusb_free_config_descriptor.argtypes = [ctypes.POINTER(struct_libusb_config_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_ss_endpoint_companion_descriptor = _libraries['libusb'].libusb_get_ss_endpoint_companion_descriptor
    libusb_get_ss_endpoint_companion_descriptor.restype = ctypes.c_int32
    libusb_get_ss_endpoint_companion_descriptor.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_endpoint_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_ss_endpoint_companion_descriptor = _libraries['libusb'].libusb_free_ss_endpoint_companion_descriptor
    libusb_free_ss_endpoint_companion_descriptor.restype = None
    libusb_free_ss_endpoint_companion_descriptor.argtypes = [ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_bos_descriptor = _libraries['libusb'].libusb_get_bos_descriptor
    libusb_get_bos_descriptor.restype = ctypes.c_int32
    libusb_get_bos_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.POINTER(ctypes.POINTER(struct_libusb_bos_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_bos_descriptor = _libraries['libusb'].libusb_free_bos_descriptor
    libusb_free_bos_descriptor.restype = None
    libusb_free_bos_descriptor.argtypes = [ctypes.POINTER(struct_libusb_bos_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_usb_2_0_extension_descriptor = _libraries['libusb'].libusb_get_usb_2_0_extension_descriptor
    libusb_get_usb_2_0_extension_descriptor.restype = ctypes.c_int32
    libusb_get_usb_2_0_extension_descriptor.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_usb_2_0_extension_descriptor = _libraries['libusb'].libusb_free_usb_2_0_extension_descriptor
    libusb_free_usb_2_0_extension_descriptor.restype = None
    libusb_free_usb_2_0_extension_descriptor.argtypes = [ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_ss_usb_device_capability_descriptor = _libraries['libusb'].libusb_get_ss_usb_device_capability_descriptor
    libusb_get_ss_usb_device_capability_descriptor.restype = ctypes.c_int32
    libusb_get_ss_usb_device_capability_descriptor.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_ss_usb_device_capability_descriptor = _libraries['libusb'].libusb_free_ss_usb_device_capability_descriptor
    libusb_free_ss_usb_device_capability_descriptor.restype = None
    libusb_free_ss_usb_device_capability_descriptor.argtypes = [ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_container_id_descriptor = _libraries['libusb'].libusb_get_container_id_descriptor
    libusb_get_container_id_descriptor.restype = ctypes.c_int32
    libusb_get_container_id_descriptor.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_container_id_descriptor))]
except AttributeError:
    pass
try:
    libusb_free_container_id_descriptor = _libraries['libusb'].libusb_free_container_id_descriptor
    libusb_free_container_id_descriptor.restype = None
    libusb_free_container_id_descriptor.argtypes = [ctypes.POINTER(struct_libusb_container_id_descriptor)]
except AttributeError:
    pass
try:
    libusb_get_bus_number = _libraries['libusb'].libusb_get_bus_number
    libusb_get_bus_number.restype = uint8_t
    libusb_get_bus_number.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_port_number = _libraries['libusb'].libusb_get_port_number
    libusb_get_port_number.restype = uint8_t
    libusb_get_port_number.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_port_numbers = _libraries['libusb'].libusb_get_port_numbers
    libusb_get_port_numbers.restype = ctypes.c_int32
    libusb_get_port_numbers.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_get_port_path = _libraries['libusb'].libusb_get_port_path
    libusb_get_port_path.restype = ctypes.c_int32
    libusb_get_port_path.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_device), ctypes.POINTER(ctypes.c_ubyte), uint8_t]
except AttributeError:
    pass
try:
    libusb_get_parent = _libraries['libusb'].libusb_get_parent
    libusb_get_parent.restype = ctypes.POINTER(struct_libusb_device)
    libusb_get_parent.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_device_address = _libraries['libusb'].libusb_get_device_address
    libusb_get_device_address.restype = uint8_t
    libusb_get_device_address.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_device_speed = _libraries['libusb'].libusb_get_device_speed
    libusb_get_device_speed.restype = ctypes.c_int32
    libusb_get_device_speed.argtypes = [ctypes.POINTER(struct_libusb_device)]
except AttributeError:
    pass
try:
    libusb_get_max_packet_size = _libraries['libusb'].libusb_get_max_packet_size
    libusb_get_max_packet_size.restype = ctypes.c_int32
    libusb_get_max_packet_size.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.c_ubyte]
except AttributeError:
    pass
try:
    libusb_get_max_iso_packet_size = _libraries['libusb'].libusb_get_max_iso_packet_size
    libusb_get_max_iso_packet_size.restype = ctypes.c_int32
    libusb_get_max_iso_packet_size.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.c_ubyte]
except AttributeError:
    pass
intptr_t = ctypes.c_int64
try:
    libusb_wrap_sys_device = _libraries['libusb'].libusb_wrap_sys_device
    libusb_wrap_sys_device.restype = ctypes.c_int32
    libusb_wrap_sys_device.argtypes = [ctypes.POINTER(struct_libusb_context), intptr_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_device_handle))]
except AttributeError:
    pass
try:
    libusb_open = _libraries['libusb'].libusb_open
    libusb_open.restype = ctypes.c_int32
    libusb_open.argtypes = [ctypes.POINTER(struct_libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_device_handle))]
except AttributeError:
    pass
try:
    libusb_close = _libraries['libusb'].libusb_close
    libusb_close.restype = None
    libusb_close.argtypes = [ctypes.POINTER(struct_libusb_device_handle)]
except AttributeError:
    pass
try:
    libusb_get_device = _libraries['libusb'].libusb_get_device
    libusb_get_device.restype = ctypes.POINTER(struct_libusb_device)
    libusb_get_device.argtypes = [ctypes.POINTER(struct_libusb_device_handle)]
except AttributeError:
    pass
try:
    libusb_set_configuration = _libraries['libusb'].libusb_set_configuration
    libusb_set_configuration.restype = ctypes.c_int32
    libusb_set_configuration.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_claim_interface = _libraries['libusb'].libusb_claim_interface
    libusb_claim_interface.restype = ctypes.c_int32
    libusb_claim_interface.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_release_interface = _libraries['libusb'].libusb_release_interface
    libusb_release_interface.restype = ctypes.c_int32
    libusb_release_interface.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_open_device_with_vid_pid = _libraries['libusb'].libusb_open_device_with_vid_pid
    libusb_open_device_with_vid_pid.restype = ctypes.POINTER(struct_libusb_device_handle)
    libusb_open_device_with_vid_pid.argtypes = [ctypes.POINTER(struct_libusb_context), uint16_t, uint16_t]
except AttributeError:
    pass
try:
    libusb_set_interface_alt_setting = _libraries['libusb'].libusb_set_interface_alt_setting
    libusb_set_interface_alt_setting.restype = ctypes.c_int32
    libusb_set_interface_alt_setting.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_clear_halt = _libraries['libusb'].libusb_clear_halt
    libusb_clear_halt.restype = ctypes.c_int32
    libusb_clear_halt.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte]
except AttributeError:
    pass
try:
    libusb_reset_device = _libraries['libusb'].libusb_reset_device
    libusb_reset_device.restype = ctypes.c_int32
    libusb_reset_device.argtypes = [ctypes.POINTER(struct_libusb_device_handle)]
except AttributeError:
    pass
try:
    libusb_alloc_streams = _libraries['libusb'].libusb_alloc_streams
    libusb_alloc_streams.restype = ctypes.c_int32
    libusb_alloc_streams.argtypes = [ctypes.POINTER(struct_libusb_device_handle), uint32_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_free_streams = _libraries['libusb'].libusb_free_streams
    libusb_free_streams.restype = ctypes.c_int32
    libusb_free_streams.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    libusb_dev_mem_alloc = _libraries['libusb'].libusb_dev_mem_alloc
    libusb_dev_mem_alloc.restype = ctypes.POINTER(ctypes.c_ubyte)
    libusb_dev_mem_alloc.argtypes = [ctypes.POINTER(struct_libusb_device_handle), size_t]
except AttributeError:
    pass
try:
    libusb_dev_mem_free = _libraries['libusb'].libusb_dev_mem_free
    libusb_dev_mem_free.restype = ctypes.c_int32
    libusb_dev_mem_free.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), size_t]
except AttributeError:
    pass
try:
    libusb_kernel_driver_active = _libraries['libusb'].libusb_kernel_driver_active
    libusb_kernel_driver_active.restype = ctypes.c_int32
    libusb_kernel_driver_active.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_detach_kernel_driver = _libraries['libusb'].libusb_detach_kernel_driver
    libusb_detach_kernel_driver.restype = ctypes.c_int32
    libusb_detach_kernel_driver.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_attach_kernel_driver = _libraries['libusb'].libusb_attach_kernel_driver
    libusb_attach_kernel_driver.restype = ctypes.c_int32
    libusb_attach_kernel_driver.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_set_auto_detach_kernel_driver = _libraries['libusb'].libusb_set_auto_detach_kernel_driver
    libusb_set_auto_detach_kernel_driver.restype = ctypes.c_int32
    libusb_set_auto_detach_kernel_driver.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_control_transfer_get_data = _libraries['libusb'].libusb_control_transfer_get_data
    libusb_control_transfer_get_data.restype = ctypes.POINTER(ctypes.c_ubyte)
    libusb_control_transfer_get_data.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_control_transfer_get_setup = _libraries['libusb'].libusb_control_transfer_get_setup
    libusb_control_transfer_get_setup.restype = ctypes.POINTER(struct_libusb_control_setup)
    libusb_control_transfer_get_setup.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_fill_control_setup = _libraries['libusb'].libusb_fill_control_setup
    libusb_fill_control_setup.restype = None
    libusb_fill_control_setup.argtypes = [ctypes.POINTER(ctypes.c_ubyte), uint8_t, uint8_t, uint16_t, uint16_t, uint16_t]
except AttributeError:
    pass
try:
    libusb_alloc_transfer = _libraries['libusb'].libusb_alloc_transfer
    libusb_alloc_transfer.restype = ctypes.POINTER(struct_libusb_transfer)
    libusb_alloc_transfer.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_submit_transfer = _libraries['libusb'].libusb_submit_transfer
    libusb_submit_transfer.restype = ctypes.c_int32
    libusb_submit_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_cancel_transfer = _libraries['libusb'].libusb_cancel_transfer
    libusb_cancel_transfer.restype = ctypes.c_int32
    libusb_cancel_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_free_transfer = _libraries['libusb'].libusb_free_transfer
    libusb_free_transfer.restype = None
    libusb_free_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_transfer_set_stream_id = _libraries['libusb'].libusb_transfer_set_stream_id
    libusb_transfer_set_stream_id.restype = None
    libusb_transfer_set_stream_id.argtypes = [ctypes.POINTER(struct_libusb_transfer), uint32_t]
except AttributeError:
    pass
try:
    libusb_transfer_get_stream_id = _libraries['libusb'].libusb_transfer_get_stream_id
    libusb_transfer_get_stream_id.restype = uint32_t
    libusb_transfer_get_stream_id.argtypes = [ctypes.POINTER(struct_libusb_transfer)]
except AttributeError:
    pass
try:
    libusb_fill_control_transfer = _libraries['libusb'].libusb_fill_control_transfer
    libusb_fill_control_transfer.restype = None
    libusb_fill_control_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.POINTER(struct_libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), libusb_transfer_cb_fn, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_fill_bulk_transfer = _libraries['libusb'].libusb_fill_bulk_transfer
    libusb_fill_bulk_transfer.restype = None
    libusb_fill_bulk_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, libusb_transfer_cb_fn, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_fill_bulk_stream_transfer = _libraries['libusb'].libusb_fill_bulk_stream_transfer
    libusb_fill_bulk_stream_transfer.restype = None
    libusb_fill_bulk_stream_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, uint32_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, libusb_transfer_cb_fn, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_fill_interrupt_transfer = _libraries['libusb'].libusb_fill_interrupt_transfer
    libusb_fill_interrupt_transfer.restype = None
    libusb_fill_interrupt_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, libusb_transfer_cb_fn, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_fill_iso_transfer = _libraries['libusb'].libusb_fill_iso_transfer
    libusb_fill_iso_transfer.restype = None
    libusb_fill_iso_transfer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.c_int32, libusb_transfer_cb_fn, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_set_iso_packet_lengths = _libraries['libusb'].libusb_set_iso_packet_lengths
    libusb_set_iso_packet_lengths.restype = None
    libusb_set_iso_packet_lengths.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_get_iso_packet_buffer = _libraries['libusb'].libusb_get_iso_packet_buffer
    libusb_get_iso_packet_buffer.restype = ctypes.POINTER(ctypes.c_ubyte)
    libusb_get_iso_packet_buffer.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_get_iso_packet_buffer_simple = _libraries['libusb'].libusb_get_iso_packet_buffer_simple
    libusb_get_iso_packet_buffer_simple.restype = ctypes.POINTER(ctypes.c_ubyte)
    libusb_get_iso_packet_buffer_simple.argtypes = [ctypes.POINTER(struct_libusb_transfer), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_control_transfer = _libraries['libusb'].libusb_control_transfer
    libusb_control_transfer.restype = ctypes.c_int32
    libusb_control_transfer.argtypes = [ctypes.POINTER(struct_libusb_device_handle), uint8_t, uint8_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_ubyte), uint16_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_bulk_transfer = _libraries['libusb'].libusb_bulk_transfer
    libusb_bulk_transfer.restype = ctypes.c_int32
    libusb_bulk_transfer.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_interrupt_transfer = _libraries['libusb'].libusb_interrupt_transfer
    libusb_interrupt_transfer.restype = ctypes.c_int32
    libusb_interrupt_transfer.argtypes = [ctypes.POINTER(struct_libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    libusb_get_descriptor = _libraries['libusb'].libusb_get_descriptor
    libusb_get_descriptor.restype = ctypes.c_int32
    libusb_get_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device_handle), uint8_t, uint8_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_get_string_descriptor = _libraries['libusb'].libusb_get_string_descriptor
    libusb_get_string_descriptor.restype = ctypes.c_int32
    libusb_get_string_descriptor.argtypes = [ctypes.POINTER(struct_libusb_device_handle), uint8_t, uint16_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_get_string_descriptor_ascii = _libraries['libusb'].libusb_get_string_descriptor_ascii
    libusb_get_string_descriptor_ascii.restype = ctypes.c_int32
    libusb_get_string_descriptor_ascii.argtypes = [ctypes.POINTER(struct_libusb_device_handle), uint8_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    libusb_try_lock_events = _libraries['libusb'].libusb_try_lock_events
    libusb_try_lock_events.restype = ctypes.c_int32
    libusb_try_lock_events.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_lock_events = _libraries['libusb'].libusb_lock_events
    libusb_lock_events.restype = None
    libusb_lock_events.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_unlock_events = _libraries['libusb'].libusb_unlock_events
    libusb_unlock_events.restype = None
    libusb_unlock_events.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_event_handling_ok = _libraries['libusb'].libusb_event_handling_ok
    libusb_event_handling_ok.restype = ctypes.c_int32
    libusb_event_handling_ok.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_event_handler_active = _libraries['libusb'].libusb_event_handler_active
    libusb_event_handler_active.restype = ctypes.c_int32
    libusb_event_handler_active.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_interrupt_event_handler = _libraries['libusb'].libusb_interrupt_event_handler
    libusb_interrupt_event_handler.restype = None
    libusb_interrupt_event_handler.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_lock_event_waiters = _libraries['libusb'].libusb_lock_event_waiters
    libusb_lock_event_waiters.restype = None
    libusb_lock_event_waiters.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_unlock_event_waiters = _libraries['libusb'].libusb_unlock_event_waiters
    libusb_unlock_event_waiters.restype = None
    libusb_unlock_event_waiters.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
class struct_timeval(Structure):
    pass

struct_timeval._pack_ = 1 # source:False
struct_timeval._fields_ = [
    ('tv_sec', ctypes.c_int64),
    ('tv_usec', ctypes.c_int64),
]

try:
    libusb_wait_for_event = _libraries['libusb'].libusb_wait_for_event
    libusb_wait_for_event.restype = ctypes.c_int32
    libusb_wait_for_event.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError:
    pass
try:
    libusb_handle_events_timeout = _libraries['libusb'].libusb_handle_events_timeout
    libusb_handle_events_timeout.restype = ctypes.c_int32
    libusb_handle_events_timeout.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError:
    pass
try:
    libusb_handle_events_timeout_completed = _libraries['libusb'].libusb_handle_events_timeout_completed
    libusb_handle_events_timeout_completed.restype = ctypes.c_int32
    libusb_handle_events_timeout_completed.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_timeval), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    libusb_handle_events = _libraries['libusb'].libusb_handle_events
    libusb_handle_events.restype = ctypes.c_int32
    libusb_handle_events.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_handle_events_completed = _libraries['libusb'].libusb_handle_events_completed
    libusb_handle_events_completed.restype = ctypes.c_int32
    libusb_handle_events_completed.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    libusb_handle_events_locked = _libraries['libusb'].libusb_handle_events_locked
    libusb_handle_events_locked.restype = ctypes.c_int32
    libusb_handle_events_locked.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError:
    pass
try:
    libusb_pollfds_handle_timeouts = _libraries['libusb'].libusb_pollfds_handle_timeouts
    libusb_pollfds_handle_timeouts.restype = ctypes.c_int32
    libusb_pollfds_handle_timeouts.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_get_next_timeout = _libraries['libusb'].libusb_get_next_timeout
    libusb_get_next_timeout.restype = ctypes.c_int32
    libusb_get_next_timeout.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_timeval)]
except AttributeError:
    pass
class struct_libusb_pollfd(Structure):
    pass

struct_libusb_pollfd._pack_ = 1 # source:False
struct_libusb_pollfd._fields_ = [
    ('fd', ctypes.c_int32),
    ('events', ctypes.c_int16),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

libusb_pollfd_added_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int16, ctypes.POINTER(None))
libusb_pollfd_removed_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.POINTER(None))
try:
    libusb_get_pollfds = _libraries['libusb'].libusb_get_pollfds
    libusb_get_pollfds.restype = ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd))
    libusb_get_pollfds.argtypes = [ctypes.POINTER(struct_libusb_context)]
except AttributeError:
    pass
try:
    libusb_free_pollfds = _libraries['libusb'].libusb_free_pollfds
    libusb_free_pollfds.restype = None
    libusb_free_pollfds.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd))]
except AttributeError:
    pass
try:
    libusb_set_pollfd_notifiers = _libraries['libusb'].libusb_set_pollfd_notifiers
    libusb_set_pollfd_notifiers.restype = None
    libusb_set_pollfd_notifiers.argtypes = [ctypes.POINTER(struct_libusb_context), libusb_pollfd_added_cb, libusb_pollfd_removed_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
libusb_hotplug_callback_handle = ctypes.c_int32

# values for enumeration 'c__EA_libusb_hotplug_event'
c__EA_libusb_hotplug_event__enumvalues = {
    1: 'LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED',
    2: 'LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT',
}
LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED = 1
LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT = 2
c__EA_libusb_hotplug_event = ctypes.c_uint32 # enum
libusb_hotplug_event = c__EA_libusb_hotplug_event
libusb_hotplug_event__enumvalues = c__EA_libusb_hotplug_event__enumvalues

# values for enumeration 'c__EA_libusb_hotplug_flag'
c__EA_libusb_hotplug_flag__enumvalues = {
    1: 'LIBUSB_HOTPLUG_ENUMERATE',
}
LIBUSB_HOTPLUG_ENUMERATE = 1
c__EA_libusb_hotplug_flag = ctypes.c_uint32 # enum
libusb_hotplug_flag = c__EA_libusb_hotplug_flag
libusb_hotplug_flag__enumvalues = c__EA_libusb_hotplug_flag__enumvalues
libusb_hotplug_callback_fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_device), c__EA_libusb_hotplug_event, ctypes.POINTER(None))
try:
    libusb_hotplug_register_callback = _libraries['libusb'].libusb_hotplug_register_callback
    libusb_hotplug_register_callback.restype = ctypes.c_int32
    libusb_hotplug_register_callback.argtypes = [ctypes.POINTER(struct_libusb_context), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, libusb_hotplug_callback_fn, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    libusb_hotplug_deregister_callback = _libraries['libusb'].libusb_hotplug_deregister_callback
    libusb_hotplug_deregister_callback.restype = None
    libusb_hotplug_deregister_callback.argtypes = [ctypes.POINTER(struct_libusb_context), libusb_hotplug_callback_handle]
except AttributeError:
    pass
try:
    libusb_hotplug_get_user_data = _libraries['libusb'].libusb_hotplug_get_user_data
    libusb_hotplug_get_user_data.restype = ctypes.POINTER(None)
    libusb_hotplug_get_user_data.argtypes = [ctypes.POINTER(struct_libusb_context), libusb_hotplug_callback_handle]
except AttributeError:
    pass

# values for enumeration 'libusb_option'
libusb_option__enumvalues = {
    0: 'LIBUSB_OPTION_LOG_LEVEL',
    1: 'LIBUSB_OPTION_USE_USBDK',
    2: 'LIBUSB_OPTION_NO_DEVICE_DISCOVERY',
    3: 'LIBUSB_OPTION_MAX',
}
LIBUSB_OPTION_LOG_LEVEL = 0
LIBUSB_OPTION_USE_USBDK = 1
LIBUSB_OPTION_NO_DEVICE_DISCOVERY = 2
LIBUSB_OPTION_MAX = 3
libusb_option = ctypes.c_uint32 # enum
LIBUSB_OPTION_WEAK_AUTHORITY = LIBUSB_OPTION_NO_DEVICE_DISCOVERY # macro
try:
    libusb_set_option = _libraries['libusb'].libusb_set_option
    libusb_set_option.restype = ctypes.c_int32
    libusb_set_option.argtypes = [ctypes.POINTER(struct_libusb_context), libusb_option]
except AttributeError:
    pass
struct_libusb_transfer._pack_ = 1 # source:False
struct_libusb_transfer._fields_ = [
    ('dev_handle', ctypes.POINTER(struct_libusb_device_handle)),
    ('flags', ctypes.c_ubyte),
    ('endpoint', ctypes.c_ubyte),
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('timeout', ctypes.c_uint32),
    ('status', libusb_transfer_status),
    ('length', ctypes.c_int32),
    ('actual_length', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('callback', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_transfer))),
    ('user_data', ctypes.POINTER(None)),
    ('buffer', ctypes.POINTER(ctypes.c_ubyte)),
    ('num_iso_packets', ctypes.c_int32),
    ('iso_packet_desc', struct_libusb_iso_packet_descriptor * 0),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

__all__ = \
    ['LIBUSBX_API_VERSION', 'LIBUSB_API_VERSION',
    'LIBUSB_BM_LPM_SUPPORT', 'LIBUSB_BM_LTM_SUPPORT',
    'LIBUSB_BT_CONTAINER_ID', 'LIBUSB_BT_CONTAINER_ID_SIZE',
    'LIBUSB_BT_SS_USB_DEVICE_CAPABILITY',
    'LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE',
    'LIBUSB_BT_USB_2_0_EXTENSION', 'LIBUSB_BT_USB_2_0_EXTENSION_SIZE',
    'LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY', 'LIBUSB_CALL',
    'LIBUSB_CAP_HAS_CAPABILITY', 'LIBUSB_CAP_HAS_HID_ACCESS',
    'LIBUSB_CAP_HAS_HOTPLUG',
    'LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER',
    'LIBUSB_CLASS_APPLICATION', 'LIBUSB_CLASS_AUDIO',
    'LIBUSB_CLASS_COMM', 'LIBUSB_CLASS_CONTENT_SECURITY',
    'LIBUSB_CLASS_DATA', 'LIBUSB_CLASS_DIAGNOSTIC_DEVICE',
    'LIBUSB_CLASS_HID', 'LIBUSB_CLASS_HUB', 'LIBUSB_CLASS_IMAGE',
    'LIBUSB_CLASS_MASS_STORAGE', 'LIBUSB_CLASS_MISCELLANEOUS',
    'LIBUSB_CLASS_PERSONAL_HEALTHCARE', 'LIBUSB_CLASS_PER_INTERFACE',
    'LIBUSB_CLASS_PHYSICAL', 'LIBUSB_CLASS_PRINTER',
    'LIBUSB_CLASS_PTP', 'LIBUSB_CLASS_SMART_CARD',
    'LIBUSB_CLASS_VENDOR_SPEC', 'LIBUSB_CLASS_VIDEO',
    'LIBUSB_CLASS_WIRELESS', 'LIBUSB_DT_BOS',
    'LIBUSB_DT_BOS_MAX_SIZE', 'LIBUSB_DT_BOS_SIZE',
    'LIBUSB_DT_CONFIG', 'LIBUSB_DT_CONFIG_SIZE', 'LIBUSB_DT_DEVICE',
    'LIBUSB_DT_DEVICE_CAPABILITY', 'LIBUSB_DT_DEVICE_CAPABILITY_SIZE',
    'LIBUSB_DT_DEVICE_SIZE', 'LIBUSB_DT_ENDPOINT',
    'LIBUSB_DT_ENDPOINT_AUDIO_SIZE', 'LIBUSB_DT_ENDPOINT_SIZE',
    'LIBUSB_DT_HID', 'LIBUSB_DT_HUB', 'LIBUSB_DT_HUB_NONVAR_SIZE',
    'LIBUSB_DT_INTERFACE', 'LIBUSB_DT_INTERFACE_SIZE',
    'LIBUSB_DT_PHYSICAL', 'LIBUSB_DT_REPORT',
    'LIBUSB_DT_SS_ENDPOINT_COMPANION',
    'LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE', 'LIBUSB_DT_STRING',
    'LIBUSB_DT_SUPERSPEED_HUB', 'LIBUSB_ENDPOINT_ADDRESS_MASK',
    'LIBUSB_ENDPOINT_DIR_MASK', 'LIBUSB_ENDPOINT_IN',
    'LIBUSB_ENDPOINT_OUT', 'LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK',
    'LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL',
    'LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT',
    'LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS',
    'LIBUSB_ERROR_ACCESS', 'LIBUSB_ERROR_BUSY', 'LIBUSB_ERROR_COUNT',
    'LIBUSB_ERROR_INTERRUPTED', 'LIBUSB_ERROR_INVALID_PARAM',
    'LIBUSB_ERROR_IO', 'LIBUSB_ERROR_NOT_FOUND',
    'LIBUSB_ERROR_NOT_SUPPORTED', 'LIBUSB_ERROR_NO_DEVICE',
    'LIBUSB_ERROR_NO_MEM', 'LIBUSB_ERROR_OTHER',
    'LIBUSB_ERROR_OVERFLOW', 'LIBUSB_ERROR_PIPE',
    'LIBUSB_ERROR_TIMEOUT', 'LIBUSB_FULL_SPEED_OPERATION', 'LIBUSB_H',
    'LIBUSB_HIGH_SPEED_OPERATION', 'LIBUSB_HOTPLUG_ENUMERATE',
    'LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED',
    'LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT', 'LIBUSB_HOTPLUG_MATCH_ANY',
    'LIBUSB_HOTPLUG_NO_FLAGS', 'LIBUSB_ISO_SYNC_TYPE_ADAPTIVE',
    'LIBUSB_ISO_SYNC_TYPE_ASYNC', 'LIBUSB_ISO_SYNC_TYPE_MASK',
    'LIBUSB_ISO_SYNC_TYPE_NONE', 'LIBUSB_ISO_SYNC_TYPE_SYNC',
    'LIBUSB_ISO_USAGE_TYPE_DATA', 'LIBUSB_ISO_USAGE_TYPE_FEEDBACK',
    'LIBUSB_ISO_USAGE_TYPE_IMPLICIT', 'LIBUSB_ISO_USAGE_TYPE_MASK',
    'LIBUSB_LOG_CB_CONTEXT', 'LIBUSB_LOG_CB_GLOBAL',
    'LIBUSB_LOG_LEVEL_DEBUG', 'LIBUSB_LOG_LEVEL_ERROR',
    'LIBUSB_LOG_LEVEL_INFO', 'LIBUSB_LOG_LEVEL_NONE',
    'LIBUSB_LOG_LEVEL_WARNING', 'LIBUSB_LOW_SPEED_OPERATION',
    'LIBUSB_OPTION_LOG_LEVEL', 'LIBUSB_OPTION_MAX',
    'LIBUSB_OPTION_NO_DEVICE_DISCOVERY', 'LIBUSB_OPTION_USE_USBDK',
    'LIBUSB_OPTION_WEAK_AUTHORITY', 'LIBUSB_RECIPIENT_DEVICE',
    'LIBUSB_RECIPIENT_ENDPOINT', 'LIBUSB_RECIPIENT_INTERFACE',
    'LIBUSB_RECIPIENT_OTHER', 'LIBUSB_REQUEST_CLEAR_FEATURE',
    'LIBUSB_REQUEST_GET_CONFIGURATION',
    'LIBUSB_REQUEST_GET_DESCRIPTOR', 'LIBUSB_REQUEST_GET_INTERFACE',
    'LIBUSB_REQUEST_GET_STATUS', 'LIBUSB_REQUEST_SET_ADDRESS',
    'LIBUSB_REQUEST_SET_CONFIGURATION',
    'LIBUSB_REQUEST_SET_DESCRIPTOR', 'LIBUSB_REQUEST_SET_FEATURE',
    'LIBUSB_REQUEST_SET_INTERFACE', 'LIBUSB_REQUEST_SET_SEL',
    'LIBUSB_REQUEST_SYNCH_FRAME', 'LIBUSB_REQUEST_TYPE_CLASS',
    'LIBUSB_REQUEST_TYPE_RESERVED', 'LIBUSB_REQUEST_TYPE_STANDARD',
    'LIBUSB_REQUEST_TYPE_VENDOR', 'LIBUSB_SET_ISOCH_DELAY',
    'LIBUSB_SPEED_FULL', 'LIBUSB_SPEED_HIGH', 'LIBUSB_SPEED_LOW',
    'LIBUSB_SPEED_SUPER', 'LIBUSB_SPEED_SUPER_PLUS',
    'LIBUSB_SPEED_UNKNOWN', 'LIBUSB_SUCCESS',
    'LIBUSB_SUPER_SPEED_OPERATION', 'LIBUSB_TRANSFER_ADD_ZERO_PACKET',
    'LIBUSB_TRANSFER_CANCELLED', 'LIBUSB_TRANSFER_COMPLETED',
    'LIBUSB_TRANSFER_ERROR', 'LIBUSB_TRANSFER_FREE_BUFFER',
    'LIBUSB_TRANSFER_FREE_TRANSFER', 'LIBUSB_TRANSFER_NO_DEVICE',
    'LIBUSB_TRANSFER_OVERFLOW', 'LIBUSB_TRANSFER_SHORT_NOT_OK',
    'LIBUSB_TRANSFER_STALL', 'LIBUSB_TRANSFER_TIMED_OUT',
    'LIBUSB_TRANSFER_TYPE_BULK', 'LIBUSB_TRANSFER_TYPE_BULK_STREAM',
    'LIBUSB_TRANSFER_TYPE_CONTROL', 'LIBUSB_TRANSFER_TYPE_INTERRUPT',
    'LIBUSB_TRANSFER_TYPE_ISOCHRONOUS', 'LIBUSB_TRANSFER_TYPE_MASK',
    'ZERO_SIZED_ARRAY', 'c__EA_libusb_hotplug_event',
    'c__EA_libusb_hotplug_flag', 'intptr_t', 'libusb_alloc_streams',
    'libusb_alloc_transfer', 'libusb_attach_kernel_driver',
    'libusb_bos_type', 'libusb_bulk_transfer',
    'libusb_cancel_transfer', 'libusb_capability',
    'libusb_claim_interface', 'libusb_class_code',
    'libusb_clear_halt', 'libusb_close', 'libusb_context',
    'libusb_control_transfer', 'libusb_control_transfer_get_data',
    'libusb_control_transfer_get_setup', 'libusb_cpu_to_le16',
    'libusb_descriptor_type', 'libusb_detach_kernel_driver',
    'libusb_dev_mem_alloc', 'libusb_dev_mem_free', 'libusb_device',
    'libusb_device_handle', 'libusb_endpoint_direction',
    'libusb_endpoint_transfer_type', 'libusb_error',
    'libusb_error_name', 'libusb_event_handler_active',
    'libusb_event_handling_ok', 'libusb_exit',
    'libusb_fill_bulk_stream_transfer', 'libusb_fill_bulk_transfer',
    'libusb_fill_control_setup', 'libusb_fill_control_transfer',
    'libusb_fill_interrupt_transfer', 'libusb_fill_iso_transfer',
    'libusb_free_bos_descriptor', 'libusb_free_config_descriptor',
    'libusb_free_container_id_descriptor', 'libusb_free_device_list',
    'libusb_free_pollfds',
    'libusb_free_ss_endpoint_companion_descriptor',
    'libusb_free_ss_usb_device_capability_descriptor',
    'libusb_free_streams', 'libusb_free_transfer',
    'libusb_free_usb_2_0_extension_descriptor',
    'libusb_get_active_config_descriptor',
    'libusb_get_bos_descriptor', 'libusb_get_bus_number',
    'libusb_get_config_descriptor',
    'libusb_get_config_descriptor_by_value',
    'libusb_get_configuration', 'libusb_get_container_id_descriptor',
    'libusb_get_descriptor', 'libusb_get_device',
    'libusb_get_device_address', 'libusb_get_device_descriptor',
    'libusb_get_device_list', 'libusb_get_device_speed',
    'libusb_get_iso_packet_buffer',
    'libusb_get_iso_packet_buffer_simple',
    'libusb_get_max_iso_packet_size', 'libusb_get_max_packet_size',
    'libusb_get_next_timeout', 'libusb_get_parent',
    'libusb_get_pollfds', 'libusb_get_port_number',
    'libusb_get_port_numbers', 'libusb_get_port_path',
    'libusb_get_ss_endpoint_companion_descriptor',
    'libusb_get_ss_usb_device_capability_descriptor',
    'libusb_get_string_descriptor',
    'libusb_get_string_descriptor_ascii',
    'libusb_get_usb_2_0_extension_descriptor', 'libusb_get_version',
    'libusb_handle_events', 'libusb_handle_events_completed',
    'libusb_handle_events_locked', 'libusb_handle_events_timeout',
    'libusb_handle_events_timeout_completed', 'libusb_has_capability',
    'libusb_hotplug_callback_fn', 'libusb_hotplug_callback_handle',
    'libusb_hotplug_deregister_callback', 'libusb_hotplug_event',
    'libusb_hotplug_event__enumvalues', 'libusb_hotplug_flag',
    'libusb_hotplug_flag__enumvalues', 'libusb_hotplug_get_user_data',
    'libusb_hotplug_register_callback', 'libusb_init',
    'libusb_interrupt_event_handler', 'libusb_interrupt_transfer',
    'libusb_iso_sync_type', 'libusb_iso_usage_type',
    'libusb_kernel_driver_active', 'libusb_le16_to_cpu',
    'libusb_lock_event_waiters', 'libusb_lock_events',
    'libusb_log_cb', 'libusb_log_cb_mode', 'libusb_log_level',
    'libusb_open', 'libusb_open_device_with_vid_pid', 'libusb_option',
    'libusb_pollfd_added_cb', 'libusb_pollfd_removed_cb',
    'libusb_pollfds_handle_timeouts', 'libusb_ref_device',
    'libusb_release_interface', 'libusb_request_recipient',
    'libusb_request_type', 'libusb_reset_device',
    'libusb_set_auto_detach_kernel_driver',
    'libusb_set_configuration', 'libusb_set_debug',
    'libusb_set_interface_alt_setting',
    'libusb_set_iso_packet_lengths', 'libusb_set_log_cb',
    'libusb_set_option', 'libusb_set_pollfd_notifiers',
    'libusb_setlocale', 'libusb_speed',
    'libusb_ss_usb_device_capability_attributes',
    'libusb_standard_request', 'libusb_strerror',
    'libusb_submit_transfer', 'libusb_supported_speed',
    'libusb_transfer_cb_fn', 'libusb_transfer_flags',
    'libusb_transfer_get_stream_id', 'libusb_transfer_set_stream_id',
    'libusb_transfer_status', 'libusb_transfer_type',
    'libusb_try_lock_events', 'libusb_unlock_event_waiters',
    'libusb_unlock_events', 'libusb_unref_device',
    'libusb_usb_2_0_extension_attributes', 'libusb_wait_for_event',
    'libusb_wrap_sys_device', 'size_t', 'ssize_t',
    'struct_libusb_bos_descriptor',
    'struct_libusb_bos_dev_capability_descriptor',
    'struct_libusb_config_descriptor',
    'struct_libusb_container_id_descriptor', 'struct_libusb_context',
    'struct_libusb_control_setup', 'struct_libusb_device',
    'struct_libusb_device_descriptor', 'struct_libusb_device_handle',
    'struct_libusb_endpoint_descriptor', 'struct_libusb_interface',
    'struct_libusb_interface_descriptor',
    'struct_libusb_iso_packet_descriptor', 'struct_libusb_pollfd',
    'struct_libusb_ss_endpoint_companion_descriptor',
    'struct_libusb_ss_usb_device_capability_descriptor',
    'struct_libusb_transfer',
    'struct_libusb_usb_2_0_extension_descriptor',
    'struct_libusb_version', 'struct_timeval', 'uint16_t', 'uint32_t',
    'uint8_t']
