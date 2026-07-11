"""ctypes interface to the V4L2 encoder API, including the msm_vidc vendor extensions.

Constants and struct layouts were extracted from the agnos-kernel-sdm845 UAPI headers
(linux/videodev2.h, linux/v4l2-controls.h). The ioctl numbers are computed from the
struct sizes and asserted against the values from the kernel headers, so a struct
layout mistake fails at import time.
"""
import ctypes
import errno
import os

u8 = ctypes.c_uint8
u16 = ctypes.c_uint16
u32 = ctypes.c_uint32
s32 = ctypes.c_int32
u64 = ctypes.c_uint64

_libc = ctypes.CDLL(None, use_errno=True)
_libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
_libc.ioctl.restype = ctypes.c_int


def safe_ioctl(fd: int, request: int, arg) -> int:
  # keep trying if interrupted by a signal
  for _ in range(100):
    ret = _libc.ioctl(fd, request, ctypes.byref(arg))
    if ret != -1 or ctypes.get_errno() != errno.EINTR:
      break
  if ret == -1:
    err = ctypes.get_errno()
    raise OSError(err, f"ioctl {request:#x} failed: {os.strerror(err)}")
  return ret


# *** ioctl number macros ***

_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS = 8, 8, 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE, _IOC_READ = 1, 2


def _IOC(direction: int, typ: str, nr: int, size: int) -> int:
  return (direction << _IOC_DIRSHIFT) | (ord(typ) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)


def _IOW(typ, nr, t):
  return _IOC(_IOC_WRITE, typ, nr, ctypes.sizeof(t))


def _IOR(typ, nr, t):
  return _IOC(_IOC_READ, typ, nr, ctypes.sizeof(t))


def _IOWR(typ, nr, t):
  return _IOC(_IOC_READ | _IOC_WRITE, typ, nr, ctypes.sizeof(t))


# *** structs ***

class timeval(ctypes.Structure):
  _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_capability(ctypes.Structure):
  _fields_ = [
    ("driver", ctypes.c_char * 16),
    ("card", ctypes.c_char * 32),
    ("bus_info", ctypes.c_char * 32),
    ("version", u32),
    ("capabilities", u32),
    ("device_caps", u32),
    ("reserved", u32 * 3),
  ]


class v4l2_plane_pix_format(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("sizeimage", u32),
    ("bytesperline", u32),
    ("reserved", u16 * 6),
  ]


class v4l2_pix_format_mplane(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("width", u32),
    ("height", u32),
    ("pixelformat", u32),
    ("field", u32),
    ("colorspace", u32),
    ("plane_fmt", v4l2_plane_pix_format * 8),
    ("num_planes", u8),
    ("flags", u8),
    ("ycbcr_enc", u8),
    ("quantization", u8),
    ("xfer_func", u8),
    ("reserved", u8 * 7),
  ]


class v4l2_format(ctypes.Structure):
  class _fmt(ctypes.Union):
    _fields_ = [
      ("pix_mp", v4l2_pix_format_mplane),
      ("raw_data", u64 * 25),  # u64 to get the 8-byte alignment of the pointers in the C union
    ]
  _fields_ = [
    ("type", u32),
    ("fmt", _fmt),
  ]


class v4l2_fract(ctypes.Structure):
  _fields_ = [("numerator", u32), ("denominator", u32)]


class v4l2_outputparm(ctypes.Structure):
  _fields_ = [
    ("capability", u32),
    ("outputmode", u32),
    ("timeperframe", v4l2_fract),
    ("extendedmode", u32),
    ("writebuffers", u32),
    ("reserved", u32 * 4),
  ]


class v4l2_streamparm(ctypes.Structure):
  class _parm(ctypes.Union):
    _fields_ = [
      ("output", v4l2_outputparm),
      ("raw_data", u8 * 200),
    ]
  _fields_ = [
    ("type", u32),
    ("parm", _parm),
  ]


class v4l2_requestbuffers(ctypes.Structure):
  _fields_ = [
    ("count", u32),
    ("type", u32),
    ("memory", u32),
    ("reserved", u32 * 2),
  ]


class v4l2_timecode(ctypes.Structure):
  _fields_ = [
    ("type", u32),
    ("flags", u32),
    ("frames", u8),
    ("seconds", u8),
    ("minutes", u8),
    ("hours", u8),
    ("userbits", u8 * 4),
  ]


class v4l2_plane(ctypes.Structure):
  class _m(ctypes.Union):
    _fields_ = [
      ("mem_offset", u32),
      ("userptr", ctypes.c_ulong),
      ("fd", s32),
    ]
  _fields_ = [
    ("bytesused", u32),
    ("length", u32),
    ("m", _m),
    ("data_offset", u32),
    ("reserved", u32 * 11),
  ]


class v4l2_buffer(ctypes.Structure):
  class _m(ctypes.Union):
    _fields_ = [
      ("offset", u32),
      ("userptr", ctypes.c_ulong),
      ("planes", ctypes.POINTER(v4l2_plane)),
      ("fd", s32),
    ]
  _fields_ = [
    ("index", u32),
    ("type", u32),
    ("bytesused", u32),
    ("flags", u32),
    ("field", u32),
    ("timestamp", timeval),
    ("timecode", v4l2_timecode),
    ("sequence", u32),
    ("memory", u32),
    ("m", _m),
    ("length", u32),
    ("reserved2", u32),
    ("reserved", u32),
  ]


class v4l2_control(ctypes.Structure):
  _fields_ = [("id", u32), ("value", s32)]


class v4l2_encoder_cmd(ctypes.Structure):
  _fields_ = [
    ("cmd", u32),
    ("flags", u32),
    ("raw_data", u32 * 8),
  ]


assert ctypes.sizeof(v4l2_capability) == 104
assert ctypes.sizeof(v4l2_format) == 208 and v4l2_format.fmt.offset == 8
assert ctypes.sizeof(v4l2_pix_format_mplane) == 192 and v4l2_pix_format_mplane.plane_fmt.offset == 20
assert ctypes.sizeof(v4l2_streamparm) == 204 and v4l2_streamparm.parm.offset == 4
assert ctypes.sizeof(v4l2_requestbuffers) == 20
assert ctypes.sizeof(v4l2_buffer) == 88 and v4l2_buffer.m.offset == 64 and v4l2_buffer.timestamp.offset == 24
assert ctypes.sizeof(v4l2_plane) == 64 and v4l2_plane.reserved.offset == 20
assert ctypes.sizeof(v4l2_control) == 8
assert ctypes.sizeof(v4l2_encoder_cmd) == 40

# *** ioctls ***

VIDIOC_QUERYCAP = _IOR('V', 0, v4l2_capability)
VIDIOC_S_FMT = _IOWR('V', 5, v4l2_format)
VIDIOC_REQBUFS = _IOWR('V', 8, v4l2_requestbuffers)
VIDIOC_QBUF = _IOWR('V', 15, v4l2_buffer)
VIDIOC_DQBUF = _IOWR('V', 17, v4l2_buffer)
VIDIOC_STREAMON = _IOW('V', 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW('V', 19, ctypes.c_int)
VIDIOC_S_PARM = _IOWR('V', 22, v4l2_streamparm)
VIDIOC_S_CTRL = _IOWR('V', 28, v4l2_control)
VIDIOC_ENCODER_CMD = _IOWR('V', 77, v4l2_encoder_cmd)

assert VIDIOC_QUERYCAP == 0x80685600
assert VIDIOC_S_FMT == 0xc0d05605
assert VIDIOC_REQBUFS == 0xc0145608
assert VIDIOC_QBUF == 0xc058560f
assert VIDIOC_DQBUF == 0xc0585611
assert VIDIOC_STREAMON == 0x40045612
assert VIDIOC_STREAMOFF == 0x40045613
assert VIDIOC_S_PARM == 0xc0cc5616
assert VIDIOC_S_CTRL == 0xc008561c
assert VIDIOC_ENCODER_CMD == 0xc028564d


def fourcc(s: str) -> int:
  return ord(s[0]) | (ord(s[1]) << 8) | (ord(s[2]) << 16) | (ord(s[3]) << 24)


# *** constants ***

V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE = 10
V4L2_MEMORY_USERPTR = 2
V4L2_FIELD_ANY = 0
V4L2_COLORSPACE_DEFAULT = 0
V4L2_COLORSPACE_470_SYSTEM_BG = 6

V4L2_PIX_FMT_HEVC = fourcc('HEVC')
V4L2_PIX_FMT_H264 = fourcc('H264')
V4L2_PIX_FMT_NV12 = fourcc('NV12')

V4L2_BUF_FLAG_KEYFRAME = 0x8
V4L2_BUF_FLAG_TIMESTAMP_COPY = 0x4000

V4L2_ENC_CMD_STOP = 1

# qcom extensions
V4L2_QCOM_BUF_FLAG_CODECCONFIG = 0x00020000
V4L2_QCOM_BUF_FLAG_EOS = 0x02000000

# *** controls ***

V4L2_CID_MPEG_VIDEO_BITRATE = 0x9909cf
V4L2_CID_MPEG_VIDEO_HEADER_MODE = 0x9909d8
V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE = 0x0
V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE = 0x9909dd
V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE = 0x990a65
V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC = 0x1
V4L2_CID_MPEG_VIDEO_H264_LEVEL = 0x990a67
V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN = 0x11
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA = 0x990a68
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA = 0x990a69
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE = 0x990a6a
V4L2_CID_MPEG_VIDEO_H264_PROFILE = 0x990a6b
V4L2_MPEG_VIDEO_H264_PROFILE_HIGH = 0x4

# msm_vidc vendor controls
V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD = 0x992005
V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES = 0x992006
V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES = 0x992007
V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME = 0x992008
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL = 0x992009
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR = 0x2
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL = 0x99200b
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0 = 0x0
V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO = 0x992013
V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED = 0x1
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE = 0x992028
V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN = 0x0
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL = 0x992029
V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5 = 0xf
V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY = 0x992034
V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE = 0x1
