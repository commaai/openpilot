import ctypes as C

# Minimal V4L2 constants and structs used by V4L2Encoder

# ioctls
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_S_PARM = 0xC0CC5616
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_S_CTRL = 0xC008561C
VIDIOC_ENCODER_CMD = 0xC028564D

# buffer types
V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE = 9
V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 10

# memory
V4L2_MEMORY_USERPTR = 2
V4L2_MEMORY_MMAP = 1

# pixel formats
V4L2_PIX_FMT_NV12 = 0x3231564E  # 'NV12'
V4L2_PIX_FMT_H264 = 0x34363248  # 'H264'
V4L2_PIX_FMT_HEVC = 0x43564548  # 'HEVC'

# controls (subset)
V4L2_CID_MPEG_VIDEO_BITRATE = 0x0099091B
V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES = 0x0093E00F
V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES = 0x0093E010
V4L2_CID_MPEG_VIDEO_HEADER_MODE = 0x00990926
V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE = 1
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL = 0x0093E02C
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR = 2
V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY = 0x0093E02D
V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE = 0
V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD = 0x0093E01D

# H264 specific
V4L2_CID_MPEG_VIDEO_H264_PROFILE = 0x00990930
V4L2_MPEG_VIDEO_H264_PROFILE_HIGH = 5
V4L2_CID_MPEG_VIDEO_H264_LEVEL = 0x0099093B
V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN = 0
V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE = 0x0099092C
V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC = 1
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL = 0x0093E01F
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0 = 0
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE = 0x0099092E

# HEVC specific
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE = 0x0093E060
V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN = 1
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL = 0x0093E063
V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5 = 0x80000005
V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO = 0x0093E066
V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED = 1

# flags
V4L2_BUF_FLAG_TIMESTAMP_COPY = 0x4000
V4L2_BUF_FLAG_KEYFRAME = 0x0008
V4L2_QCOM_BUF_FLAG_CODECCONFIG = 0x00020000
V4L2_QCOM_BUF_FLAG_EOS = 0x02000000

# encoder commands
V4L2_ENC_CMD_STOP = 0x00000001


class v4l2_capability(C.Structure):
  _fields_ = [
    ("driver", C.c_uint8 * 16),
    ("card", C.c_uint8 * 32),
    ("bus_info", C.c_uint8 * 32),
    ("version", C.c_uint32),
    ("capabilities", C.c_uint32),
    ("device_caps", C.c_uint32),
    ("reserved", C.c_uint32 * 3),
  ]


class v4l2_fract(C.Structure):
  _fields_ = [("numerator", C.c_uint32), ("denominator", C.c_uint32)]


class v4l2_outputparm(C.Structure):
  _fields_ = [
    ("capability", C.c_uint32),
    ("outputmode", C.c_uint32),
    ("capturemode", C.c_uint32),
    ("timeperframe", v4l2_fract),
    ("extendedmode", C.c_uint32),
    ("readbuffers", C.c_uint32),
    ("reserved", C.c_uint32 * 8),
  ]


class v4l2_streamparm_parm(C.Union):
  _fields_ = [("output", v4l2_outputparm)]


class v4l2_streamparm(C.Structure):
  _fields_ = [("type", C.c_uint32), ("parm", v4l2_streamparm_parm)]


class v4l2_pix_format_mplane(C.Structure):
  _fields_ = [
    ("width", C.c_uint32),
    ("height", C.c_uint32),
    ("pixelformat", C.c_uint32),
    ("field", C.c_uint32),
    ("colorspace", C.c_uint32),
    ("plane_fmt", (C.c_uint32 * 5) * 8),
    ("num_planes", C.c_uint8),
    ("flags", C.c_uint8),
    ("ycbcr_enc", C.c_uint8),
    ("quantization", C.c_uint8),
    ("xfer_func", C.c_uint8),
    ("_pad", C.c_uint8 * 7),
  ]


class v4l2_format_union(C.Union):
  _fields_ = [("pix_mp", v4l2_pix_format_mplane)]


class v4l2_format(C.Structure):
  _fields_ = [("type", C.c_uint32), ("fmt", v4l2_format_union)]


class v4l2_requestbuffers(C.Structure):
  _fields_ = [("type", C.c_uint32), ("memory", C.c_uint32), ("count", C.c_uint32), ("_reserved", C.c_uint32 * 1)]


class v4l2_control(C.Structure):
  _fields_ = [("id", C.c_uint32), ("value", C.c_int32)]


class timeval(C.Structure):
  _fields_ = [("tv_sec", C.c_long), ("tv_usec", C.c_long)]


class v4l2_plane_m(C.Union):
  _fields_ = [("mem_offset", C.c_uint32), ("userptr", C.c_ulong), ("fd", C.c_int32)]


class v4l2_plane(C.Structure):
  _fields_ = [
    ("bytesused", C.c_uint32),
    ("length", C.c_uint32),
    ("m", v4l2_plane_m),
    ("data_offset", C.c_uint32),
    ("reserved", C.c_uint32 * 11),
  ]


class v4l2_buffer_m(C.Union):
  _fields_ = [("planes", C.POINTER(v4l2_plane))]


class v4l2_buffer(C.Structure):
  _fields_ = [
    ("type", C.c_uint32),
    ("index", C.c_uint32),
    ("bytesused", C.c_uint32),
    ("flags", C.c_uint32),
    ("field", C.c_uint32),
    ("timestamp", timeval),
    ("timecode", C.c_uint32 * 8),
    ("sequence", C.c_uint32),
    ("memory", C.c_uint32),
    ("m", v4l2_buffer_m),
    ("length", C.c_uint32),
    ("reserved2", C.c_uint32),
    ("_reserved", C.c_uint32),
  ]


class v4l2_encoder_cmd(C.Structure):
  _fields_ = [("cmd", C.c_uint32), ("flags", C.c_uint32), ("raw", C.c_uint32 * 16)]
