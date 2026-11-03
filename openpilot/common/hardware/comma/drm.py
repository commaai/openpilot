import array
import ctypes
import os
import socket

# AGNOS's magic.py holds DRM master and hands out dups of it
DRM_FD_SOCK = "/tmp/drmfd.sock"
DRM_MODE_OBJECT_CONNECTOR = 0xc0c0c0c0
DRM_MODE_DPMS_ON = 0
DRM_MODE_DPMS_OFF = 3


class _Resources(ctypes.Structure):
  _fields_ = [("count_fbs", ctypes.c_int), ("fbs", ctypes.POINTER(ctypes.c_uint32)),
              ("count_crtcs", ctypes.c_int), ("crtcs", ctypes.POINTER(ctypes.c_uint32)),
              ("count_connectors", ctypes.c_int), ("connectors", ctypes.POINTER(ctypes.c_uint32))]

class _ObjectProperties(ctypes.Structure):
  _fields_ = [("count_props", ctypes.c_uint32), ("props", ctypes.POINTER(ctypes.c_uint32)),
              ("prop_values", ctypes.POINTER(ctypes.c_uint64))]

class _Property(ctypes.Structure):
  _fields_ = [("prop_id", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("name", ctypes.c_char * 32)]


def _get_drm_fd() -> int:
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    s.connect(DRM_FD_SOCK)
    _, anc, _, _ = s.recvmsg(1, socket.CMSG_LEN(array.array("i").itemsize))
    return array.array("i", anc[0][2])[0]


def set_dpms(on: bool) -> None:
  """Power the panel and DSI link on or off. The backlight alone leaves the panel scanning out."""
  drm = ctypes.CDLL("libdrm.so.2")
  drm.drmModeGetResources.restype = ctypes.POINTER(_Resources)
  drm.drmModeObjectGetProperties.restype = ctypes.POINTER(_ObjectProperties)
  drm.drmModeGetProperty.restype = ctypes.POINTER(_Property)

  fd = _get_drm_fd()
  try:
    res = drm.drmModeGetResources(fd)
    for i in range(res.contents.count_connectors):
      conn = res.contents.connectors[i]
      props = drm.drmModeObjectGetProperties(fd, conn, DRM_MODE_OBJECT_CONNECTOR)
      for j in range(props.contents.count_props):
        prop = drm.drmModeGetProperty(fd, props.contents.props[j])
        if prop.contents.name == b"DPMS":
          drm.drmModeConnectorSetProperty(fd, conn, prop.contents.prop_id, ctypes.c_uint64(DRM_MODE_DPMS_ON if on else DRM_MODE_DPMS_OFF))
        drm.drmModeFreeProperty(prop)
      drm.drmModeFreeObjectProperties(props)
    drm.drmModeFreeResources(res)
  finally:
    os.close(fd)
