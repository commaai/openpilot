// Portable raster core for the raylib-compatible MICI CPU backend.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <msm_drm.h>
#endif

typedef struct {
  uint8_t *pixels;
  int width;
  int height;
  int stride;
  int clip_x0;
  int clip_y0;
  int clip_x1;
  int clip_y1;
} Surface;

typedef struct {
  float x;
  float y;
} Point;

typedef struct {
  const Surface *surface;
  float source_x;
  float source_y;
  float source_width;
  float source_height;
  int destination_x;
  int destination_y;
  int destination_width;
  int destination_height;
} BlitItem;

void sr_triangle(Surface *s, Point a, Point b, Point c, uint32_t color);

#ifdef __linux__
typedef struct {
  uint32_t handle;
  uint32_t pitch;
  uint64_t size;
  uint32_t fb_id;
  uint8_t *map;
  int cached;
  int cpu_prepared;
} DrmBuffer;

// camerad currently publishes 18 VisionIPC buffers. Leave a little headroom
// for reconnects so every live DMA-BUF can remain imported for the session.
#define MAX_CAMERA_BUFFERS 24
typedef struct {
  dev_t device;
  ino_t inode;
  uint32_t handle;
  uint32_t fb_id;
  uint64_t last_used;
} CameraBuffer;

typedef struct {
  int fd;
  int server_sock;
  uint32_t connector_id;
  uint32_t crtc_id;
  drmModeCrtc *original_crtc;
  drmModeModeInfo mode;
  DrmBuffer buffers[2];
  int front;
  int initialized;
  int presented;
  int clockwise;
  int mdp_ui;
  int mdp_camera;
  uint32_t primary_plane_id;
  uint32_t camera_plane_id;
  uint32_t plane_fb_id_prop;
  uint32_t plane_crtc_id_prop;
  uint32_t plane_crtc_x_prop;
  uint32_t plane_crtc_y_prop;
  uint32_t plane_crtc_w_prop;
  uint32_t plane_crtc_h_prop;
  uint32_t plane_src_x_prop;
  uint32_t plane_src_y_prop;
  uint32_t plane_src_w_prop;
  uint32_t plane_src_h_prop;
  uint32_t plane_rotation_prop;
  uint32_t plane_zpos_prop;
  uint32_t plane_alpha_prop;
  uint32_t plane_blend_prop;
  uint32_t plane_csc_prop;
  uint32_t plane_rot_dst_x_prop;
  uint32_t plane_rot_dst_y_prop;
  uint32_t plane_rot_dst_w_prop;
  uint32_t plane_rot_dst_h_prop;
  CameraBuffer camera_buffers[MAX_CAMERA_BUFFERS];
  int camera_buffer_count;
  uint64_t camera_frame;
  uint32_t displayed_camera_fb_id;
  int camera_active;
  uint32_t camera_fb_id;
  int camera_src_x;
  int camera_src_y;
  int camera_src_w;
  int camera_src_h;
  int camera_dst_x;
  int camera_dst_y;
  int camera_dst_w;
  int camera_dst_h;
  int camera_flip_x;
  int camera_alpha;
  int camera_engaged;
  int direct_render;
  int color_correction;
  float color_contribution[3][3][256];
  uint8_t color_gamma[4096];
  double last_copy_ms;
  struct timespec present_deadline;
} DrmState;

static DrmState drm_state = { .fd = -1, .server_sock = -1 };

// Downstream SDE's csc_v1 property takes a userspace pointer to this payload,
// rather than a DRM property blob. Coefficients are S31.32. This combines the
// kernel's limited-range BT.601 YUV conversion with the engaged shader's
// linear 20%-saturation and +20%-contrast stages. The shader's final nonlinear
// gamma cannot be represented by a VIG CSC.
typedef struct {
  int64_t ctm_coeff[9];
  uint32_t pre_bias[3];
  uint32_t post_bias[3];
  uint32_t pre_clamp[6];
  uint32_t post_clamp[6];
} SdeDrmCscV1;

static const SdeDrmCscV1 engaged_camera_csc = {
  .ctm_coeff = {
    91546LL * 65536, -28LL * 65536, 25109LL * 65536,
    91546LL * 65536, -6202LL * 65536, -12768LL * 65536,
    91546LL * 65536, 31706LL * 65536, 11LL * 65536,
  },
  .pre_bias = {0xfff0, 0xff80, 0xff80},
  .post_bias = {(uint32_t)-26, (uint32_t)-26, (uint32_t)-26},
  .pre_clamp = {0x10, 0xeb, 0x10, 0xf0, 0x10, 0xf0},
  .post_clamp = {0x00, 0xff, 0x00, 0xff, 0x00, 0xff},
};

static uint32_t get_object_property(int fd, uint32_t object_id, uint32_t object_type,
                                    const char *name, uint64_t *value) {
  uint32_t property_id = 0;
  drmModeObjectProperties *properties = drmModeObjectGetProperties(fd, object_id, object_type);
  if (!properties) return 0;
  for (uint32_t index = 0; index < properties->count_props; ++index) {
    drmModePropertyRes *property = drmModeGetProperty(fd, properties->props[index]);
    if (property && strcmp(property->name, name) == 0) {
      property_id = property->prop_id;
      if (value) *value = properties->prop_values[index];
    }
    drmModeFreeProperty(property);
    if (property_id) break;
  }
  drmModeFreeObjectProperties(properties);
  return property_id;
}

static int find_atomic_planes(int fd, uint32_t crtc_id) {
  if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0 ||
      drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) return -1;
  drmModePlaneRes *planes = drmModeGetPlaneResources(fd);
  if (!planes) return -1;
  int primary_found = 0;
  int camera_found = 0;
  for (uint32_t index = 0; index < planes->count_planes; ++index) {
    drmModePlane *plane = drmModeGetPlane(fd, planes->planes[index]);
    if (!plane) continue;
    uint64_t type = 0, assigned_crtc = 0;
    get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "type", &type);
    get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", &assigned_crtc);
    if (!primary_found && type == DRM_PLANE_TYPE_PRIMARY &&
        (assigned_crtc == crtc_id || assigned_crtc == 0)) {
      drm_state.primary_plane_id = plane->plane_id;
      drm_state.plane_fb_id_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "FB_ID", NULL);
      drm_state.plane_crtc_id_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", NULL);
      drm_state.plane_crtc_x_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_X", NULL);
      drm_state.plane_crtc_y_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_Y", NULL);
      drm_state.plane_crtc_w_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_W", NULL);
      drm_state.plane_crtc_h_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_H", NULL);
      drm_state.plane_src_x_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_X", NULL);
      drm_state.plane_src_y_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_Y", NULL);
      drm_state.plane_src_w_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_W", NULL);
      drm_state.plane_src_h_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_H", NULL);
      drm_state.plane_rotation_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rotation", NULL);
      drm_state.plane_zpos_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "zpos", NULL);
      drm_state.plane_alpha_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "alpha", NULL);
      drm_state.plane_blend_prop = get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "blend_op", NULL);
      primary_found = drm_state.plane_fb_id_prop && drm_state.plane_crtc_id_prop &&
                      drm_state.plane_crtc_x_prop && drm_state.plane_crtc_y_prop &&
                      drm_state.plane_crtc_w_prop && drm_state.plane_crtc_h_prop &&
                      drm_state.plane_src_x_prop && drm_state.plane_src_y_prop &&
                      drm_state.plane_src_w_prop && drm_state.plane_src_h_prop &&
                      drm_state.plane_rotation_prop && drm_state.plane_zpos_prop &&
                      drm_state.plane_alpha_prop && drm_state.plane_blend_prop;
    } else if (!camera_found && assigned_crtc == 0) {
      int supports_nv12 = 0;
      for (uint32_t format_index = 0; format_index < plane->count_formats; ++format_index) {
        if (plane->formats[format_index] == DRM_FORMAT_NV12) supports_nv12 = 1;
      }
      // rot_dst_w is only installed on planes backed by the inline SBUF rotator.
      if (supports_nv12 &&
          get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rot_dst_w", NULL)) {
        drm_state.camera_plane_id = plane->plane_id;
        drm_state.plane_rot_dst_x_prop =
            get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rot_dst_x", NULL);
        drm_state.plane_rot_dst_y_prop =
            get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rot_dst_y", NULL);
        drm_state.plane_rot_dst_w_prop =
            get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rot_dst_w", NULL);
        drm_state.plane_rot_dst_h_prop =
            get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "rot_dst_h", NULL);
        drm_state.plane_csc_prop =
            get_object_property(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE, "csc_v1", NULL);
        camera_found = drm_state.plane_rot_dst_x_prop && drm_state.plane_rot_dst_y_prop &&
                       drm_state.plane_rot_dst_w_prop && drm_state.plane_rot_dst_h_prop &&
                       drm_state.plane_csc_prop;
      }
    }
    drmModeFreePlane(plane);
  }
  drmModeFreePlaneResources(planes);
  return primary_found && camera_found ? 0 : -1;
}

static void page_flip_handler(int fd, unsigned int sequence, unsigned int tv_sec,
                              unsigned int tv_usec, void *user_data) {
  (void)fd;
  (void)sequence;
  (void)tv_sec;
  (void)tv_usec;
  *(int *)user_data = 0;
}

static int wait_for_page_flip(int *waiting) {
  drmEventContext context = {
    .version = 2,
    .page_flip_handler = page_flip_handler,
  };
  while (*waiting) {
    struct pollfd descriptor = {.fd = drm_state.fd, .events = POLLIN};
    const int result = poll(&descriptor, 1, -1);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return -errno;
    if (drmHandleEvent(drm_state.fd, &context) != 0) return -errno;
  }
  return 0;
}

typedef struct __attribute__((__packed__)) {
  uint16_t gamma;
  uint16_t ccm[9];
  uint16_t rgb_color_gains[3];
} ColorCorrectionValues;

static float decode_float16(uint16_t value) {
  const uint32_t sign = value >> 15;
  int exponent = (value >> 10) & 0x1f;
  uint32_t fraction = value & 0x3ff;
  uint32_t output;
  if (exponent == 0) {
    if (fraction == 0) {
      output = sign << 31;
    } else {
      exponent = 127 - 14;
      while ((fraction & (1 << 10)) == 0) {
        --exponent;
        fraction <<= 1;
      }
      fraction &= 0x3ff;
      output = (sign << 31) | ((uint32_t)exponent << 23) | (fraction << 13);
    }
  } else if (exponent == 0x1f) {
    output = (sign << 31) | (0xff << 23) | (fraction << 13);
  } else {
    output = (sign << 31) | ((uint32_t)(exponent + 112) << 23) | (fraction << 13);
  }
  float result;
  memcpy(&result, &output, sizeof(result));
  return result;
}

static int init_color_correction(void) {
  if (getenv("DISABLE_COLOR_CORRECTION")) return 0;
  const char *paths[] = {
    getenv("COLOR_CORRECTION_PATH"),
    "/data/misc/display/color_cal/color_cal",
    "/sys/devices/platform/soc/894000.i2c/i2c-2/2-0017/color_cal",
    "/persist/comma/color_cal",
  };
  ColorCorrectionValues values;
  int loaded = 0;
  for (size_t index = 0; index < sizeof(paths) / sizeof(paths[0]); ++index) {
    if (!paths[index]) continue;
    FILE *file = fopen(paths[index], "rb");
    if (!file) continue;
    loaded = fread(&values, sizeof(values), 1, file) == 1;
    fclose(file);
    if (loaded) break;
  }
  if (!loaded) return 0;

  float gain[3];
  float matrix[9];
  const float gamma = decode_float16(values.gamma);
  for (int channel = 0; channel < 3; ++channel) {
    const float decoded = decode_float16(values.rgb_color_gains[channel]);
    if (!isfinite(decoded) || decoded == 0) return 0;
    gain[channel] = 1.0f / decoded;
  }
  if (!isfinite(gamma) || gamma == 0) return 0;
  for (int index = 0; index < 9; ++index) {
    matrix[index] = decode_float16(values.ccm[index]);
    if (!isfinite(matrix[index])) return 0;
  }
  for (int input = 0; input < 3; ++input) {
    for (int output = 0; output < 3; ++output) {
      for (int value = 0; value < 256; ++value) {
        const float linear = powf(value / 255.0f, 2.2f) * gain[input];
        drm_state.color_contribution[input][output][value] = linear * matrix[input * 3 + output];
      }
    }
  }
  const float output_exponent = (1.0f / gamma) / 2.2f;
  for (int index = 0; index < 4096; ++index) {
    const float value = powf(index / 4095.0f, output_exponent) * 255.0f;
    drm_state.color_gamma[index] = (uint8_t)fminf(255.0f, fmaxf(0.0f, roundf(value)));
  }
  drm_state.color_correction = 1;
  return 1;
}

static inline uint32_t correct_display_color(uint32_t pixel) {
  if (!drm_state.color_correction) return pixel;
  const int input[3] = {pixel & 255, (pixel >> 8) & 255, (pixel >> 16) & 255};
  int output[3];
  for (int channel = 0; channel < 3; ++channel) {
    float linear = drm_state.color_contribution[0][channel][input[0]] +
                   drm_state.color_contribution[1][channel][input[1]] +
                   drm_state.color_contribution[2][channel][input[2]];
    int index = (int)roundf(linear * 4095.0f);
    if (index < 0) index = 0;
    if (index > 4095) index = 4095;
    output[channel] = drm_state.color_gamma[index];
  }
  return (pixel & 0xff000000U) | ((uint32_t)output[2] << 16) |
         ((uint32_t)output[1] << 8) | (uint32_t)output[0];
}

static int recv_fd(int sock) {
  char byte = 0;
  struct iovec io = { .iov_base = &byte, .iov_len = 1 };
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr msg = {
    .msg_iov = &io, .msg_iovlen = 1,
    .msg_control = control, .msg_controllen = sizeof(control),
  };
  if (recvmsg(sock, &msg, 0) < 0) return -1;
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) return -1;
  int fd = -1;
  memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
  return fd;
}

static int get_drm_fd(void) {
  const char *fd_env = getenv("DRM_FD");
  if (fd_env) return atoi(fd_env);
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) return -1;
  struct sockaddr_un addr = { .sun_family = AF_UNIX };
  snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", "/tmp/drmfd.sock");
  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    close(sock);
    return -1;
  }
  int fd = recv_fd(sock);
  if (fd < 0) {
    close(sock);
  } else {
    // magic.py uses the lifetime of this socket to arbitrate the display. It
    // restores the boot background as soon as the last client disconnects.
    drm_state.server_sock = sock;
  }
  return fd;
}

static int create_scanout_buffer(int fd, uint32_t width, uint32_t height,
                                 uint32_t format, int cached, DrmBuffer *buffer) {
  uint64_t map_offset;
  if (cached) {
    buffer->pitch = (width * 4 + 63) & ~63U;
    buffer->size = (buffer->pitch * height + 4095) & ~4095ULL;
    struct drm_msm_gem_new create = {
      .size = buffer->size,
      .flags = MSM_BO_SCANOUT | MSM_BO_CACHED,
    };
    if (ioctl(fd, DRM_IOCTL_MSM_GEM_NEW, &create) < 0) return -1;
    buffer->handle = create.handle;
    struct drm_msm_gem_info info = {.handle = buffer->handle};
    if (ioctl(fd, DRM_IOCTL_MSM_GEM_INFO, &info) < 0) return -1;
    map_offset = info.offset;
    buffer->cached = 1;
  } else {
    struct drm_mode_create_dumb create = { .width = width, .height = height, .bpp = 32 };
    if (ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) return -1;
    buffer->handle = create.handle;
    buffer->pitch = create.pitch;
    buffer->size = create.size;
    struct drm_mode_map_dumb map = { .handle = buffer->handle };
    if (ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map) < 0) return -1;
    map_offset = map.offset;
  }
  uint32_t handles[4] = { buffer->handle };
  uint32_t pitches[4] = { buffer->pitch };
  uint32_t offsets[4] = { 0 };
  if (drmModeAddFB2(fd, width, height, format, handles, pitches, offsets,
                    &buffer->fb_id, 0) != 0) return -1;
  buffer->map = mmap(NULL, buffer->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_offset);
  if (buffer->map == MAP_FAILED) return -1;
  memset(buffer->map, 0, buffer->size);
  return 0;
}

static int prepare_cpu_buffer(DrmBuffer *buffer) {
  if (!buffer->cached || buffer->cpu_prepared) return 0;
  struct drm_msm_gem_cpu_prep prep = {
    .handle = buffer->handle,
    .op = MSM_PREP_WRITE,
    .timeout = {.tv_sec = 1},
  };
  if (ioctl(drm_state.fd, DRM_IOCTL_MSM_GEM_CPU_PREP, &prep) < 0) return -errno;
  buffer->cpu_prepared = 1;
  return 0;
}

static int finish_cpu_buffer(DrmBuffer *buffer) {
  if (!buffer->cached || !buffer->cpu_prepared) return 0;
  struct drm_msm_gem_cpu_fini fini = {.handle = buffer->handle};
  if (ioctl(drm_state.fd, DRM_IOCTL_MSM_GEM_CPU_FINI, &fini) < 0) return -errno;
  buffer->cpu_prepared = 0;
  return 0;
}

int sr_drm_init(void) {
  if (drm_state.initialized) return 0;
  drm_state.fd = get_drm_fd();
  if (drm_state.fd < 0 || !drmIsMaster(drm_state.fd)) return -1;
  drmModeRes *resources = drmModeGetResources(drm_state.fd);
  if (!resources) return -1;
  drmModeConnector *connector = NULL;
  for (int i = 0; i < resources->count_connectors; ++i) {
    connector = drmModeGetConnector(drm_state.fd, resources->connectors[i]);
    if (connector && connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) break;
    drmModeFreeConnector(connector);
    connector = NULL;
  }
  if (!connector || resources->count_crtcs == 0) {
    drmModeFreeConnector(connector);
    drmModeFreeResources(resources);
    return -1;
  }
  drm_state.connector_id = connector->connector_id;
  drm_state.crtc_id = resources->crtcs[0];
  drm_state.original_crtc = drmModeGetCrtc(drm_state.fd, drm_state.crtc_id);
  drm_state.mode = connector->modes[0];
  drmModeFreeConnector(connector);
  drmModeFreeResources(resources);
  const char *mdp_camera = getenv("CPU_MDP_CAMERA");
  const int atomic_planes = find_atomic_planes(drm_state.fd, drm_state.crtc_id) == 0;
  drm_state.mdp_ui = atomic_planes && drm_state.mode.vdisplay > drm_state.mode.hdisplay;
  drm_state.mdp_camera = atomic_planes && (!mdp_camera || strcmp(mdp_camera, "0") != 0);
  const uint32_t buffer_width =
      drm_state.mdp_ui ? drm_state.mode.vdisplay : drm_state.mode.hdisplay;
  const uint32_t buffer_height =
      drm_state.mdp_ui ? drm_state.mode.hdisplay : drm_state.mode.vdisplay;
  // The software renderer's native RGBA byte order maps directly to
  // DRM ABGR8888, a format supported by MICI's inline SDE rotator.
  const uint32_t buffer_format = DRM_FORMAT_ABGR8888;
  if (create_scanout_buffer(drm_state.fd, buffer_width, buffer_height,
                            buffer_format, drm_state.mdp_ui, &drm_state.buffers[0]) ||
      create_scanout_buffer(drm_state.fd, buffer_width, buffer_height,
                            buffer_format, drm_state.mdp_ui, &drm_state.buffers[1])) return -1;
  FILE *origin = fopen("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id", "r");
  int canonical_zero = 0;
  if (origin) {
    if (fscanf(origin, "%d", &canonical_zero) != 1) canonical_zero = 0;
    fclose(origin);
  }
  drm_state.clockwise = !canonical_zero;
  if (!canonical_zero) init_color_correction();
  const char *direct_render = getenv("CPU_DIRECT_KMS");
  drm_state.direct_render = !direct_render || strcmp(direct_render, "0") != 0;
  drm_state.front = 0;
  drm_state.initialized = 1;
  return 0;
}

uint8_t *sr_drm_back_buffer(int *stride) {
  if (!drm_state.initialized || !drm_state.mdp_ui || !drm_state.direct_render ||
      drm_state.color_correction) return NULL;
  DrmBuffer *next = &drm_state.buffers[1 - drm_state.front];
  if (prepare_cpu_buffer(next) != 0) return NULL;
  if (stride) *stride = (int)next->pitch;
  return next->map;
}

void sr_drm_camera_begin_frame(void) {
  ++drm_state.camera_frame;
  drm_state.camera_active = 0;
}

static void release_camera_buffer(CameraBuffer *buffer) {
  if (buffer->fb_id) drmModeRmFB(drm_state.fd, buffer->fb_id);
  if (buffer->handle) {
    struct drm_gem_close close_args = {.handle = buffer->handle};
    ioctl(drm_state.fd, DRM_IOCTL_GEM_CLOSE, &close_args);
  }
  memset(buffer, 0, sizeof(*buffer));
}

int sr_drm_set_camera(int dma_fd, int width, int height, int stride, int uv_offset,
                      float source_x, float source_y, float source_width, float source_height,
                      int destination_x, int destination_y, int destination_width,
                      int destination_height, int flip_x, int engaged, int enhance_driver) {
  if (!drm_state.mdp_camera || dma_fd < 0 || width <= 0 || height <= 0 ||
      stride <= 0 || uv_offset <= 0) return -ENOTSUP;
  // The driver-camera shader uses a nonlinear smoothstep/gamma curve that the
  // VIG plane cannot reproduce. Keep that infrequent view on the exact CPU
  // path instead of silently changing its appearance.
  if (enhance_driver) return -ENOTSUP;
  struct stat metadata;
  if (fstat(dma_fd, &metadata) != 0) return -errno;
  CameraBuffer *buffer = NULL;
  for (int index = 0; index < drm_state.camera_buffer_count; ++index) {
    if (drm_state.camera_buffers[index].device == metadata.st_dev &&
        drm_state.camera_buffers[index].inode == metadata.st_ino) {
      buffer = &drm_state.camera_buffers[index];
      break;
    }
  }
  if (!buffer) {
    int appended = 0;
    if (drm_state.camera_buffer_count < MAX_CAMERA_BUFFERS) {
      buffer = &drm_state.camera_buffers[drm_state.camera_buffer_count++];
      appended = 1;
    } else {
      // A camerad restart replaces its complete DMA-BUF ring. Reuse the
      // least-recently-used import, but never remove the framebuffer that may
      // still be scanned out by the previous atomic commit.
      for (int index = 0; index < drm_state.camera_buffer_count; ++index) {
        CameraBuffer *candidate = &drm_state.camera_buffers[index];
        if (candidate->fb_id == drm_state.displayed_camera_fb_id) continue;
        if (!buffer || candidate->last_used < buffer->last_used) buffer = candidate;
      }
      if (!buffer) return -EBUSY;
      release_camera_buffer(buffer);
    }
    if (drmPrimeFDToHandle(drm_state.fd, dma_fd, &buffer->handle) != 0) {
      if (appended) --drm_state.camera_buffer_count;
      return -errno;
    }
    uint32_t handles[4] = {buffer->handle, buffer->handle};
    uint32_t pitches[4] = {(uint32_t)stride, (uint32_t)stride};
    uint32_t offsets[4] = {0, (uint32_t)uv_offset};
    if (drmModeAddFB2(drm_state.fd, width, height, DRM_FORMAT_NV12,
                      handles, pitches, offsets, &buffer->fb_id, 0) != 0) {
      struct drm_gem_close close_args = {.handle = buffer->handle};
      ioctl(drm_state.fd, DRM_IOCTL_GEM_CLOSE, &close_args);
      memset(buffer, 0, sizeof(*buffer));
      if (appended) --drm_state.camera_buffer_count;
      return -errno;
    }
    buffer->device = metadata.st_dev;
    buffer->inode = metadata.st_ino;
  }
  buffer->last_used = drm_state.camera_frame;

  // The inline rotator's downscaler supports symmetric power-of-two ratios.
  // Align the input crop to four pixels so its 2x NV12 output remains even.
  int sx = (int)floorf(source_x) & ~3;
  int sy = (int)floorf(source_y) & ~3;
  int sx1 = ((int)ceilf(source_x + source_width) + 3) & ~3;
  int sy1 = ((int)ceilf(source_y + source_height) + 3) & ~3;
  sx = sx < 0 ? 0 : sx;
  sy = sy < 0 ? 0 : sy;
  sx1 = sx1 > width ? width : sx1;
  sy1 = sy1 > height ? height : sy1;
  if (sx1 <= sx || sy1 <= sy || destination_width <= 0 || destination_height <= 0) return -EINVAL;
  drm_state.camera_fb_id = buffer->fb_id;
  drm_state.camera_src_x = sx;
  drm_state.camera_src_y = sy;
  drm_state.camera_src_w = sx1 - sx;
  drm_state.camera_src_h = sy1 - sy;
  drm_state.camera_dst_x = destination_x;
  drm_state.camera_dst_y = destination_y;
  drm_state.camera_dst_w = destination_width;
  drm_state.camera_dst_h = destination_height;
  drm_state.camera_flip_x = flip_x;
  // The existing shader renders disengaged road video at 85% over black.
  drm_state.camera_alpha = engaged ? 255 : 217;
  drm_state.camera_engaged = engaged;
  drm_state.camera_active = 1;
  return 0;
}

void sr_clear_transparent(Surface *surface, int x, int y, int width, int height) {
  if (!surface || !surface->pixels) return;
  const int x0 = x > surface->clip_x0 ? x : surface->clip_x0;
  const int y0 = y > surface->clip_y0 ? y : surface->clip_y0;
  const int x1 = x + width < surface->clip_x1 ? x + width : surface->clip_x1;
  const int y1 = y + height < surface->clip_y1 ? y + height : surface->clip_y1;
  if (x1 <= x0 || y1 <= y0) return;
  for (int row = y0; row < y1; ++row) {
    memset(surface->pixels + row * surface->stride + x0 * 4, 0, (size_t)(x1 - x0) * 4);
  }
}

int sr_drm_present(const Surface *surface) {
  if (!drm_state.initialized) return -ENODEV;
  DrmBuffer *next = &drm_state.buffers[1 - drm_state.front];
  const int physical_width = drm_state.mode.hdisplay;
  const int physical_height = drm_state.mode.vdisplay;
  const int direct_copy =
      surface->width * 4 <= (int)next->pitch &&
      (size_t)surface->height * next->pitch <= next->size;
  if (!direct_copy &&
      (surface->width != physical_height || surface->height != physical_width)) return -EINVAL;
  int buffer_result = prepare_cpu_buffer(next);
  if (buffer_result != 0) return buffer_result;
  struct timespec copy_start, copy_end;
  clock_gettime(CLOCK_MONOTONIC, &copy_start);
  if (surface->pixels == next->map && surface->stride == (int)next->pitch) {
    // The renderer drew directly into the next KMS buffer.
  } else if (direct_copy) {
    for (int y = 0; y < surface->height; ++y) {
      const uint32_t *src = (const uint32_t *)(surface->pixels + y * surface->stride);
      uint32_t *dst = (uint32_t *)(next->map + y * next->pitch);
      for (int x = 0; x < surface->width; ++x) {
        const uint32_t p = correct_display_color(src[x]);
        dst[x] = p;
      }
    }
  } else {
    // Work in small tiles so the source and transposed destination stay hot.
    const int tile = 16;
    for (int sy0 = 0; sy0 < surface->height; sy0 += tile) {
      const int sy1 = sy0 + tile < surface->height ? sy0 + tile : surface->height;
      for (int sx0 = 0; sx0 < surface->width; sx0 += tile) {
        const int sx1 = sx0 + tile < surface->width ? sx0 + tile : surface->width;
        for (int sx = sx0; sx < sx1; ++sx) {
          const int dy = drm_state.clockwise ? sx : surface->width - 1 - sx;
          uint32_t *dst = (uint32_t *)(next->map + dy * next->pitch);
          for (int sy = sy0; sy < sy1; ++sy) {
            const uint32_t *src = (const uint32_t *)(surface->pixels + sy * surface->stride);
            const int dx = drm_state.clockwise ? surface->height - 1 - sy : sy;
            const uint32_t p = correct_display_color(src[sx]);
            dst[dx] = p;
          }
        }
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &copy_end);
  buffer_result = finish_cpu_buffer(next);
  if (buffer_result != 0) return buffer_result;
  drm_state.last_copy_ms = (copy_end.tv_sec - copy_start.tv_sec) * 1000.0 +
                           (copy_end.tv_nsec - copy_start.tv_nsec) / 1000000.0;
  int ret;
  if (drm_state.mdp_ui || drm_state.mdp_camera) {
    drmModeAtomicReq *request = drmModeAtomicAlloc();
    if (!request) return -ENOMEM;
#define ADD_ATOMIC(object, property, value) \
    do { if (drmModeAtomicAddProperty(request, (object), (property), (value)) < 0) { \
      drmModeAtomicFree(request); return -errno; \
    } } while (0)
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_fb_id_prop, next->fb_id);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_crtc_id_prop, drm_state.crtc_id);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_crtc_x_prop, 0);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_crtc_y_prop, 0);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_crtc_w_prop, physical_width);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_crtc_h_prop, physical_height);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_src_x_prop, 0);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_src_y_prop, 0);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_src_w_prop, (uint64_t)surface->width << 16);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_src_h_prop, (uint64_t)surface->height << 16);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_rotation_prop,
               drm_state.mdp_ui ?
                 (drm_state.clockwise ? DRM_MODE_ROTATE_270 : DRM_MODE_ROTATE_90) :
                 DRM_MODE_ROTATE_0);
    if (drm_state.mdp_ui) {
      ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_rot_dst_x_prop, 0);
      ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_rot_dst_y_prop, 0);
      ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_rot_dst_w_prop,
                 (uint64_t)physical_width << 16);
      ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_rot_dst_h_prop,
                 (uint64_t)physical_height << 16);
    }
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_zpos_prop, 1);
    ADD_ATOMIC(drm_state.primary_plane_id, drm_state.plane_blend_prop, 2);  // premultiplied
    if (drm_state.camera_active) {
      const int camera_x = drm_state.mdp_ui ? drm_state.camera_dst_x : drm_state.clockwise ?
          surface->height - drm_state.camera_dst_y - drm_state.camera_dst_h :
          drm_state.camera_dst_y;
      const int camera_y = drm_state.mdp_ui ? drm_state.camera_dst_y : drm_state.clockwise ?
          drm_state.camera_dst_x :
          surface->width - drm_state.camera_dst_x - drm_state.camera_dst_w;
      const int camera_w = drm_state.mdp_ui ? drm_state.camera_dst_w : drm_state.camera_dst_h;
      const int camera_h = drm_state.mdp_ui ? drm_state.camera_dst_h : drm_state.camera_dst_w;
      const uint64_t rotation =
          (drm_state.mdp_ui ? DRM_MODE_ROTATE_0 :
           (drm_state.clockwise ? DRM_MODE_ROTATE_270 : DRM_MODE_ROTATE_90)) |
          (drm_state.camera_flip_x ? DRM_MODE_REFLECT_X : 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_fb_id_prop, drm_state.camera_fb_id);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_id_prop, drm_state.crtc_id);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_x_prop, camera_x);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_y_prop, camera_y);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_w_prop, camera_w);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_h_prop, camera_h);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_src_x_prop, (uint64_t)drm_state.camera_src_x << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_src_y_prop, (uint64_t)drm_state.camera_src_y << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_src_w_prop, (uint64_t)drm_state.camera_src_w << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_src_h_prop, (uint64_t)drm_state.camera_src_h << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_rotation_prop, rotation);
      // MICI uses a video-mode DSI panel. The SDE driver rejects >=1.1x
      // QSEED downscale following inline rotation, but the rotator itself can
      // downscale NV12 by an integral 2x. Its tiled output is then modestly
      // upscaled by QSEED to the display rectangle.
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_rot_dst_x_prop, 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_rot_dst_y_prop, 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_rot_dst_w_prop,
                 (uint64_t)((drm_state.mdp_ui ? drm_state.camera_src_w : drm_state.camera_src_h) / 2) << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_rot_dst_h_prop,
                 (uint64_t)((drm_state.mdp_ui ? drm_state.camera_src_h : drm_state.camera_src_w) / 2) << 16);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_zpos_prop, 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_alpha_prop, drm_state.camera_alpha);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_csc_prop,
                 drm_state.camera_engaged ? (uint64_t)(uintptr_t)&engaged_camera_csc : 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_blend_prop, 1);  // opaque
    } else {
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_fb_id_prop, 0);
      ADD_ATOMIC(drm_state.camera_plane_id, drm_state.plane_crtc_id_prop, 0);
    }
#undef ADD_ATOMIC
    // The downstream SDE driver completes blocking atomic commits at vblank
    // but does not reliably emit a generic DRM page-flip event for them.
    ret = drmModeAtomicCommit(drm_state.fd, request, 0, NULL);
    drmModeAtomicFree(request);
    if (ret != 0) {
      fprintf(stderr, "CPU DRM MDP camera commit failed: %s\n", strerror(errno));
      // Permanently fall back to the proven legacy KMS path. The current
      // frame may contain a transparent camera rectangle, but the next frame
      // will use CPU conversion and normal opaque scanout.
      drmModeSetPlane(drm_state.fd, drm_state.camera_plane_id, drm_state.crtc_id,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      drm_state.displayed_camera_fb_id = 0;
      drm_state.mdp_camera = 0;
      ret = drmModeSetCrtc(drm_state.fd, drm_state.crtc_id, next->fb_id, 0, 0,
                           &drm_state.connector_id, 1, &drm_state.mode);
      if (ret != 0) return -errno;
    } else {
      drm_state.displayed_camera_fb_id =
          drm_state.camera_active ? drm_state.camera_fb_id : 0;
      // Inline-rotator commits complete before the 60 Hz panel scanout and
      // this downstream driver exposes neither a usable output fence nor a
      // reliable page-flip event for the path. Use an absolute deadline to
      // avoid redrawing buffers at 200+ FPS and drifting over time.
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      int64_t deadline_ns = (int64_t)drm_state.present_deadline.tv_sec * 1000000000LL +
                            drm_state.present_deadline.tv_nsec;
      const int64_t now_ns = (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
      if (deadline_ns == 0 || now_ns > deadline_ns + 4 * 16666667LL) deadline_ns = now_ns;
      deadline_ns += 16666667LL;
      drm_state.present_deadline.tv_sec = deadline_ns / 1000000000LL;
      drm_state.present_deadline.tv_nsec = deadline_ns % 1000000000LL;
      while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME,
                             &drm_state.present_deadline, NULL) == EINTR) {}
    }
  } else if (!drm_state.presented) {
    ret = drmModeSetCrtc(drm_state.fd, drm_state.crtc_id, next->fb_id, 0, 0,
                         &drm_state.connector_id, 1, &drm_state.mode);
    if (ret != 0) {
      fprintf(stderr, "CPU DRM modeset failed: %s\n", strerror(errno));
      return -errno;
    }
    if (ret == 0) {
      drmVBlank vblank = {0};
      vblank.request.type = DRM_VBLANK_RELATIVE;
      vblank.request.sequence = 1;
      if (drmWaitVBlank(drm_state.fd, &vblank) != 0) {
        fprintf(stderr, "CPU DRM initial vblank failed: %s\n", strerror(errno));
        return -errno;
      }
    }
  } else {
    int waiting = 1;
    ret = drmModePageFlip(drm_state.fd, drm_state.crtc_id, next->fb_id,
                          DRM_MODE_PAGE_FLIP_EVENT, &waiting);
    if (ret != 0) {
      fprintf(stderr, "CPU DRM page flip failed: %s\n", strerror(errno));
      return -errno;
    }
    ret = wait_for_page_flip(&waiting);
    if (ret != 0) {
      fprintf(stderr, "CPU DRM page flip wait failed: %s\n", strerror(-ret));
      return ret;
    }
  }
  drm_state.presented = 1;
  drm_state.front = 1 - drm_state.front;
  return 0;
}

double sr_drm_last_copy_ms(void) {
  return drm_state.last_copy_ms;
}

void sr_drm_close(void) {
  if (!drm_state.initialized) return;
  int restored = 0;
  if (drm_state.original_crtc && drm_state.original_crtc->mode_valid && drm_state.original_crtc->buffer_id) {
    drmModeFB *original_fb = drmModeGetFB(drm_state.fd, drm_state.original_crtc->buffer_id);
    if (original_fb) {
      restored = drmModeSetCrtc(drm_state.fd, drm_state.original_crtc->crtc_id,
                                drm_state.original_crtc->buffer_id,
                                drm_state.original_crtc->x, drm_state.original_crtc->y,
                                &drm_state.connector_id, 1, &drm_state.original_crtc->mode) == 0;
      drmModeFreeFB(original_fb);
    }
  }
  if (!restored) drmModeSetCrtc(drm_state.fd, drm_state.crtc_id, 0, 0, 0, NULL, 0, NULL);
  drmModeFreeCrtc(drm_state.original_crtc);
  for (int i = 0; i < drm_state.camera_buffer_count; ++i) {
    release_camera_buffer(&drm_state.camera_buffers[i]);
  }
  for (int i = 0; i < 2; ++i) {
    DrmBuffer *buffer = &drm_state.buffers[i];
    if (buffer->map && buffer->map != MAP_FAILED) munmap(buffer->map, buffer->size);
    if (buffer->fb_id) drmModeRmFB(drm_state.fd, buffer->fb_id);
    if (buffer->handle && buffer->cached) {
      struct drm_gem_close close_args = {.handle = buffer->handle};
      ioctl(drm_state.fd, DRM_IOCTL_GEM_CLOSE, &close_args);
    } else if (buffer->handle) {
      struct drm_mode_destroy_dumb destroy = { .handle = buffer->handle };
      ioctl(drm_state.fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
    }
  }
  close(drm_state.fd);
  if (drm_state.server_sock >= 0) close(drm_state.server_sock);
  memset(&drm_state, 0, sizeof(drm_state));
  drm_state.fd = -1;
  drm_state.server_sock = -1;
}
#else
int sr_drm_init(void) { return -1; }
int sr_drm_present(const Surface *surface) { (void)surface; return -1; }
double sr_drm_last_copy_ms(void) { return 0; }
uint8_t *sr_drm_back_buffer(int *stride) { (void)stride; return NULL; }
void sr_drm_camera_begin_frame(void) {}
int sr_drm_set_camera(int dma_fd, int width, int height, int stride, int uv_offset,
                      float source_x, float source_y, float source_width, float source_height,
                      int destination_x, int destination_y, int destination_width,
                      int destination_height, int flip_x, int engaged, int enhance_driver) {
  (void)dma_fd; (void)width; (void)height; (void)stride; (void)uv_offset;
  (void)source_x; (void)source_y; (void)source_width; (void)source_height;
  (void)destination_x; (void)destination_y; (void)destination_width;
  (void)destination_height; (void)flip_x; (void)engaged; (void)enhance_driver;
  return -1;
}
void sr_clear_transparent(Surface *surface, int x, int y, int width, int height) {
  (void)surface; (void)x; (void)y; (void)width; (void)height;
}
void sr_drm_close(void) {}
#endif

static inline uint32_t blend(uint32_t dst, uint32_t src) {
  const uint32_t a = src >> 24;
  if (a == 255) return src;
  if (a == 0) return dst;

  const uint32_t ia = 255 - a;
  const uint32_t sr = (src >> 16) & 255;
  const uint32_t sg = (src >> 8) & 255;
  const uint32_t sb = src & 255;
  const uint32_t dr = (dst >> 16) & 255;
  const uint32_t dg = (dst >> 8) & 255;
  const uint32_t db = dst & 255;
  const uint32_t r = sr + ((dr * ia + 127) / 255);
  const uint32_t g = sg + ((dg * ia + 127) / 255);
  const uint32_t b = sb + ((db * ia + 127) / 255);
  const uint32_t da = dst >> 24;
  const uint32_t oa = a + ((da * ia + 127) / 255);
  return (oa << 24) | (r << 16) | (g << 8) | b;
}

static inline uint32_t premultiply(uint32_t color) {
  const uint32_t a = color >> 24;
  if (a == 255) return color;
  const uint32_t r = (color >> 16) & 255;
  const uint32_t g = (color >> 8) & 255;
  const uint32_t b = color & 255;
  return (a << 24) | (((r * a + 127) / 255) << 16) |
         (((g * a + 127) / 255) << 8) | ((b * a + 127) / 255);
}

void sr_set_clip(Surface *s, int x, int y, int width, int height) {
  s->clip_x0 = x < 0 ? 0 : (x > s->width ? s->width : x);
  s->clip_y0 = y < 0 ? 0 : (y > s->height ? s->height : y);
  const int x1 = x + width;
  const int y1 = y + height;
  s->clip_x1 = x1 < 0 ? 0 : (x1 > s->width ? s->width : x1);
  s->clip_y1 = y1 < 0 ? 0 : (y1 > s->height ? s->height : y1);
}

void sr_reset_clip(Surface *s) {
  s->clip_x0 = 0;
  s->clip_y0 = 0;
  s->clip_x1 = s->width;
  s->clip_y1 = s->height;
}

void sr_clear(Surface *s, uint32_t color) {
  for (int y = s->clip_y0; y < s->clip_y1; ++y) {
    uint32_t *row = (uint32_t *)(s->pixels + y * s->stride);
    for (int x = s->clip_x0; x < s->clip_x1; ++x) row[x] = color;
  }
}

void sr_rect(Surface *s, int x, int y, int w, int h, uint32_t color) {
  const int x0 = x < s->clip_x0 ? s->clip_x0 : x;
  const int y0 = y < s->clip_y0 ? s->clip_y0 : y;
  const int x1 = x + w > s->clip_x1 ? s->clip_x1 : x + w;
  const int y1 = y + h > s->clip_y1 ? s->clip_y1 : y + h;
  if (x0 >= x1 || y0 >= y1) return;
  const uint32_t src = premultiply(color);
  for (int py = y0; py < y1; ++py) {
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    if ((src >> 24) == 255) {
      for (int px = x0; px < x1; ++px) row[px] = src;
    } else {
      for (int px = x0; px < x1; ++px) row[px] = blend(row[px], src);
    }
  }
}

void sr_gradient_v(Surface *s, int x, int y, int w, int h, uint32_t top, uint32_t bottom) {
  if (h <= 0) return;
  for (int py = 0; py < h; ++py) {
    const int t = h == 1 ? 0 : (py * 256) / (h - 1);
    uint32_t c = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      const int a = (top >> shift) & 255;
      const int b = (bottom >> shift) & 255;
      c |= (uint32_t)(a + (((b - a) * t) >> 8)) << shift;
    }
    sr_rect(s, x, y + py, w, 1, c);
  }
}

void sr_gradient_4(Surface *s, int x, int y, int width, int height,
                   uint32_t top_left, uint32_t bottom_left,
                   uint32_t top_right, uint32_t bottom_right) {
  if (width <= 0 || height <= 0) return;
  const int x0 = x < s->clip_x0 ? s->clip_x0 : x;
  const int y0 = y < s->clip_y0 ? s->clip_y0 : y;
  const int x1 = x + width > s->clip_x1 ? s->clip_x1 : x + width;
  const int y1 = y + height > s->clip_y1 ? s->clip_y1 : y + height;
  for (int py = y0; py < y1; ++py) {
    const float v = height == 1 ? 0 : (float)(py - y) / (height - 1);
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    for (int px = x0; px < x1; ++px) {
      const float u = width == 1 ? 0 : (float)(px - x) / (width - 1);
      uint32_t color = 0;
      for (int shift = 0; shift < 32; shift += 8) {
        const float left = ((top_left >> shift) & 255) * (1 - v) + ((bottom_left >> shift) & 255) * v;
        const float right = ((top_right >> shift) & 255) * (1 - v) + ((bottom_right >> shift) & 255) * v;
        color |= (uint32_t)roundf(left * (1 - u) + right * u) << shift;
      }
      row[px] = blend(row[px], premultiply(color));
    }
  }
}

void sr_rounded_rect(Surface *s, int x, int y, int width, int height,
                     float radius, float thickness, uint32_t color) {
  if (width <= 0 || height <= 0) return;
  radius = fmaxf(0, fminf(radius, fminf(width, height) * .5f));
  const int x0 = x < s->clip_x0 ? s->clip_x0 : x;
  const int y0 = y < s->clip_y0 ? s->clip_y0 : y;
  const int x1 = x + width > s->clip_x1 ? s->clip_x1 : x + width;
  const int y1 = y + height > s->clip_y1 ? s->clip_y1 : y + height;
  const float cx = x + width * .5f, cy = y + height * .5f;
  const float bx = width * .5f - radius, by = height * .5f - radius;
  const uint32_t src = premultiply(color);
  for (int py = y0; py < y1; ++py) {
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    for (int px = x0; px < x1; ++px) {
      const float qx = fabsf(px + .5f - cx) - bx;
      const float qy = fabsf(py + .5f - cy) - by;
      const float outside = sqrtf(fmaxf(qx, 0) * fmaxf(qx, 0) + fmaxf(qy, 0) * fmaxf(qy, 0));
      const float distance = outside + fminf(fmaxf(qx, qy), 0) - radius;
      if (distance <= 0 && (thickness <= 0 || distance >= -thickness)) row[px] = blend(row[px], src);
    }
  }
}

void sr_circle(Surface *s, int cx, int cy, int radius, uint32_t color) {
  if (radius <= 0) return;
  const uint32_t src = premultiply(color);
  for (int dy = -radius; dy <= radius; ++dy) {
    const int py = cy + dy;
    if (py < s->clip_y0 || py >= s->clip_y1) continue;
    const int half = (int)sqrtf((float)(radius * radius - dy * dy));
    int x0 = cx - half;
    int x1 = cx + half + 1;
    if (x0 < s->clip_x0) x0 = s->clip_x0;
    if (x1 > s->clip_x1) x1 = s->clip_x1;
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    for (int px = x0; px < x1; ++px) row[px] = blend(row[px], src);
  }
}

void sr_ring_arc(Surface *s, int cx, int cy, int inner_radius, int outer_radius,
                 float start_angle, float end_angle, uint32_t color) {
  if (outer_radius <= 0 || inner_radius >= outer_radius) return;
  const uint32_t src = premultiply(color);
  const int inner2 = inner_radius * inner_radius;
  const int outer2 = outer_radius * outer_radius;
  for (int y = -outer_radius; y <= outer_radius; ++y) {
    const int py = cy + y;
    if (py < s->clip_y0 || py >= s->clip_y1) continue;
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    for (int x = -outer_radius; x <= outer_radius; ++x) {
      const int px = cx + x;
      const int distance2 = x * x + y * y;
      if (px >= s->clip_x0 && px < s->clip_x1 && distance2 >= inner2 && distance2 <= outer2) {
        float angle = atan2f((float)y, (float)x) * 180.0f / (float)M_PI;
        if (angle < 0) angle += 360.0f;
        float start = fmodf(start_angle, 360.0f);
        if (start < 0) start += 360.0f;
        float span = end_angle - start_angle;
        if (span < 0) span = -span;
        float relative = angle - start;
        if (relative < 0) relative += 360.0f;
        if (span >= 360.0f || relative <= span) row[px] = blend(row[px], src);
      }
    }
  }
}

void sr_ring(Surface *s, int cx, int cy, int inner_radius, int outer_radius, uint32_t color) {
  sr_ring_arc(s, cx, cy, inner_radius, outer_radius, 0, 360, color);
}

void sr_circle_gradient(Surface *s, int cx, int cy, int radius, uint32_t inner, uint32_t outer) {
  if (radius <= 0) return;
  uint32_t colors[256];
  for (int index = 0; index < 256; ++index) {
    const float t = sqrtf(index / 255.0f);
    uint32_t color = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      const int a = (inner >> shift) & 255;
      const int b = (outer >> shift) & 255;
      color |= (uint32_t)roundf(a + (b - a) * t) << shift;
    }
    colors[index] = premultiply(color);
  }
  const int radius2 = radius * radius;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int py = cy + dy;
    if (py < s->clip_y0 || py >= s->clip_y1) continue;
    const int half = (int)sqrtf((float)(radius * radius - dy * dy));
    int x0 = cx - half, x1 = cx + half + 1;
    if (x0 < s->clip_x0) x0 = s->clip_x0;
    if (x1 > s->clip_x1) x1 = s->clip_x1;
    uint32_t *row = (uint32_t *)(s->pixels + py * s->stride);
    for (int px = x0; px < x1; ++px) {
      const int distance2 = (px - cx) * (px - cx) + dy * dy;
      const int index = distance2 >= radius2 ? 255 : distance2 * 255 / radius2;
      row[px] = blend(row[px], colors[index]);
    }
  }
}

void sr_line(Surface *s, float x0, float y0, float x1, float y1, float thickness, uint32_t color) {
  if (thickness <= 1.1f) {
    int ax = (int)roundf(x0), ay = (int)roundf(y0);
    const int bx = (int)roundf(x1), by = (int)roundf(y1);
    const int dx_i = abs(bx - ax), step_x = ax < bx ? 1 : -1;
    const int dy_i = -abs(by - ay), step_y = ay < by ? 1 : -1;
    int error = dx_i + dy_i;
    const uint32_t src = premultiply(color);
    for (;;) {
      if (ax >= s->clip_x0 && ax < s->clip_x1 && ay >= s->clip_y0 && ay < s->clip_y1) {
        uint32_t *row = (uint32_t *)(s->pixels + ay * s->stride);
        row[ax] = blend(row[ax], src);
      }
      if (ax == bx && ay == by) break;
      const int twice = 2 * error;
      if (twice >= dy_i) { error += dy_i; ax += step_x; }
      if (twice <= dx_i) { error += dx_i; ay += step_y; }
    }
    return;
  }
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float length = sqrtf(dx * dx + dy * dy);
  if (length <= 0.001f) {
    sr_circle(s, (int)roundf(x0), (int)roundf(y0), (int)ceilf(thickness * .5f), color);
    return;
  }
  const float nx = -dy * thickness * .5f / length;
  const float ny = dx * thickness * .5f / length;
  Point a = {x0 + nx, y0 + ny};
  Point b = {x1 + nx, y1 + ny};
  Point c = {x1 - nx, y1 - ny};
  Point d = {x0 - nx, y0 - ny};
  sr_triangle(s, a, b, c, color);
  sr_triangle(s, a, c, d, color);
  const int radius = (int)ceilf(thickness * .5f);
  sr_circle(s, (int)roundf(x0), (int)roundf(y0), radius, color);
  sr_circle(s, (int)roundf(x1), (int)roundf(y1), radius, color);
}

static inline float edge(Point a, Point b, float x, float y) {
  return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}

void sr_triangle(Surface *s, Point a, Point b, Point c, uint32_t color) {
  int x0 = (int)floorf(fminf(a.x, fminf(b.x, c.x)));
  int y0 = (int)floorf(fminf(a.y, fminf(b.y, c.y)));
  int x1 = (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x)));
  int y1 = (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y)));
  if (x0 < s->clip_x0) x0 = s->clip_x0;
  if (y0 < s->clip_y0) y0 = s->clip_y0;
  if (x1 > s->clip_x1) x1 = s->clip_x1;
  if (y1 > s->clip_y1) y1 = s->clip_y1;
  const float area = edge(a, b, c.x, c.y);
  if (area == 0) return;
  const uint32_t src = premultiply(color);
  for (int y = y0; y < y1; ++y) {
    uint32_t *row = (uint32_t *)(s->pixels + y * s->stride);
    for (int x = x0; x < x1; ++x) {
      const float w0 = edge(b, c, x + .5f, y + .5f);
      const float w1 = edge(c, a, x + .5f, y + .5f);
      const float w2 = edge(a, b, x + .5f, y + .5f);
      if ((area > 0 && w0 >= 0 && w1 >= 0 && w2 >= 0) ||
          (area < 0 && w0 <= 0 && w1 <= 0 && w2 <= 0)) {
        row[x] = blend(row[x], src);
      }
    }
  }
}

static uint32_t gradient_color_at(float t, const uint32_t *colors, const float *stops, int color_count) {
  if (color_count <= 1) return colors[0];
  if (t <= stops[0]) return colors[0];
  if (t >= stops[color_count - 1]) return colors[color_count - 1];
  int index = 0;
  while (index + 1 < color_count && t > stops[index + 1]) ++index;
  const float span = fmaxf(stops[index + 1] - stops[index], 1e-6f);
  const float k = fminf(fmaxf((t - stops[index]) / span, 0.0f), 1.0f);
  uint32_t result = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    const int a = (colors[index] >> shift) & 255;
    const int b = (colors[index + 1] >> shift) & 255;
    result |= (uint32_t)roundf(a + (b - a) * k) << shift;
  }
  return result;
}

static Point ribbon_point(const Point *perimeter, int count, int strip_index,
                          float scale_x, float scale_y, float translate_x, float translate_y) {
  Point result = (strip_index & 1) ? perimeter[count - 1 - strip_index / 2] : perimeter[strip_index / 2];
  result.x = result.x * scale_x + translate_x;
  result.y = result.y * scale_y + translate_y;
  return result;
}

void sr_ribbon(Surface *s, const Point *perimeter, int count,
               float scale_x, float scale_y, float translate_x, float translate_y,
               uint32_t solid_color,
               float start_x, float start_y, float end_x, float end_y,
               const uint32_t *colors, const float *stops, int color_count) {
  if (!perimeter || count < 3) return;
  count &= ~1;
  uint32_t gradient_lut[256];
  float gradient_dx = 0, gradient_dy = 0, gradient_inv_length2 = 0;
  if (color_count > 0) {
    gradient_dx = start_x - end_x;
    gradient_dy = start_y - end_y;
    gradient_inv_length2 = 1.0f / fmaxf(
      gradient_dx * gradient_dx + gradient_dy * gradient_dy, 1e-6f);
    for (int value = 0; value < 256; ++value) {
      gradient_lut[value] = premultiply(gradient_color_at(
        value / 255.0f, colors, stops, color_count));
    }
  }
  for (int index = 0; index < count - 2; ++index) {
    Point a = ribbon_point(perimeter, count, index, scale_x, scale_y, translate_x, translate_y);
    Point b = ribbon_point(perimeter, count, index + 1, scale_x, scale_y, translate_x, translate_y);
    Point c = ribbon_point(perimeter, count, index + 2, scale_x, scale_y, translate_x, translate_y);
    if (index & 1) { Point swap = a; a = b; b = swap; }
    if (color_count <= 0) {
      sr_triangle(s, a, b, c, solid_color);
      continue;
    }
    int x0 = (int)floorf(fminf(a.x, fminf(b.x, c.x)));
    int y0 = (int)floorf(fminf(a.y, fminf(b.y, c.y)));
    int x1 = (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x)));
    int y1 = (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y)));
    if (x0 < s->clip_x0) x0 = s->clip_x0;
    if (y0 < s->clip_y0) y0 = s->clip_y0;
    if (x1 > s->clip_x1) x1 = s->clip_x1;
    if (y1 > s->clip_y1) y1 = s->clip_y1;
    const float area = edge(a, b, c.x, c.y);
    if (area == 0) continue;
    for (int y = y0; y < y1; ++y) {
      uint32_t *row = (uint32_t *)(s->pixels + y * s->stride);
      for (int x = x0; x < x1; ++x) {
        const float px = x + .5f, py = y + .5f;
        const float w0 = edge(b, c, px, py);
        const float w1 = edge(c, a, px, py);
        const float w2 = edge(a, b, px, py);
        if ((area > 0 && w0 >= 0 && w1 >= 0 && w2 >= 0) ||
            (area < 0 && w0 <= 0 && w1 <= 0 && w2 <= 0)) {
          float t = ((px - end_x) * gradient_dx + (py - end_y) * gradient_dy) * gradient_inv_length2;
          int lut_index = (int)roundf(t * 255.0f);
          if (lut_index < 0) lut_index = 0;
          if (lut_index > 255) lut_index = 255;
          row[x] = blend(row[x], gradient_lut[lut_index]);
        }
      }
    }
  }
}

void sr_blit(Surface *dst, const Surface *src, int dx, int dy, uint8_t opacity) {
  int sx = 0, sy = 0;
  int w = src->width, h = src->height;
  if (dx < dst->clip_x0) { sx = dst->clip_x0 - dx; w -= sx; dx = dst->clip_x0; }
  if (dy < dst->clip_y0) { sy = dst->clip_y0 - dy; h -= sy; dy = dst->clip_y0; }
  if (dx + w > dst->clip_x1) w = dst->clip_x1 - dx;
  if (dy + h > dst->clip_y1) h = dst->clip_y1 - dy;
  if (w <= 0 || h <= 0) return;
  for (int y = 0; y < h; ++y) {
    uint32_t *d = (uint32_t *)(dst->pixels + (dy + y) * dst->stride);
    const uint32_t *sp = (const uint32_t *)(src->pixels + (sy + y) * src->stride);
    for (int x = 0; x < w; ++x) {
      uint32_t p = sp[sx + x];
      if (opacity != 255) {
        const uint32_t a = ((p >> 24) * opacity + 127) / 255;
        const uint32_t rb = (((p & 0x00ff00ffU) * opacity) + 0x00800080U) >> 8;
        const uint32_t g = (((p & 0x0000ff00U) * opacity) + 0x00008000U) >> 8;
        p = (a << 24) | (rb & 0x00ff00ffU) | (g & 0x0000ff00U);
      }
      d[dx + x] = blend(d[dx + x], p);
    }
  }
}

void sr_blit_opaque(Surface *dst, const Surface *src, int dx, int dy) {
  int sx = 0, sy = 0;
  int width = src->width, height = src->height;
  if (dx < dst->clip_x0) { sx = dst->clip_x0 - dx; width -= sx; dx = dst->clip_x0; }
  if (dy < dst->clip_y0) { sy = dst->clip_y0 - dy; height -= sy; dy = dst->clip_y0; }
  if (dx + width > dst->clip_x1) width = dst->clip_x1 - dx;
  if (dy + height > dst->clip_y1) height = dst->clip_y1 - dy;
  if (width <= 0 || height <= 0) return;
  for (int y = 0; y < height; ++y) {
    void *out = dst->pixels + (dy + y) * dst->stride + dx * 4;
    const void *in = src->pixels + (sy + y) * src->stride + sx * 4;
    memcpy(out, in, (size_t)width * 4);
  }
}

void sr_burn_in_filter(Surface *surface, int x, int y, int width, int height) {
  const int x0 = x > surface->clip_x0 ? x : surface->clip_x0;
  const int y0 = y > surface->clip_y0 ? y : surface->clip_y0;
  const int x1 = x + width < surface->clip_x1 ? x + width : surface->clip_x1;
  const int y1 = y + height < surface->clip_y1 ? y + height : surface->clip_y1;
  for (int py = y0; py < y1; ++py) {
    uint32_t *row = (uint32_t *)(surface->pixels + py * surface->stride);
    for (int px = x0; px < x1; ++px) {
      const uint32_t sampled = row[px];
      const uint32_t alpha = sampled >> 24;
      const uint32_t intensity = (sampled >> 16) & 255;
      // Match application.py's diagnostic shader: blue maps linearly from
      // green through yellow to red while preserving the sampled alpha.
      const uint32_t red = intensity < 128 ? intensity * 2 : 255;
      const uint32_t green = intensity < 128 ? 255 : (255 - intensity) * 2;
      row[px] = (alpha << 24) | (green << 8) | red;
    }
  }
}

static void sr_blit_scaled_impl(Surface *dst, const Surface *src, float src_x, float src_y, float src_w, float src_h,
                                int dst_x, int dst_y, int dst_w, int dst_h, uint32_t tint, int smooth) {
  if (dst_w <= 0 || dst_h <= 0 || src_w == 0 || src_h == 0) return;
  const int flip_x = src_w < 0;
  const int flip_y = src_h < 0;
  if (flip_x) src_w = -src_w;
  if (flip_y) src_h = -src_h;
  const uint32_t ta = tint >> 24;
  const uint32_t tr = (tint >> 16) & 255;
  const uint32_t tg = (tint >> 8) & 255;
  const uint32_t tb = tint & 255;
  if (!flip_x && !flip_y &&
      fabsf(src_w - dst_w) < .001f && fabsf(src_h - dst_h) < .001f &&
      fabsf(src_x - roundf(src_x)) < .001f && fabsf(src_y - roundf(src_y)) < .001f) {
    int sx = (int)roundf(src_x), sy = (int)roundf(src_y);
    int dx = dst_x, dy = dst_y, width = dst_w, height = dst_h;
    if (dx < dst->clip_x0) { const int skip = dst->clip_x0 - dx; sx += skip; width -= skip; dx += skip; }
    if (dy < dst->clip_y0) { const int skip = dst->clip_y0 - dy; sy += skip; height -= skip; dy += skip; }
    if (dx + width > dst->clip_x1) width = dst->clip_x1 - dx;
    if (dy + height > dst->clip_y1) height = dst->clip_y1 - dy;
    if (sx < 0) { const int skip = -sx; dx += skip; width -= skip; sx = 0; }
    if (sy < 0) { const int skip = -sy; dy += skip; height -= skip; sy = 0; }
    if (sx + width > src->width) width = src->width - sx;
    if (sy + height > src->height) height = src->height - sy;
    for (int y = 0; y < height; ++y) {
      uint32_t *out = (uint32_t *)(dst->pixels + (dy + y) * dst->stride);
      const uint32_t *in = (const uint32_t *)(src->pixels + (sy + y) * src->stride);
      for (int x = 0; x < width; ++x) {
        uint32_t p = in[sx + x];
        if (tint != 0xffffffffU) {
          if (tr == 255 && tg == 255 && tb == 255) {
            const uint32_t a = ((p >> 24) * ta + 127) / 255;
            const uint32_t rb = (((p & 0x00ff00ffU) * ta) + 0x00800080U) >> 8;
            const uint32_t g = (((p & 0x0000ff00U) * ta) + 0x00008000U) >> 8;
            p = (a << 24) | (rb & 0x00ff00ffU) | (g & 0x0000ff00U);
          } else {
            const uint32_t a = ((p >> 24) * ta + 127) / 255;
            const uint32_t r = (((p >> 16) & 255) * tr * ta + 32512) / 65025;
            const uint32_t g = (((p >> 8) & 255) * tg * ta + 32512) / 65025;
            const uint32_t b = ((p & 255) * tb * ta + 32512) / 65025;
            p = (a << 24) | (r << 16) | (g << 8) | b;
          }
        }
        out[dx + x] = blend(out[dx + x], p);
      }
    }
    return;
  }
  const int bilinear = smooth && (fabsf(src_w - dst_w) > .001f || fabsf(src_h - dst_h) > .001f);
  for (int y = 0; y < dst_h; ++y) {
    const int py = dst_y + y;
    if (py < dst->clip_y0 || py >= dst->clip_y1) continue;
    float v = (y + .5f) / dst_h;
    if (flip_y) v = 1.0f - v;
    uint32_t *d = (uint32_t *)(dst->pixels + py * dst->stride);
    for (int x = 0; x < dst_w; ++x) {
      const int px = dst_x + x;
      if (px < dst->clip_x0 || px >= dst->clip_x1) continue;
      float u = (x + .5f) / dst_w;
      if (flip_x) u = 1.0f - u;
      uint32_t p;
      if (!bilinear) {
        int sx = (int)(src_x + u * src_w);
        int sy = (int)(src_y + v * src_h);
        if (sx < 0) sx = 0;
        if (sx >= src->width) sx = src->width - 1;
        if (sy < 0) sy = 0;
        if (sy >= src->height) sy = src->height - 1;
        p = *(const uint32_t *)(src->pixels + sy * src->stride + sx * 4);
      } else {
        const float sample_x = src_x + u * src_w - .5f;
        const float sample_y = src_y + v * src_h - .5f;
        int sx0 = (int)floorf(sample_x), sy0 = (int)floorf(sample_y);
        const int fx = (int)((sample_x - sx0) * 256.0f);
        const int fy = (int)((sample_y - sy0) * 256.0f);
        int sx1 = sx0 + 1, sy1 = sy0 + 1;
        if (sx0 < 0) sx0 = 0;
        if (sy0 < 0) sy0 = 0;
        if (sx1 < 0) sx1 = 0;
        if (sy1 < 0) sy1 = 0;
        if (sx0 >= src->width) sx0 = src->width - 1;
        if (sx1 >= src->width) sx1 = src->width - 1;
        if (sy0 >= src->height) sy0 = src->height - 1;
        if (sy1 >= src->height) sy1 = src->height - 1;
        const uint32_t p00 = *(const uint32_t *)(src->pixels + sy0 * src->stride + sx0 * 4);
        const uint32_t p10 = *(const uint32_t *)(src->pixels + sy0 * src->stride + sx1 * 4);
        const uint32_t p01 = *(const uint32_t *)(src->pixels + sy1 * src->stride + sx0 * 4);
        const uint32_t p11 = *(const uint32_t *)(src->pixels + sy1 * src->stride + sx1 * 4);
        p = 0;
        for (int shift = 0; shift < 32; shift += 8) {
          const int top = ((p00 >> shift) & 255) * (256 - fx) + ((p10 >> shift) & 255) * fx;
          const int bottom = ((p01 >> shift) & 255) * (256 - fx) + ((p11 >> shift) & 255) * fx;
          p |= (uint32_t)((top * (256 - fy) + bottom * fy + 32768) >> 16) << shift;
        }
      }
      const uint32_t a = ((p >> 24) * ta + 127) / 255;
      const uint32_t r = (((p >> 16) & 255) * tr * ta + 32512) / 65025;
      const uint32_t g = (((p >> 8) & 255) * tg * ta + 32512) / 65025;
      const uint32_t b = ((p & 255) * tb * ta + 32512) / 65025;
      d[px] = blend(d[px], (a << 24) | (r << 16) | (g << 8) | b);
    }
  }
}

void sr_blit_scaled(Surface *dst, const Surface *src, float src_x, float src_y, float src_w, float src_h,
                    int dst_x, int dst_y, int dst_w, int dst_h, uint32_t tint) {
  sr_blit_scaled_impl(dst, src, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, tint, 1);
}

void sr_blit_scaled_nearest(Surface *dst, const Surface *src, float src_x, float src_y, float src_w, float src_h,
                            int dst_x, int dst_y, int dst_w, int dst_h, uint32_t tint) {
  sr_blit_scaled_impl(dst, src, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, tint, 0);
}

void sr_blit_many(Surface *dst, const BlitItem *items, int count, uint32_t tint) {
  for (int index = 0; index < count; ++index) {
    const BlitItem *item = &items[index];
    sr_blit_scaled_impl(
      dst, item->surface,
      item->source_x, item->source_y, item->source_width, item->source_height,
      item->destination_x, item->destination_y,
      item->destination_width, item->destination_height, tint, 1);
  }
}

void sr_blit_transform(Surface *dst, const Surface *src, float src_x, float src_y, float src_w, float src_h,
                       float dst_x, float dst_y, float dst_w, float dst_h,
                       float origin_x, float origin_y, float rotation, uint32_t tint) {
  if (dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0) return;
  const int flip_x = src_w < 0;
  const int flip_y = src_h < 0;
  if (flip_x) src_w = -src_w;
  if (flip_y) src_h = -src_h;
  const float radians = rotation * (float)M_PI / 180.0f;
  const float cs = cosf(radians);
  const float sn = sinf(radians);
  const float corners[4][2] = {
    {-origin_x, -origin_y}, {dst_w - origin_x, -origin_y},
    {dst_w - origin_x, dst_h - origin_y}, {-origin_x, dst_h - origin_y},
  };
  float min_x = INFINITY, min_y = INFINITY, max_x = -INFINITY, max_y = -INFINITY;
  for (int i = 0; i < 4; ++i) {
    const float x = dst_x + corners[i][0] * cs - corners[i][1] * sn;
    const float y = dst_y + corners[i][0] * sn + corners[i][1] * cs;
    min_x = fminf(min_x, x); min_y = fminf(min_y, y);
    max_x = fmaxf(max_x, x); max_y = fmaxf(max_y, y);
  }
  int x0 = (int)floorf(min_x), y0 = (int)floorf(min_y);
  int x1 = (int)ceilf(max_x), y1 = (int)ceilf(max_y);
  if (x0 < dst->clip_x0) x0 = dst->clip_x0;
  if (y0 < dst->clip_y0) y0 = dst->clip_y0;
  if (x1 > dst->clip_x1) x1 = dst->clip_x1;
  if (y1 > dst->clip_y1) y1 = dst->clip_y1;
  const uint32_t ta = tint >> 24;
  const uint32_t tr = (tint >> 16) & 255;
  const uint32_t tg = (tint >> 8) & 255;
  const uint32_t tb = tint & 255;
  for (int y = y0; y < y1; ++y) {
    uint32_t *out = (uint32_t *)(dst->pixels + y * dst->stride);
    for (int x = x0; x < x1; ++x) {
      const float rx = x + .5f - dst_x;
      const float ry = y + .5f - dst_y;
      const float local_x = rx * cs + ry * sn + origin_x;
      const float local_y = -rx * sn + ry * cs + origin_y;
      if (local_x < 0 || local_x >= dst_w || local_y < 0 || local_y >= dst_h) continue;
      const float u = flip_x ? 1.0f - local_x / dst_w : local_x / dst_w;
      const float v = flip_y ? 1.0f - local_y / dst_h : local_y / dst_h;
      const float sample_x = src_x + u * src_w - .5f;
      const float sample_y = src_y + v * src_h - .5f;
      int sx0 = (int)floorf(sample_x), sy0 = (int)floorf(sample_y);
      const int fx = (int)((sample_x - sx0) * 256.0f);
      const int fy = (int)((sample_y - sy0) * 256.0f);
      int sx1 = sx0 + 1, sy1 = sy0 + 1;
      if (sx0 < 0) sx0 = 0;
      if (sy0 < 0) sy0 = 0;
      if (sx1 < 0) sx1 = 0;
      if (sy1 < 0) sy1 = 0;
      if (sx0 >= src->width) sx0 = src->width - 1;
      if (sx1 >= src->width) sx1 = src->width - 1;
      if (sy0 >= src->height) sy0 = src->height - 1;
      if (sy1 >= src->height) sy1 = src->height - 1;
      const uint32_t p00 = *(const uint32_t *)(src->pixels + sy0 * src->stride + sx0 * 4);
      const uint32_t p10 = *(const uint32_t *)(src->pixels + sy0 * src->stride + sx1 * 4);
      const uint32_t p01 = *(const uint32_t *)(src->pixels + sy1 * src->stride + sx0 * 4);
      const uint32_t p11 = *(const uint32_t *)(src->pixels + sy1 * src->stride + sx1 * 4);
      uint32_t p = 0;
      for (int shift = 0; shift < 32; shift += 8) {
        const int top = ((p00 >> shift) & 255) * (256 - fx) + ((p10 >> shift) & 255) * fx;
        const int bottom = ((p01 >> shift) & 255) * (256 - fx) + ((p11 >> shift) & 255) * fx;
        p |= (uint32_t)((top * (256 - fy) + bottom * fy + 32768) >> 16) << shift;
      }
      const uint32_t a = ((p >> 24) * ta + 127) / 255;
      const uint32_t r = (((p >> 16) & 255) * tr * ta + 32512) / 65025;
      const uint32_t g = (((p >> 8) & 255) * tg * ta + 32512) / 65025;
      const uint32_t b = ((p & 255) * tb * ta + 32512) / 65025;
      out[x] = blend(out[x], (a << 24) | (r << 16) | (g << 8) | b);
    }
  }
}

static inline uint8_t clamp_u8(int value) {
  return value < 0 ? 0 : (value > 255 ? 255 : (uint8_t)value);
}

static uint8_t engaged_gamma_lut[256];
static uint8_t driver_enhance_lut[256];
static int camera_luts_initialized = 0;

static void init_camera_luts(void) {
  if (camera_luts_initialized) return;
  for (int value = 0; value < 256; ++value) {
    engaged_gamma_lut[value] = clamp_u8((int)roundf(powf(value / 255.0f, 1.0f / 1.28f) * 255));
    float enhanced = fminf(fmaxf(value / 255.0f + .15f, 0), 1);
    enhanced = fminf(fmaxf((enhanced - .5f) * .88f + .5f, 0), 1);
    enhanced = enhanced * enhanced * (3 - 2 * enhanced);
    driver_enhance_lut[value] = clamp_u8((int)roundf(powf(enhanced, .8f) * 255));
  }
  camera_luts_initialized = 1;
}

void sr_draw_nv12_crop(Surface *dst, const uint8_t *data, int frame_width, int frame_height,
                       int stride, int uv_offset, float source_x, float source_y,
                       float source_width, float source_height,
                       int dx, int dy, int dw, int dh, int flip_x, int engaged, int enhance_driver) {
  if (!data || dw <= 0 || dh <= 0) return;
  init_camera_luts();
  for (int y = 0; y < dh; ++y) {
    const int py = dy + y;
    if (py < dst->clip_y0 || py >= dst->clip_y1) continue;
    int sy = (int)(source_y + ((y + .5f) * source_height) / dh);
    if (sy < 0) sy = 0;
    if (sy >= frame_height) sy = frame_height - 1;
    uint32_t *out = (uint32_t *)(dst->pixels + py * dst->stride);
    for (int x = 0; x < dw; ++x) {
      const int px = dx + x;
      if (px < dst->clip_x0 || px >= dst->clip_x1) continue;
      int sx = (int)(source_x + ((x + .5f) * source_width) / dw);
      if (sx < 0) sx = 0;
      if (sx >= frame_width) sx = frame_width - 1;
      if (flip_x) sx = frame_width - 1 - sx;
      const int yy = data[sy * stride + sx];
      const int uv_index = uv_offset + (sy / 2) * stride + (sx & ~1);
      const int u = (int)data[uv_index] - 128;
      const int v = (int)data[uv_index + 1] - 128;
      int r = yy + ((359 * v) >> 8);
      int g = yy - ((88 * u + 183 * v) >> 8);
      int b = yy + ((454 * u) >> 8);
      r = clamp_u8(r); g = clamp_u8(g); b = clamp_u8(b);
      if (engaged) {
        const int gray = (77 * r + 150 * g + 29 * b) >> 8;
        r = (gray * 4 + r) / 5;
        g = (gray * 4 + g) / 5;
        b = (gray * 4 + b) / 5;
        r = clamp_u8(((r - 128) * 307 >> 8) + 128);
        g = clamp_u8(((g - 128) * 307 >> 8) + 128);
        b = clamp_u8(((b - 128) * 307 >> 8) + 128);
        r = engaged_gamma_lut[r];
        g = engaged_gamma_lut[g];
        b = engaged_gamma_lut[b];
      } else {
        r = r * 217 / 255;
        g = g * 217 / 255;
        b = b * 217 / 255;
      }
      if (enhance_driver) {
        r = driver_enhance_lut[r];
        g = driver_enhance_lut[g];
        b = driver_enhance_lut[b];
      }
      out[px] = 0xff000000U | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
    }
  }
}

void sr_draw_nv12(Surface *dst, const uint8_t *data, int frame_width, int frame_height,
                  int stride, int uv_offset, int dx, int dy, int dw, int dh,
                  int flip_x, int engaged, int enhance_driver) {
  sr_draw_nv12_crop(dst, data, frame_width, frame_height, stride, uv_offset,
                    0, 0, frame_width, frame_height, dx, dy, dw, dh,
                    flip_x, engaged, enhance_driver);
}

// Representative on-road overlay submitted in one call, avoiding per-primitive
// Python/C transitions in the benchmark and in the intended production design.
void sr_demo_frame(Surface *s, int frame) {
  sr_clear(s, 0xff101820U);
  sr_gradient_v(s, 0, 120, s->width, 120, 0x00101820U, 0xd0000000U);

  Point lane_l[] = {{185, 240}, {235, 95}, {247, 95}, {222, 240}};
  Point lane_r[] = {{314, 240}, {279, 95}, {291, 95}, {351, 240}};
  sr_triangle(s, lane_l[0], lane_l[1], lane_l[2], 0xb000d090U);
  sr_triangle(s, lane_l[0], lane_l[2], lane_l[3], 0xb000d090U);
  sr_triangle(s, lane_r[0], lane_r[1], lane_r[2], 0xb000d090U);
  sr_triangle(s, lane_r[0], lane_r[2], lane_r[3], 0xb000d090U);

  const int pulse = frame % 20;
  sr_circle(s, 472, 34, 18 + pulse / 8, 0xe020c060U);
  sr_rect(s, 12, 12, 92, 54, 0xd0181818U);
  sr_rect(s, 20, 22, 55 + pulse, 8, 0xfff0f0f0U);
  sr_rect(s, 20, 40, 34, 8, 0xffa0a0a0U);
}
