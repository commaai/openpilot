// Inspect the DRM/KMS ABI exported through AGNOS' /tmp/drmfd.sock broker.
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

static int receive_drm_fd(void) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) return -1;
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", "/tmp/drmfd.sock");
  if (connect(sock, (struct sockaddr *)&address, sizeof(address)) < 0) {
    close(sock);
    return -1;
  }

  char byte;
  struct iovec iov = {.iov_base = &byte, .iov_len = 1};
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr message = {
    .msg_iov = &iov,
    .msg_iovlen = 1,
    .msg_control = control,
    .msg_controllen = sizeof(control),
  };
  if (recvmsg(sock, &message, 0) < 0) {
    close(sock);
    return -1;
  }
  struct cmsghdr *header = CMSG_FIRSTHDR(&message);
  if (!header || header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) {
    close(sock);
    errno = EPROTO;
    return -1;
  }
  int fd;
  memcpy(&fd, CMSG_DATA(header), sizeof(fd));
  // Keep the broker connection alive until process exit.
  return fd;
}

static void print_fourcc(uint32_t format) {
  char name[5] = {
    format & 0xff,
    (format >> 8) & 0xff,
    (format >> 16) & 0xff,
    (format >> 24) & 0x7f,
    0,
  };
  printf("%s", name);
}

static void print_properties(int fd, uint32_t object_id, uint32_t object_type) {
  drmModeObjectProperties *properties = drmModeObjectGetProperties(fd, object_id, object_type);
  if (!properties) return;
  for (uint32_t index = 0; index < properties->count_props; ++index) {
    drmModePropertyRes *property = drmModeGetProperty(fd, properties->props[index]);
    if (!property) continue;
    printf("    property %u %s=%llu flags=0x%x", property->prop_id, property->name,
           (unsigned long long)properties->prop_values[index], property->flags);
    if (property->count_enums) {
      printf(" enums=");
      for (int enum_index = 0; enum_index < property->count_enums; ++enum_index) {
        printf("%s%s:%llu", enum_index ? "," : "", property->enums[enum_index].name,
               (unsigned long long)property->enums[enum_index].value);
      }
    }
    printf("\n");
    drmModeFreeProperty(property);
  }
  drmModeFreeObjectProperties(properties);
}

int main(void) {
  int fd = receive_drm_fd();
  if (fd < 0) {
    fprintf(stderr, "failed to receive DRM fd: %s\n", strerror(errno));
    return 1;
  }
  printf("driver=%s master=%d\n", drmGetDeviceNameFromFd2(fd), drmIsMaster(fd));
  int universal = drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  int atomic = drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
  printf("universal_planes=%d atomic=%d\n", universal, atomic);

  drmModeRes *resources = drmModeGetResources(fd);
  if (!resources) {
    fprintf(stderr, "drmModeGetResources: %s\n", strerror(errno));
    return 1;
  }
  for (int index = 0; index < resources->count_crtcs; ++index) {
    printf("crtc %u\n", resources->crtcs[index]);
    print_properties(fd, resources->crtcs[index], DRM_MODE_OBJECT_CRTC);
  }
  for (int index = 0; index < resources->count_connectors; ++index) {
    drmModeConnector *connector = drmModeGetConnector(fd, resources->connectors[index]);
    if (!connector) continue;
    printf("connector %u connected=%d modes=%d\n", connector->connector_id,
           connector->connection == DRM_MODE_CONNECTED, connector->count_modes);
    print_properties(fd, connector->connector_id, DRM_MODE_OBJECT_CONNECTOR);
    drmModeFreeConnector(connector);
  }
  drmModeFreeResources(resources);

  drmModePlaneRes *planes = drmModeGetPlaneResources(fd);
  if (!planes) {
    fprintf(stderr, "drmModeGetPlaneResources: %s\n", strerror(errno));
    return 1;
  }
  printf("planes=%u\n", planes->count_planes);
  for (uint32_t index = 0; index < planes->count_planes; ++index) {
    drmModePlane *plane = drmModeGetPlane(fd, planes->planes[index]);
    if (!plane) continue;
    printf("plane %u possible_crtcs=0x%x formats=", plane->plane_id, plane->possible_crtcs);
    for (uint32_t format_index = 0; format_index < plane->count_formats; ++format_index) {
      if (format_index) printf(",");
      print_fourcc(plane->formats[format_index]);
    }
    printf("\n");
    print_properties(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE);
    drmModeFreePlane(plane);
  }
  drmModeFreePlaneResources(planes);
  close(fd);
  return 0;
}
