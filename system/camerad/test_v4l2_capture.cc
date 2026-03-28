// Standalone V4L2 capture test for mainline CAMSS on comma 3X
// Build: g++ -O2 -o test_v4l2_capture test_v4l2_capture.cc
// Run: ./test_v4l2_capture

#include <cstdint>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/media.h>
#include <linux/videodev2.h>
#include <linux/v4l2-subdev.h>

#include <string>
#include <vector>

static volatile bool running = true;
static void sighandler(int) { running = false; }

// *** Media controller helpers ***

static int find_subdev_by_name(const char *name) {
  for (int i = 0; i < 30; i++) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/video4linux/v4l-subdev%d/name", i);
    FILE *f = fopen(path, "r");
    if (!f) break;
    char buf[128] = {};
    fgets(buf, sizeof(buf), f);
    fclose(f);
    // Strip newline
    char *nl = strchr(buf, '\n');
    if (nl) *nl = 0;
    if (strstr(buf, name) == buf) {
      snprintf(path, sizeof(path), "/dev/v4l-subdev%d", i);
      return open(path, O_RDWR);
    }
  }
  return -1;
}

static int subdev_set_fmt(int fd, int pad, uint32_t code, uint32_t w, uint32_t h) {
  struct v4l2_subdev_format fmt = {};
  fmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
  fmt.pad = pad;
  fmt.format.code = code;
  fmt.format.width = w;
  fmt.format.height = h;
  fmt.format.field = V4L2_FIELD_NONE;
  fmt.format.colorspace = V4L2_COLORSPACE_SRGB;
  return ioctl(fd, VIDIOC_SUBDEV_S_FMT, &fmt);
}

static void setup_media_link(int media_fd, const char *src_name, int src_pad,
                             const char *sink_name, int sink_pad) {
  // Use MEDIA_IOC_ENUM_LINKS approach via entity enumeration
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;

    if (strstr(ent.name, src_name)) {
      // Found source entity, enumerate its links
      struct media_links_enum lenum = {};
      lenum.entity = ent.id;
      auto links = std::vector<struct media_link_desc>(ent.links);
      auto pads = std::vector<struct media_pad_desc>(ent.pads);
      lenum.links = links.data();
      lenum.pads = pads.data();
      if (ioctl(media_fd, MEDIA_IOC_ENUM_LINKS, &lenum) < 0) continue;

      for (uint32_t i = 0; i < ent.links; i++) {
        if (links[i].source.index == (uint32_t)src_pad) {
          // Check if sink matches
          struct media_entity_desc sink_ent = {};
          sink_ent.id = links[i].sink.entity;
          if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &sink_ent) == 0) {
            if (strstr(sink_ent.name, sink_name) && links[i].sink.index == (uint32_t)sink_pad) {
              if (links[i].flags & MEDIA_LNK_FL_IMMUTABLE) return;
              links[i].flags |= MEDIA_LNK_FL_ENABLED;
              ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &links[i]);
              printf("  link: %s:%d -> %s:%d [enabled]\n", src_name, src_pad, sink_name, sink_pad);
              return;
            }
          }
        }
      }
    }
  }
  printf("  link: %s:%d -> %s:%d [FAILED]\n", src_name, src_pad, sink_name, sink_pad);
}

// *** Camera config ***

struct CamConfig {
  const char *sensor_name;
  const char *csiphy_name;
  const char *csid_name;
  const char *vfe_pix_name;
  int video_idx;
  const char *label;
};

static const CamConfig CAMS[] = {
  {"ox03c10 16-0036", "msm_csiphy0", "msm_csid0", "msm_vfe0_pix", 3, "wide"},
  {"ox03c10 16-0010", "msm_csiphy1", "msm_csid1", "msm_vfe1_pix", 7, "road"},
};

static const uint32_t MBUS_SGRBG12 = 0x3012;  // MEDIA_BUS_FMT_SGRBG12_1X12

int main(int argc, char **argv) {
  signal(SIGINT, sighandler);
  signal(SIGTERM, sighandler);

  int num_cams = (argc > 1) ? atoi(argv[1]) : 2;
  if (num_cams > 2) num_cams = 2;
  int num_frames = (argc > 2) ? atoi(argv[2]) : 100;

  printf("V4L2 capture test: %d camera(s), %d frames\n", num_cams, num_frames);

  // Open media device
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) { perror("open media0"); return 1; }

  // Per-camera state
  struct CamState {
    int video_fd = -1;
    int sensor_fd = -1;
    void *bufs[18] = {};
    size_t buf_lens[18] = {};
    int buf_count = 0;
    uint32_t frame_count = 0;
    uint32_t stride = 0;
  };
  std::vector<CamState> states(num_cams);

  // Set up each camera using media-ctl (handles format negotiation correctly)
  for (int c = 0; c < num_cams; c++) {
    const auto &cfg = CAMS[c];
    printf("\n=== Setting up %s camera (%s) ===\n", cfg.label, cfg.sensor_name);

    char cmd[512];
    // Media links
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 -l '\"%s\":1->\"%s\":0[1]'", cfg.csiphy_name, cfg.csid_name);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 -l '\"%s\":4->\"%s\":0[1]'", cfg.csid_name, cfg.vfe_pix_name);
    system(cmd);

    // Formats
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":0[fmt:SGRBG12_1X12/1928x1224]'", cfg.sensor_name);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":1[fmt:SGRBG12_1X12/1928x1224]'", cfg.csiphy_name);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":0[fmt:SGRBG12_1X12/1928x1224]'", cfg.csid_name);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":4[fmt:SGRBG12_1X12/1928x1224]'", cfg.csid_name);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":0[fmt:SGRBG12_1X12/1928x1224]'", cfg.vfe_pix_name);
    system(cmd);

    // Open video device
    char vpath[32];
    snprintf(vpath, sizeof(vpath), "/dev/video%d", cfg.video_idx);
    states[c].video_fd = open(vpath, O_RDWR);
    if (states[c].video_fd < 0) { perror(vpath); return 1; }

    // Set NV12 format
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = 1920;
    fmt.fmt.pix_mp.height = 1224;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.num_planes = 1;
    ioctl(states[c].video_fd, VIDIOC_S_FMT, &fmt);
    ioctl(states[c].video_fd, VIDIOC_G_FMT, &fmt);
    states[c].stride = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    printf("  format: %dx%d stride=%d\n", fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height, states[c].stride);

    // Request buffers
    struct v4l2_requestbuffers reqbufs = {};
    reqbufs.count = 4;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    if (ioctl(states[c].video_fd, VIDIOC_REQBUFS, &reqbufs) < 0) { perror("REQBUFS"); return 1; }
    states[c].buf_count = reqbufs.count;

    // mmap and queue buffers
    for (uint32_t i = 0; i < reqbufs.count; i++) {
      struct v4l2_buffer buf = {};
      struct v4l2_plane planes[1] = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      buf.m.planes = planes;
      buf.length = 1;
      ioctl(states[c].video_fd, VIDIOC_QUERYBUF, &buf);

      states[c].buf_lens[i] = planes[0].length;
      states[c].bufs[i] = mmap(NULL, planes[0].length, PROT_READ | PROT_WRITE,
                               MAP_SHARED, states[c].video_fd, planes[0].m.mem_offset);

      ioctl(states[c].video_fd, VIDIOC_QBUF, &buf);
    }

    // Sensor exposure
    states[c].sensor_fd = find_subdev_by_name(cfg.sensor_name);
    if (states[c].sensor_fd >= 0) {
      struct v4l2_control ctrl;
      ctrl.id = V4L2_CID_EXPOSURE;
      ctrl.value = 500;
      ioctl(states[c].sensor_fd, VIDIOC_S_CTRL, &ctrl);
      ctrl.id = V4L2_CID_ANALOGUE_GAIN;
      ctrl.value = 512;
      ioctl(states[c].sensor_fd, VIDIOC_S_CTRL, &ctrl);
    }

    // Start streaming
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(states[c].video_fd, VIDIOC_STREAMON, &type) < 0) {
      perror("STREAMON");
      return 1;
    }
    printf("  streaming started\n");
  }

  // Capture loop
  printf("\n=== Capturing %d frames ===\n", num_frames);
  int total_frames = 0;

  while (running && total_frames < num_frames * num_cams) {
    std::vector<struct pollfd> fds;
    for (int c = 0; c < num_cams; c++) {
      fds.push_back({.fd = states[c].video_fd, .events = POLLIN});
    }

    int ret = poll(fds.data(), fds.size(), 2000);
    if (ret <= 0) {
      if (ret == 0) printf("poll timeout\n");
      continue;
    }

    for (int c = 0; c < num_cams; c++) {
      if (!(fds[c].revents & POLLIN)) continue;

      struct v4l2_buffer buf = {};
      struct v4l2_plane planes[1] = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.m.planes = planes;
      buf.length = 1;

      if (ioctl(states[c].video_fd, VIDIOC_DQBUF, &buf) < 0) continue;

      states[c].frame_count++;
      total_frames++;

      // Simple AE: sample Y values and adjust exposure
      if (states[c].frame_count % 10 == 0) {
        uint8_t *y = (uint8_t *)states[c].bufs[buf.index];
        uint64_t sum = 0;
        int samples = 0;
        for (int row = 200; row < 1000; row += 50) {
          for (int col = 200; col < 1700; col += 50) {
            sum += y[row * states[c].stride + col];
            samples++;
          }
        }
        float avg_y = (float)sum / samples;
        float target = 80.0f;
        float ratio = target / (avg_y + 1.0f);

        struct v4l2_control ctrl;
        ctrl.id = V4L2_CID_EXPOSURE;
        ioctl(states[c].sensor_fd, VIDIOC_G_CTRL, &ctrl);
        int new_exp = (int)(ctrl.value * ratio);
        if (new_exp < 2) new_exp = 2;
        if (new_exp > 2016) new_exp = 2016;
        ctrl.value = new_exp;
        ioctl(states[c].sensor_fd, VIDIOC_S_CTRL, &ctrl);

        printf("%s: frame %d, avg_y=%.0f, exp=%d\n", CAMS[c].label, states[c].frame_count, avg_y, new_exp);
      }

      // Save first frame as file
      if (states[c].frame_count == 5) {
        char fname[64];
        snprintf(fname, sizeof(fname), "/tmp/v4l2_%s.nv12", CAMS[c].label);
        FILE *f = fopen(fname, "wb");
        if (f) {
          fwrite(states[c].bufs[buf.index], 1, states[c].buf_lens[buf.index], f);
          fclose(f);
          printf("%s: saved frame to %s (%zu bytes)\n", CAMS[c].label, fname, states[c].buf_lens[buf.index]);
        }
      }

      // Requeue
      ioctl(states[c].video_fd, VIDIOC_QBUF, &buf);
    }
  }

  // Cleanup
  printf("\n=== Results ===\n");
  for (int c = 0; c < num_cams; c++) {
    printf("%s: %d frames captured\n", CAMS[c].label, states[c].frame_count);
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(states[c].video_fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < states[c].buf_count; i++) {
      if (states[c].bufs[i]) munmap(states[c].bufs[i], states[c].buf_lens[i]);
    }
    close(states[c].video_fd);
    if (states[c].sensor_fd >= 0) close(states[c].sensor_fd);
  }
  close(media_fd);

  return 0;
}
