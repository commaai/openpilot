#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "visionipc.h"

typedef struct VisionPacketWire {
  int type;
  VisionPacketData d;
} VisionPacketWire;

int vipc_connect() {
  int err;

  int sock = socket(AF_UNIX, SOCK_SEQPACKET, 0);
  assert(sock >= 0);
  struct sockaddr_un addr = {
    .sun_family = AF_UNIX,
    .sun_path = VIPC_SOCKET_PATH,
  };
  err = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
  if (err != 0) {
    close(sock);
    return -1;
  }

  return sock;
}

static int sendrecv_with_fds(bool send, int fd, void *buf, size_t buf_size, int* fds, int num_fds,
                             int *out_num_fds) {
  int err;

  char control_buf[CMSG_SPACE(sizeof(int) * num_fds)];
  memset(control_buf, 0, CMSG_SPACE(sizeof(int) * num_fds));

  struct iovec iov = {
    .iov_base = buf,
    .iov_len = buf_size,
  };
  struct msghdr msg = {
    .msg_iov = &iov,
    .msg_iovlen = 1,
  };

  if (num_fds > 0) {
    assert(fds);

    msg.msg_control = control_buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * num_fds);
  }

  if (send) {
    if (num_fds) {
      struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
      assert(cmsg);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(int) * num_fds);
      memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * num_fds);
      // printf("send clen %d -> %d\n", num_fds, cmsg->cmsg_len);
    }
    return sendmsg(fd, &msg, 0);
  } else {
    int r = recvmsg(fd, &msg, 0);
    if (r < 0) return r;

    int recv_fds = 0;
    if (msg.msg_controllen > 0) {
      struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
      assert(cmsg);
      assert(cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
      recv_fds = (cmsg->cmsg_len - CMSG_LEN(0));
      assert(recv_fds > 0 && (recv_fds % sizeof(int)) == 0);
      recv_fds /= sizeof(int);
      // printf("recv clen %d -> %d\n", cmsg->cmsg_len, recv_fds);
      // assert(cmsg->cmsg_len == CMSG_LEN(sizeof(int) * num_fds));

      assert(fds && recv_fds <= num_fds);
      memcpy(fds, CMSG_DATA(cmsg), sizeof(int) * recv_fds);
    }

    if (msg.msg_flags) {
      for (int i=0; i<recv_fds; i++) {
        close(fds[i]);
      }
      return -1;
    }

    if (fds) {
      assert(out_num_fds);
      *out_num_fds = recv_fds;
    }
    return r;
  }
}

int vipc_recv(int fd, VisionPacket *out_p) {
  VisionPacketWire p = {0};
  VisionPacket p2 = {0};
  int ret = sendrecv_with_fds(false, fd, &p, sizeof(p), (int*)p2.fds, VIPC_MAX_FDS, &p2.num_fds);
  if (ret < 0) {
    printf("vipc_recv err: %s\n", strerror(errno));
  } else {
    p2.type = p.type;
    p2.d = p.d;
    *out_p = p2;
  }
  return ret;
}

int vipc_send(int fd, const VisionPacket *p2) {
  assert(p2->num_fds <= VIPC_MAX_FDS);

  VisionPacketWire p = {
    .type = p2->type,
    .d = p2->d,
  };
  return sendrecv_with_fds(true, fd, (void*)&p, sizeof(p), (int*)p2->fds, p2->num_fds, NULL);
}

void vipc_bufs_load(VIPCBuf *bufs, const VisionStreamBufs *stream_bufs,
                     int num_fds, const int* fds) {
  for (int i=0; i<num_fds; i++) {
    if (bufs[i].addr) {
      munmap(bufs[i].addr, bufs[i].len);
      bufs[i].addr = NULL;
      close(bufs[i].fd);
    }
    bufs[i].fd = fds[i];
    bufs[i].len = stream_bufs->buf_len;
    bufs[i].addr = mmap(NULL, bufs[i].len,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED, bufs[i].fd, 0);
    // printf("b %d %zu -> %p\n", bufs[i].fd, bufs[i].len, bufs[i].addr);
    assert(bufs[i].addr != MAP_FAILED);
  }
}


int visionstream_init(VisionStream *s, VisionStreamType type, bool tbuffer, VisionStreamBufs *out_bufs_info) {
  int err;

  memset(s, 0, sizeof(*s));

  s->last_idx = -1;

  s->ipc_fd = vipc_connect();
  if (s->ipc_fd < 0) return -1;

  VisionPacket p = {
    .type = VIPC_STREAM_SUBSCRIBE,
    .d = { .stream_sub = {
      .type = type,
      .tbuffer = tbuffer,
    }, },
  };
  err = vipc_send(s->ipc_fd, &p);
  if (err < 0) {
    close(s->ipc_fd);
    return -1;
  }

  VisionPacket rp;
  err = vipc_recv(s->ipc_fd, &rp);
  if (err <= 0) {
    close(s->ipc_fd);
    return -1;
  }
  assert(rp.type = VIPC_STREAM_BUFS);
  assert(rp.d.stream_bufs.type == type);

  s->bufs_info = rp.d.stream_bufs;

  s->num_bufs = rp.num_fds;
  s->bufs = calloc(s->num_bufs, sizeof(VIPCBuf));
  assert(s->bufs);

  vipc_bufs_load(s->bufs, &rp.d.stream_bufs, s->num_bufs, rp.fds);

  if (out_bufs_info) {
    *out_bufs_info = s->bufs_info;
  }

  return 0;
}

void visionstream_release(VisionStream *s) {
  int err;
  if (s->last_idx >= 0) {
    VisionPacket rep = {
      .type = VIPC_STREAM_RELEASE,
      .d = { .stream_rel = {
        .type = s->last_type,
        .idx = s->last_idx,
      }}
    };
    err = vipc_send(s->ipc_fd, &rep);
    s->last_idx = -1;
  }
}

VIPCBuf* visionstream_get(VisionStream *s, VIPCBufExtra *out_extra) {
  int err;

  VisionPacket rp;
  err = vipc_recv(s->ipc_fd, &rp);
  if (err <= 0) {
    return NULL;
  }
  assert(rp.type == VIPC_STREAM_ACQUIRE);

  if (s->last_idx >= 0) {
    VisionPacket rep = {
      .type = VIPC_STREAM_RELEASE,
      .d = { .stream_rel = {
        .type = s->last_type,
        .idx = s->last_idx,
      }}
    };
    err = vipc_send(s->ipc_fd, &rep);
    if (err <= 0) {
      return NULL;
    }
  }

  s->last_type = rp.d.stream_acq.type;
  s->last_idx = rp.d.stream_acq.idx;
  assert(s->last_idx < s->num_bufs);

  if (out_extra) {
    *out_extra = rp.d.stream_acq.extra;
  }

  return &s->bufs[s->last_idx];
}

void visionstream_destroy(VisionStream *s) {
  int err;

  if (s->last_idx >= 0) {
    VisionPacket rep = {
      .type = VIPC_STREAM_RELEASE,
      .d = { .stream_rel = {
        .type = s->last_type,
        .idx = s->last_idx,
      }}
    };
    err = vipc_send(s->ipc_fd, &rep);
    s->last_idx = -1;
  }

  for (int i=0; i<s->num_bufs; i++) {
    if (s->bufs[i].addr) {
      munmap(s->bufs[i].addr, s->bufs[i].len);
      s->bufs[i].addr = NULL;
      close(s->bufs[i].fd);
    }
  }
  if (s->bufs) free(s->bufs);
  close(s->ipc_fd);
}
