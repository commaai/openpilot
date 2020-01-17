

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