#include "sde_rotator.h"
#include "third_party/linux/include/msm_media_info.h"
#include "common/swaglog.h"

#ifndef V4L2_PIX_FMT_NV12_UBWC
#define V4L2_PIX_FMT_NV12_UBWC v4l2_fourcc('Q', '1', '2', '8')
#endif

static void checked_ioctl(int fd, unsigned long request, void *argp) {
  int ret = util::safe_ioctl(fd, request, argp);
  if (ret != 0) {
    LOGE("checked_ioctl failed with error %d (%d %lx %p)", errno, fd, request, argp);
    assert(0);
  }
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR
  };
  checked_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}


SdeRotator::SdeRotator() {
  memset(&fmt_cap, 0, sizeof(fmt_cap));
  memset(&fmt_out, 0, sizeof(fmt_out));
  memset(&cached_cap_buf, 0, sizeof(cached_cap_buf));
  fd = HANDLE_EINTR(open("/dev/video2", O_RDWR|O_NONBLOCK));
  assert(fd >= 0);
  LOG("opened rotator device fd=%d", fd);
  pfd = { .fd = fd, .events = POLLIN | POLLRDNORM, .revents = 0 };
  struct v4l2_capability cap;
  memset(&cap, 0, sizeof(cap));
  checked_ioctl(fd, VIDIOC_QUERYCAP, &cap); // check if this needed
}


SdeRotator::~SdeRotator() {
  cleanup();
}

/**
 * @brief Configures the SdeRotator operation for the specified frame dimensions.
 *
 * This function sets up the video output and capture formats, allocates and manages
 * the necessary buffers, and starts streaming on both output and capture devices.
 * It ensures that any previously allocated ION buffer is freed and unmapped if the
 * size has changed, and allocates a new buffer for the current configuration.
 * The function also queries and caches the capture buffer information after allocation.
 *
 * @param width  The width of the video frame to configure.
 * @param height The height of the video frame to configure.
 * @return int Returns 0 on successful configuration, or asserts on failure.
 */
int SdeRotator::config_ubwc_to_nv12_op(int width, int height) {
  LOG("Configuring rotator for width=%d height=%d", width, height);
  queued = false;
  fmt_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt_out.fmt.pix.width       = width;
  fmt_out.fmt.pix.height      = height;
  fmt_out.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12_UBWC;
  fmt_out.fmt.pix.field       = V4L2_FIELD_NONE;
  checked_ioctl(fd, VIDIOC_S_FMT, &fmt_out);

  fmt_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt_cap.fmt.pix.width       = width;
  fmt_cap.fmt.pix.height      = height;
  fmt_cap.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
  fmt_cap.fmt.pix.field       = V4L2_FIELD_NONE;
  checked_ioctl(fd, VIDIOC_S_FMT, &fmt_cap);

  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT, 1);

  // Free previous ION buffer and unmap if size changed
  if (vision_buf.fd >= 0) {
    if (linear_ptr && mapped_size) {
      munmap(linear_ptr, mapped_size);
      linear_ptr = nullptr;
      mapped_size = 0;
    }
    vision_buf.free();
  }
  vision_buf.allocate(fmt_cap.fmt.pix.sizeimage);
  vision_buf.width = width;
  vision_buf.height = height;

  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, 1);

  // Query and cache capture buffer info (only needed after (re)alloc)
  memset(&cached_cap_buf, 0, sizeof(cached_cap_buf));
  cached_cap_buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cached_cap_buf.memory = V4L2_MEMORY_USERPTR;
  cached_cap_buf.index  = 0;
  checked_ioctl(fd, VIDIOC_QUERYBUF, &cached_cap_buf);

  // Only streamon after (re)configuration
  enum v4l2_buf_type t;
  t = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  checked_ioctl(fd, VIDIOC_STREAMON, &t);
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  checked_ioctl(fd, VIDIOC_STREAMON, &t);

  return 0;
}


int SdeRotator::cleanup() {
    int err = 0;

    if (linear_ptr && mapped_size) {
      err = munmap(linear_ptr, mapped_size);
      linear_ptr = nullptr;
      mapped_size = 0;
    }
    if (fd >= 0) {
      err = close(fd);
      fd = -1;
    }
    vision_buf.free();
    vision_buf = VisionBuf();
    queued = false;
    return err;
}

/**
 * @brief Queues an output and capture buffer for frame rotation.
 *
 * This function configures the rotator operation if the provided width and height
 * differ from the current vision buffer dimensions. It then queues the output buffer
 * (using the provided VisionBuf file descriptor) and the capture buffer (using cached information)
 * to the V4L2 device for processing. The function marks the buffers as queued.
 *
 * @param input_frame an allocated VisionBuf for the output buffer. output is the input frame to be operated on.
 * @return int       Returns 0 on success, asserts on failure.
 */
int SdeRotator::put_frame(VisionBuf *input_frame) {
    if (input_frame->width != vision_buf.width || input_frame->height != vision_buf.height) {
      // I think Port Reconfigure events can trigger this because of a msm_vidc driver quirk where I can't set the pic_struct and bit_depth. This is a venus firmware limitation.
      // Essentially, the decoder needs to see a few frames and then it tells userspace to reconfigure the decoder which will round up the width and height to the nearest multiple of 32 and 16 respectivly.
      // This is a workaround for that and can be removed once its better understood.
      this->config_ubwc_to_nv12_op(input_frame->width, input_frame->height);
    }
    assert(input_frame->fd > 0);
    // Queue output buffer
    struct v4l2_buffer buf = {0};
    int sizeimage = fmt_out.fmt.pix.sizeimage;
    buf.type      = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    buf.memory    = V4L2_MEMORY_USERPTR;
    buf.index     = 0;
    buf.m.userptr = (unsigned long)input_frame->fd;
    buf.length    = sizeimage;
    checked_ioctl(fd, VIDIOC_QBUF, &buf);

    // Queue capture buffer (use cached info)
    struct v4l2_buffer cap_buf = cached_cap_buf;
    cap_buf.m.userptr = (unsigned long)vision_buf.fd;
    checked_ioctl(fd, VIDIOC_QBUF, &cap_buf);
    queued = true;

    return 0;
}

/**
 * @brief Retrieves a decoded video frame from the SdeRotator device.
 *
 * This function waits for a frame to become available, dequeues it from the video capture buffer,
 * queries the output buffer, and ensures the frame data is memory-mapped for access. The function
 * returns pointers to the linear frame data and its size. If the memory mapping changes or is not
 * yet established, it remaps the buffer accordingly.
 *
 * @param[out] linear_data Pointer to a pointer that will be set to the address of the linear frame data.
 * @param[out] linear_size Pointer to a variable that will be set to the size of the frame data in bytes.
 * @param[in] timeout_ms Timeout in milliseconds to wait for a frame (default: 100 ms).
 * @return 0 on success, -1 on poll timeout, and asserts on memory mapping or ioctl errors.
 */
int SdeRotator::get_frame(unsigned char **linear_data, size_t *linear_size, int timeout_ms = 100) {
  if (poll(&pfd, 1, timeout_ms) < 0) {
    LOGE("poll failed with error %d", errno);
    return -1;
  }

  struct v4l2_buffer dq = {
    .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    .memory = V4L2_MEMORY_USERPTR
  };
  checked_ioctl(fd, VIDIOC_DQBUF, &dq);
  queued = false;

  struct v4l2_buffer dqout = {
    .type   = V4L2_BUF_TYPE_VIDEO_OUTPUT,
    .memory = V4L2_MEMORY_USERPTR,
  };
  checked_ioctl(fd, VIDIOC_QUERYBUF, &dqout);

  // Only mmap if not already mapped
  if (!linear_ptr || mapped_size != dq.length) {
    LOG("mmaping linear buffer, size=%u", dq.length);
    if (linear_ptr && mapped_size)
      munmap(linear_ptr, mapped_size);

    linear_ptr = mmap(NULL, dq.length,
                      PROT_READ|PROT_WRITE,
                      MAP_SHARED,
                      vision_buf.fd, 0);
    assert(linear_ptr != MAP_FAILED);
    mapped_size = dq.length;
  }

  *linear_data = (unsigned char *)linear_ptr;
  *linear_size = dq.length;

  return 0;
}
