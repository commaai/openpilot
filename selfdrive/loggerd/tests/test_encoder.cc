#include <assert.h>
#include <unistd.h>
#include <zmq.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/times.h>
#include <memory>
#include <signal.h>
#include "common/timing.h"
#include "common/visionipc.h"
#include "encoder.h"
#include "raw_logger.h"
#define MAIN_FPS 20
#define MAIN_BITRATE 5000000
#define QCAM_BITRATE 128000
const int segment_length = 60;

enum TestType {
  typeEncoder = 0x01,
  typeRawLogger = 0x02,
  typeAll = typeEncoder | typeRawLogger
};

static clock_t lastCPU, lastSysCPU, lastUserCPU;
static int numProcessors;

volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

void cpuusage_init() {
  FILE *file;
  struct tms timeSample;
  char line[128];

  lastCPU = times(&timeSample);
  lastSysCPU = timeSample.tms_stime;
  lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  numProcessors = 0;
  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "processor", 9) == 0) numProcessors++;
  }
  fclose(file);
}

double cupusage_get() {
  struct tms timeSample;
  clock_t now;
  double percent;

  now = times(&timeSample);
  if (now <= lastCPU || timeSample.tms_stime < lastSysCPU ||
      timeSample.tms_utime < lastUserCPU) {
    percent = -1.0;
  } else {
    percent = (timeSample.tms_stime - lastSysCPU) +
              (timeSample.tms_utime - lastUserCPU);
    percent /= (now - lastCPU);
    percent /= numProcessors;
    percent *= 100;
  }
  lastCPU = now;
  lastSysCPU = timeSample.tms_stime;
  lastUserCPU = timeSample.tms_utime;
  return percent;
}

LogCameraInfo cameras_logged[LOG_CAMERA_ID_MAX] = {
  [LOG_CAMERA_ID_FCAMERA] = {
    .stream_type = VISION_STREAM_YUV,
    .filename = "fcamera.hevc",
    .frame_packet_name = "frame",
    .encode_idx_name = "encodeIdx",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true
  },
  [LOG_CAMERA_ID_QCAMERA] = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = QCAM_BITRATE,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
  },
};

// USAGE:
// test_encoder 1 
//   test encoder
// test_encoder 2
//   test rawlogger
// test_encoder
//   test encoder&rawlogger
int main(int argc, char *argv[]) {
  signal(SIGINT, (sighandler_t)set_do_exit);

  TestType test_type = typeAll;
  if (argc > 1) {
    int t = strtol(argv[1], nullptr, 10);
    if (t == 1) {
      test_type = typeEncoder;
      printf(" test Encoder\n");
    } else if (t == 2){
      test_type = typeRawLogger;
      printf(" test RawLogger\n");
    }
  }
  if (test_type == typeAll) {
     printf(" test Encoder&RawLogger\n");
  }

  std::unique_ptr<EncoderState> encoder;
  std::unique_ptr<EncoderState> encoder_alt;
  std::unique_ptr<RawLogger> rawlogger;

  const char *output_path = "./output";
  mkdir(output_path, 0777);

  VisionStream stream;
  VisionStreamBufs buf_info;
  while (!do_exit) {
    if (visionstream_init(&stream, VISION_STREAM_YUV, false, &buf_info) != 0) {
      printf("visionstream fail\n");
      usleep(100000);
      continue;
    }
    break;
  }

  if (do_exit) return 0;

  if (test_type & typeEncoder) {
    encoder.reset(new EncoderState);
    encoder_alt.reset(new EncoderState);
    encoder_init(encoder.get(), &cameras_logged[LOG_CAMERA_ID_FCAMERA], buf_info.width, buf_info.height);
    encoder_init(encoder_alt.get(), &cameras_logged[LOG_CAMERA_ID_QCAMERA], buf_info.width, buf_info.height);

    encoder_open(encoder.get(), output_path);
    encoder_open(encoder_alt.get(), output_path);
  }
  if (test_type & typeRawLogger) {
    rawlogger.reset(new RawLogger("prcamera", buf_info.width, buf_info.height, MAIN_FPS));
    rawlogger->Rotate(output_path, 1);
  }
  
  cpuusage_init();
  double t1 = millis_since_boot();
  
  int cnt = 0;
  for (; cnt < segment_length * MAIN_FPS && !do_exit; cnt++) {
    VIPCBufExtra extra;
    VIPCBuf *buf = visionstream_get(&stream, &extra);
    if (buf == NULL) {
      printf("visionstream get failed\n");
      break;
    }
    uint8_t *y = (uint8_t *)buf->addr;
    uint8_t *u = y + (buf_info.width * buf_info.height);
    uint8_t *v = u + (buf_info.width / 2) * (buf_info.height / 2);
    if (encoder) {
      encoder_encode_frame(encoder.get(), y, u, v, &extra);
      encoder_encode_frame(encoder_alt.get(), y, u, v, &extra);
    }
    if (rawlogger) {
      double r1 = millis_since_boot();
      rawlogger->LogFrame(cnt, y, u, v, nullptr);
      double r2 = millis_since_boot();
      printf("rawlogger->logFrame %d %.2f\n", cnt, (r2 - r1));
    }
  }
  double t2 = millis_since_boot();
  printf("total time: %.0f ms, cnt: %d, avg: %.0fms/frame, avg cpu:%.2f\n", t2 - t1, cnt, (t2 - t1) / cnt, cupusage_get());

  visionstream_destroy(&stream);

  if (encoder) {
    encoder_close(encoder.get());
    encoder_close(encoder_alt.get());
    encoder_destroy(encoder.get());
    encoder_destroy(encoder_alt.get());
  }
  if (rawlogger) {
    rawlogger->Close();
  }
  return 0;
}
