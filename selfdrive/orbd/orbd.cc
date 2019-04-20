#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <sys/resource.h>

#include "common/visionipc.h"
#include "common/swaglog.h"

#include "extractor.h"

#ifdef DSP
#include "dsp/gen/calculator.h"
#else
#include "turbocv.h"
#endif

#include <zmq.h>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#ifndef PATH_MAX
#include <linux/limits.h>
#endif

volatile int do_exit = 0;

static void set_do_exit(int sig) {
  do_exit = 1;
}

int main(int argc, char *argv[]) {
  int err;
  setpriority(PRIO_PROCESS, 0, -13);
  printf("starting orbd\n");

#ifdef DSP
  uint32_t test_leet = 0;
  char my_path[PATH_MAX+1];
  memset(my_path, 0, sizeof(my_path));

  ssize_t len = readlink("/proc/self/exe", my_path, sizeof(my_path));
  assert(len > 5);
  my_path[len-5] = '\0';
  LOGW("running from %s with PATH_MAX %d", my_path, PATH_MAX);

  char adsp_path[PATH_MAX+1];
  snprintf(adsp_path, PATH_MAX, "ADSP_LIBRARY_PATH=%s/dsp/gen", my_path);
  assert(putenv(adsp_path) == 0);

  assert(calculator_init(&test_leet) == 0);
  assert(test_leet == 0x1337);
  LOGW("orbd init complete");
#else
  init_gpyrs();
#endif

  signal(SIGINT, (sighandler_t) set_do_exit);
  signal(SIGTERM, (sighandler_t) set_do_exit);

  void *ctx = zmq_ctx_new();

  void *orb_features_sock = zmq_socket(ctx, ZMQ_PUB);
  assert(orb_features_sock);
  zmq_bind(orb_features_sock, "tcp://*:8058");

  void *orb_features_summary_sock = zmq_socket(ctx, ZMQ_PUB);
  assert(orb_features_summary_sock);
  zmq_bind(orb_features_summary_sock, "tcp://*:8062");

  struct orb_features *features = (struct orb_features *)malloc(sizeof(struct orb_features));
  int last_frame_id = 0;
  uint64_t frame_count = 0;

  // every other frame
  const int RATE = 2;

  VisionStream stream;
  while (!do_exit) {
    VisionStreamBufs buf_info;
    err = visionstream_init(&stream, VISION_STREAM_YUV, true, &buf_info);
    if (err) {
      printf("visionstream connect fail\n");
      usleep(100000);
      continue;
    }
    uint64_t timestamp_last_eof = 0;
    while (!do_exit) {
      VIPCBuf *buf;
      VIPCBufExtra extra;
      buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        printf("visionstream get failed\n");
        break;
      }

      // every other frame
      frame_count++;
      if ((frame_count%RATE) != 0) {
        continue;
      }

      uint64_t start = nanos_since_boot();
#ifdef DSP
      int ret = calculator_extract_and_match((uint8_t *)buf->addr, ORBD_HEIGHT*ORBD_WIDTH, (uint8_t *)features, sizeof(struct orb_features));
#else
      int ret = extract_and_match_gpyrs((uint8_t *) buf->addr, features);
#endif
      uint64_t end = nanos_since_boot();
      LOGD("total(%d): %6.2f ms to get %4d features on %d", ret, (end-start)/1000000.0, features->n_corners, extra.frame_id);
      assert(ret == 0);

      if (last_frame_id+RATE != extra.frame_id) {
        LOGW("dropped frame!");
      }

      last_frame_id = extra.frame_id;

      if (timestamp_last_eof == 0) {
        timestamp_last_eof = extra.timestamp_eof;
        continue;
      }

      int match_count = 0;

      // *** send OrbFeatures ***
      {
        // create capnp message
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());

        auto orb_features = event.initOrbFeatures();

        // set timestamps
        orb_features.setTimestampEof(extra.timestamp_eof);
        orb_features.setTimestampLastEof(timestamp_last_eof);

        // init descriptors for send
        kj::ArrayPtr<capnp::byte> descriptorsPtr = kj::arrayPtr((uint8_t *)features->des, ORBD_DESCRIPTOR_LENGTH * features->n_corners);
        orb_features.setDescriptors(descriptorsPtr);

        auto xs = orb_features.initXs(features->n_corners);
        auto ys = orb_features.initYs(features->n_corners);
        auto octaves = orb_features.initOctaves(features->n_corners);
        auto matches = orb_features.initMatches(features->n_corners);

        // copy out normalized keypoints
        for (int i = 0; i < features->n_corners; i++) {
          xs.set(i, (features->xy[i][0] * 1.0f - ORBD_WIDTH / 2) / ORBD_FOCAL);
          ys.set(i, (features->xy[i][1] * 1.0f - ORBD_HEIGHT / 2) / ORBD_FOCAL);
          octaves.set(i, features->octave[i]);
          matches.set(i, features->matches[i]);
          match_count += features->matches[i] != -1;
        }

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        zmq_send(orb_features_sock, bytes.begin(), bytes.size(), 0);
      }

      // *** send OrbFeaturesSummary ***

      {
        // create capnp message
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());

        auto orb_features_summary = event.initOrbFeaturesSummary();

        orb_features_summary.setTimestampEof(extra.timestamp_eof);
        orb_features_summary.setTimestampLastEof(timestamp_last_eof);
        orb_features_summary.setFeatureCount(features->n_corners);
        orb_features_summary.setMatchCount(match_count);
        orb_features_summary.setComputeNs(end-start);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        zmq_send(orb_features_summary_sock, bytes.begin(), bytes.size(), 0);
      }

      timestamp_last_eof = extra.timestamp_eof;
    }
  }
  visionstream_destroy(&stream);
  return 0;
}

