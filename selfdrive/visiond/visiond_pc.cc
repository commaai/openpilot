#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>
#include <time.h>
#include <semaphore.h>
#include <signal.h>
#include <pthread.h>

#include <algorithm>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <czmq.h>
#include <capnp/serialize.h>

#include "common/version.h"
#include "common/util.h"
#include "common/timing.h"
#include "common/mat.h"
#include "common/swaglog.h"
#include "common/buffering.h"

#include "clutil.h"
#include "bufs.h"

#include "camera_eon_stream.h"

#include "model.h"

#include "cereal/gen/cpp/log.capnp.h"

#define M_PI 3.14159265358979323846

// send net input on port 9000
//#define SEND_NET_INPUT

#define MAX_CLIENTS 5

#ifdef __APPLE__
typedef void (*sighandler_t) (int);
#endif

extern "C" {
volatile int do_exit = 0;
}

namespace {
struct VisionState {

  int frame_width, frame_height;

  // cl state
  cl_device_id device_id;
  cl_context context;

  mat3 yuv_transform;
  // OpenCL buffers for storing yuv frames from Eon stream
  cl_mem yuv_cl[FRAME_BUF_COUNT];
  size_t yuv_buf_size;
  int yuv_width, yuv_height;


  ModelState model;
  ModelData model_bufs[FRAME_BUF_COUNT];

  // Protected by transform_lock.
  bool run_model;
  mat3 cur_transform;
  pthread_mutex_t transform_lock;

  DualCameraState cameras;

  zsock_t *terminate_pub;
  zsock_t *recorder_sock;
  void* recorder_sock_raw;

  zsock_t *posenet_sock;
  void* posenet_sock_raw;
};

void cl_init(VisionState *s) {
  int err;
  cl_platform_id platform_id = NULL;
  cl_uint num_devices;
  cl_uint num_platforms;

  err = clGetPlatformIDs(1, &platform_id, &num_platforms);
  assert(err == 0);
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                       &s->device_id, &num_devices);
  assert(err == 0);

  cl_print_info(platform_id, s->device_id);
  printf("\n");

  s->context = clCreateContext(NULL, 1, &s->device_id, NULL, NULL, &err);
  assert(err == 0);
}

void cl_free(VisionState *s) {
  int err;

  err = clReleaseContext(s->context);
  assert(err == 0);
}

void init_buffers(VisionState *s) {
  int err;

  s->yuv_width = s->frame_width;
  s->yuv_height = s->frame_height;
  s->yuv_buf_size = s->frame_width * s->frame_height * 3 / 2;
  for (int i=0; i<FRAME_BUF_COUNT; i++) {
    s->yuv_cl[i] = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       s->yuv_buf_size, NULL, &err);
    assert(err == 0);
  }

  s->yuv_transform = s->cameras.rear.transform;
}

void free_buffers(VisionState *s) {
  // free bufs
  for (int i=0; i<FRAME_BUF_COUNT; i++) {
    clReleaseMemObject(s->yuv_cl[i]);
  }
}

#define POSENET

#ifdef POSENET
#include "snpemodel.h"
extern const uint8_t posenet_model_data[] asm("_binary_posenet_dlc_start");
extern const uint8_t posenet_model_end[] asm("_binary_posenet_dlc_end");
const size_t posenet_model_size = posenet_model_end - posenet_model_data;
#endif

void* processing_thread(void *arg) {
  int err;
  VisionState *s = (VisionState*)arg;
  set_thread_name("processing");
  err = set_realtime_priority(1);
  LOG("setpriority returns %d", err);

  zsock_t *model_sock = zsock_new_pub("@tcp://*:8009");
  assert(model_sock);
  void *model_sock_raw = zsock_resolve(model_sock);

#ifdef SEND_NET_INPUT
  zsock_t *img_sock = zsock_new_pub("@tcp://*:9000");
  assert(img_sock);
  void *img_sock_raw = zsock_resolve(img_sock);
#else
  void *img_sock_raw = NULL;
#endif

#ifdef POSENET
  int posenet_counter = 0;
  float pose_output[12];
  float *posenet_input = (float*)malloc(2*200*532*sizeof(float));
  SNPEModel *posenet = new SNPEModel(posenet_model_data, posenet_model_size,
    pose_output, sizeof(pose_output)/sizeof(float));
#endif

  LOG("processing start!");

  for (int cnt = 0; !do_exit; cnt++) {
    int buf_idx = tbuffer_acquire(&s->cameras.rear.camera_tb);
    if (buf_idx < 0) {
      break;
    }

    double t1 = millis_since_boot();

    FrameMetadata frame_data = s->cameras.rear.camera_bufs_metadata[buf_idx];
    uint32_t frame_id = frame_data.frame_id;

    if (frame_id == -1) {
      LOGE("no frame data? wtf");
      tbuffer_release(&s->cameras.rear.camera_tb, buf_idx);
      continue;
    }

    double t2 = millis_since_boot();

    // Frames from Eon streaming is in yuv420 format
    cl_mem yuv_cl = s->yuv_cl[buf_idx];
    pthread_mutex_lock(&s->transform_lock);
    mat3 transform = s->cur_transform;
    const bool run_model_this_iter = s->run_model;
    pthread_mutex_unlock(&s->transform_lock);

    double mt1 = 0, mt2 = 0;
    if (run_model_this_iter) {
      mat3 model_transform = matmul3(s->yuv_transform, transform);
      mt1 = millis_since_boot();
      s->model_bufs[buf_idx] =
          model_eval_frame(&s->model, s->cameras.rear.q, yuv_cl, s->yuv_width, s->yuv_height,
                           model_transform, img_sock_raw);
      mt2 = millis_since_boot();
      model_publish(model_sock_raw, frame_id, model_transform, s->model_bufs[buf_idx]);
    }
    cl_event map_event;
    uint8_t* yuv_ptr_y = (uint8_t *)clEnqueueMapBuffer(s->cameras.rear.q, yuv_cl, CL_TRUE,
                                            CL_MAP_READ, 0, s->yuv_buf_size,
                                            0, NULL, &map_event, &err);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    // send frame event
    {
      capnp::MallocMessageBuilder msg;
      cereal::Event::Builder event = msg.initRoot<cereal::Event>();
      event.setLogMonoTime(nanos_since_boot());

      auto framed = event.initFrame();
      framed.setFrameId(frame_data.frame_id);
      framed.setEncodeId(cnt);
      framed.setTimestampEof(frame_data.timestamp_eof);
      framed.setFrameLength(frame_data.frame_length);
      framed.setIntegLines(frame_data.integ_lines);
      framed.setGlobalGain(frame_data.global_gain);
      framed.setLensPos(frame_data.lens_pos);
      framed.setLensSag(frame_data.lens_sag);
      framed.setLensErr(frame_data.lens_err);
      framed.setLensTruePos(frame_data.lens_true_pos);

      framed.setImage(kj::arrayPtr((const uint8_t*)yuv_ptr_y, s->yuv_buf_size));

      kj::ArrayPtr<const float> transform_vs(&s->yuv_transform.v[0], 9);
      framed.setTransform(transform_vs);

      auto words = capnp::messageToFlatArray(msg);
      auto bytes = words.asBytes();
      zmq_send(s->recorder_sock_raw, bytes.begin(), bytes.size(), ZMQ_DONTWAIT);
    }


#ifdef POSENET
    double pt1 = 0, pt2 = 0, pt3 = 0;
    pt1 = millis_since_boot();

    // move second frame to first frame
    memmove(&posenet_input[0], &posenet_input[1], sizeof(float)*(200*532*2 - 1));

    // fill posenet input
    float a;
    // posenet uses a half resolution cropped frame
    // with upper left corner: [50, 237] and
    // bottom right corner: [1114, 637]
    // So the resulting crop is 532 X 200
    for (int y=237; y<637; y+=2) {
      int yy = (y-237)/2;
      for (int x = 50; x < 1114; x+=2) {
        int xx = (x-50)/2;
        a = 0;
        a += yuv_ptr_y[s->yuv_width*(y+0) + (x+1)];
        a += yuv_ptr_y[s->yuv_width*(y+1) + (x+1)];
        a += yuv_ptr_y[s->yuv_width*(y+0) + (x+0)];
        a += yuv_ptr_y[s->yuv_width*(y+1) + (x+0)];
        // The posenet takes a normalized image input
        // like the driving model so [0,255] is remapped
        // to [-1,1]
        posenet_input[(yy*532+xx)*2 + 1] = (a/512.0 - 1.0);
      }
    }
    //FILE *fp;
    //fp = fopen( "testing" , "r" );
    //fread(posenet_input , sizeof(float) , 200*532*2 , fp);
    //fclose(fp);
    //sleep(5);

    pt2 = millis_since_boot();

    posenet_counter++;

    if (posenet_counter % 5 == 0){
      // run posenet
      //printf("avg %f\n", pose_output[0]);
      posenet->execute(posenet_input);

        
      // fix stddevs
      for (int i = 6; i < 12; i++) {
        pose_output[i] = log1p(exp(pose_output[i])) + 1e-6;
      }
      // to radians
      for (int i = 3; i < 6; i++) {
        pose_output[i] = M_PI * pose_output[i] / 180.0;
      }
      // to radians
      for (int i = 9; i < 12; i++) {
        pose_output[i] = M_PI * pose_output[i] / 180.0;
      }

      // send posenet event
      {
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());

        auto posenetd = event.initCameraOdometry();
        kj::ArrayPtr<const float> trans_vs(&pose_output[0], 3);
        posenetd.setTrans(trans_vs);
        kj::ArrayPtr<const float> rot_vs(&pose_output[3], 3);
        posenetd.setRot(rot_vs);
        kj::ArrayPtr<const float> trans_std_vs(&pose_output[6], 3);
        posenetd.setTransStd(trans_std_vs);
        kj::ArrayPtr<const float> rot_std_vs(&pose_output[9], 3);
        posenetd.setRotStd(rot_std_vs);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        zmq_send(s->posenet_sock_raw, bytes.begin(), bytes.size(), ZMQ_DONTWAIT);
      }
      pt3 = millis_since_boot();
      LOGD("pre: %.2fms | posenet: %.2fms", (pt2-pt1), (pt3-pt1));
    }
#endif


    // auto exposure over big box
    const int exposure_x = 290;
    const int exposure_y = 282 + 40;
    const int exposure_height = 314;
    const int exposure_width = 560;
    if (cnt % 3 == 0) {
      // find median box luminance for AE
      uint32_t lum_binning[256] = {0,};
      for (int y=0; y<exposure_height; y++) {
        for (int x=0; x<exposure_width; x++) {
          uint8_t lum = yuv_ptr_y[((exposure_y+y)*s->yuv_width) + exposure_x + x];
          lum_binning[lum]++;
        }
      }
      const unsigned int lum_total = exposure_height * exposure_width;
      unsigned int lum_cur = 0;
      int lum_med = 0;
      for (lum_med=0; lum_med<256; lum_med++) {
        // shouldn't be any values less than 16 - yuv footroom
        lum_cur += lum_binning[lum_med];
        if (lum_cur >= lum_total / 2) {
          break;
        }
      }
      // double avg = (double)acc / (big_box_width * big_box_height) - 16;
      // printf("avg %d\n", lum_med);

      camera_autoexposure(&s->cameras.rear, lum_med / 256.0);
    }

    clEnqueueUnmapMemObject(s->cameras.rear.q, yuv_cl, yuv_ptr_y, 0, NULL, &map_event);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    tbuffer_release(&s->cameras.rear.camera_tb, buf_idx);
    double t5 = millis_since_boot();
    LOGD("queued: %.2fms, model: %.2fms | processing: %.3fms",
            (t2-t1), (mt2-mt1), (t5-t1));
  }

  zsock_destroy(&model_sock);

  return NULL;
}

void* live_thread(void *arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("live");

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);

  zsock_t *liveCalibration_sock = zsock_new_sub(">tcp://127.0.0.1:8019", "");
  assert(liveCalibration_sock);

  zpoller_t *poller = zpoller_new(liveCalibration_sock, terminate, NULL);
  assert(poller);

  while (!do_exit) {
    zsock_t *which = (zsock_t*)zpoller_wait(poller, -1);
    if (which == terminate || which == NULL) {
      break;
    }

    zmq_msg_t msg;
    err = zmq_msg_init(&msg);
    assert(err == 0);

    err = zmq_msg_recv(&msg, zsock_resolve(which), 0);
    assert(err >= 0);
    size_t len = zmq_msg_size(&msg);

    // make copy due to alignment issues, will be freed on out of scope
    auto amsg = kj::heapArray<capnp::word>((len / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), (const uint8_t*)zmq_msg_data(&msg), len);

    // track camera frames to sync to encoder
    capnp::FlatArrayMessageReader cmsg(amsg);
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    if (event.isLiveCalibration()) {
      pthread_mutex_lock(&s->transform_lock);
#ifdef BIGMODEL
      auto wm2 = event.getLiveCalibration().getWarpMatrixBig();
#else
      auto wm2 = event.getLiveCalibration().getWarpMatrix2();
#endif
      assert(wm2.size() == 3*3);
      for (int i=0; i<3*3; i++) {
        s->cur_transform.v[i] = wm2[i];
      }
      s->run_model = true;
      pthread_mutex_unlock(&s->transform_lock);
    }

    zmq_msg_close(&msg);
  }

  zpoller_destroy(&poller);
  zsock_destroy(&terminate);

  zsock_destroy(&liveCalibration_sock);

  return NULL;
}

void set_do_exit(int sig) {
  do_exit = 1;
}

void party(VisionState *s, bool nomodel) {
  int err;

  s->terminate_pub = zsock_new_pub("@inproc://terminate");
  assert(s->terminate_pub);

  pthread_t proc_thread_handle;
  err = pthread_create(&proc_thread_handle, NULL,
                       processing_thread, s);
  assert(err == 0);

  pthread_t live_thread_handle;
  err = pthread_create(&live_thread_handle, NULL,
                       live_thread, s);
  assert(err == 0);

  // priority for cameras
  err = set_realtime_priority(1);
  LOG("setpriority returns %d", err);

  cameras_run(&s->cameras);

  zsock_signal(s->terminate_pub, 0);

  LOG("joining proc_thread");
  err = pthread_join(proc_thread_handle, NULL);
  assert(err == 0);

  LOG("joining live_thread");
  err = pthread_join(live_thread_handle, NULL);
  assert(err == 0);

  zsock_destroy (&s->terminate_pub);
}

}

int main(int argc, char **argv) {
  int err;

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  // boringssl via curl via the calibration api can sometimes
  // try to write to a closed socket. just ignore SIGPIPE
  signal(SIGPIPE, SIG_IGN);

  bool no_model = false;
  if (argc > 1 && strcmp(argv[1], "--no-model") == 0) {
    no_model = true;
  }

  VisionState state = {0};
  VisionState *s = &state;

  clu_init();
  cl_init(s);

  model_init(&s->model, s->device_id, s->context, true);

  cameras_init(&s->cameras);

  s->frame_width = s->cameras.rear.ci.frame_width;
  s->frame_height = s->cameras.rear.ci.frame_height;
 
  // Do not run the model until we receive valid calibration.
  s->run_model = false;
  pthread_mutex_init(&s->transform_lock, NULL);

  init_buffers(s);

  s->recorder_sock = zsock_new_pub("@tcp://*:8002");
  assert(s->recorder_sock);
  s->recorder_sock_raw = zsock_resolve(s->recorder_sock);

  s->posenet_sock = zsock_new_pub("@tcp://*:8066");
  assert(s->posenet_sock);
  s->posenet_sock_raw = zsock_resolve(s->posenet_sock);

  const cl_queue_properties props[] = {0};
  cl_command_queue q = clCreateCommandQueueWithProperties(s->context, s->device_id, props, &err);
  cameras_open(&s->cameras, &s->yuv_cl[0], s->device_id, s->context, q);

  party(s, no_model);

  zsock_destroy(&s->recorder_sock);

  model_free(&s->model);
	free_buffers(s);

  cl_free(s);

  return 0;
}
