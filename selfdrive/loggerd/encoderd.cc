#include "selfdrive/loggerd/loggerd.h"

ExitHandler do_exit;

void encoder_thread(const LogCameraInfo &cam_info) {
  util::set_thread_name(cam_info.filename);

  //int cur_seg = -1;
  int encode_idx = 0;
  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

  // While we write them right to the log for sync, we also publish the encode idx to the socket
  /*const char *service_name = cam_info.type == DriverCam ? "driverEncodeIdx" : (cam_info.type == WideRoadCam ? "wideRoadEncodeIdx" : "roadEncodeIdx");
  PubMaster pm({service_name});*/

  while (!do_exit) {
    if (!vipc_client.connect(false)) {
      util::sleep_for(5);
      continue;
    }

    // init encoders
    if (encoders.empty()) {
      VisionBuf buf_info = vipc_client.buffers[0];
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);

      // main encoder
      encoders.push_back(new Encoder(cam_info.filename, cam_info.type, buf_info.width, buf_info.height,
                                     cam_info.fps, cam_info.bitrate, cam_info.is_h265,
                                     buf_info.width, buf_info.height, false));
      // qcamera encoder
      if (cam_info.has_qcamera) {
        encoders.push_back(new Encoder(qcam_info.filename, cam_info.type, buf_info.width, buf_info.height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265,
                                       qcam_info.frame_width, qcam_info.frame_height, false));
      }
    }

    for (int i = 0; i < encoders.size(); ++i) {
      encoders[i]->encoder_open(NULL);
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height, &extra);

        if (out_id == -1) {
          LOGE("Failed to encode frame. frame_id: %d encode_id: %d", extra.frame_id, encode_idx);
        }

        // publish encode index
        /*if (i == 0 && out_id != -1) {
          MessageBuilder msg;
          // this is really ugly
          bool valid = (buf->get_frame_id() == extra.frame_id);
          auto eidx = cam_info.type == DriverCam ? msg.initEvent(valid).initDriverEncodeIdx() :
                     (cam_info.type == WideRoadCam ? msg.initEvent(valid).initWideRoadEncodeIdx() : msg.initEvent(valid).initRoadEncodeIdx());
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
          if (Hardware::TICI()) {
            eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
          } else {
            eidx.setType(cam_info.type == DriverCam ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
          }
          eidx.setEncodeId(encode_idx);
          eidx.setSegmentNum(cur_seg);
          eidx.setSegmentId(out_id);
          pm.send(service_name, msg);
        }*/
      }

      encode_idx++;
    }
  }

  LOG("encoder destroy");
  for(auto &e : encoders) {
    e->encoder_close();
    delete e;
  }
}

void encoderd_thread() {
  std::vector<std::thread> encoder_threads;
  for (const auto &cam : cameras_logged) {
    if (cam.enable) {
      encoder_threads.push_back(std::thread(encoder_thread, cam));
    }
  }
  for (auto &t : encoder_threads) t.join();
}

int main() {
  if (Hardware::TICI()) {
    int ret;
    ret = util::set_core_affinity({7});
    assert(ret == 0);
  }
  encoderd_thread();
  return 0;
}
