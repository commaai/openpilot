#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/remote_encoder.h"

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re) {
  const LogCameraInfo &cam_info = (name == "driverEncodeData") ? cameras_logged[1] :
    ((name == "wideRoadEncodeData") ? cameras_logged[2] :
    ((name == "qRoadEncodeData") ? qcam_info : cameras_logged[0]));

  // rotation happened, process the queue (happens before the current message)
  int bytes_count = 0;
  if (re.logger_segment != s->rotate_segment) {
    re.logger_segment = s->rotate_segment;
    for (auto &qmsg: re.q) {
      bytes_count += handle_encoder_msg(s, qmsg, name, re);
    }
    re.q.clear();
  }

  // extract the message
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (name == "driverEncodeData") ? event.getDriverEncodeData() :
    ((name == "wideRoadEncodeData") ? event.getWideRoadEncodeData() :
    ((name == "qRoadEncodeData") ? event.getQRoadEncodeData() : event.getRoadEncodeData()));
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  // if we aren't recording, don't create the writer
  if (!re.writer && cam_info.record) {
    // only create on iframe
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      if (re.dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGD("%s: dropped %d non iframe packets before init", name.c_str(), re.dropped_frames);
        re.dropped_frames = 0;
      }
      re.writer.reset(new VideoWriter(s->segment_path,
        cam_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
        cam_info.frame_width, cam_info.frame_height, cam_info.fps, idx.getType()));
      // write the header
      auto header = edata.getHeader();
      re.writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof()/1000, true, false);
      re.segment = idx.getSegmentNum();
    } else {
      ++re.dropped_frames;
      return bytes_count;
    }
  }

  if (re.segment != idx.getSegmentNum()) {
    if (re.q.size() == 0) {
      // encoder is on the next segment, this segment is over so we close the videowriter
      re.writer.reset();
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s", re.segment, idx.getSegmentNum(), s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }
    // queue up all the new segment messages, they go in after the rotate
    re.q.push_back(msg);
  } else {
    if (re.writer) {
      auto data = edata.getData();
      re.writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof()/1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);
    }

    // put it in log stream as the idx packet
    MessageBuilder bmsg;
    auto evt = bmsg.initEvent(event.getValid());
    evt.setLogMonoTime(event.getLogMonoTime());
    if (name == "driverEncodeData") { evt.setDriverEncodeIdx(idx); }
    if (name == "wideRoadEncodeData") { evt.setWideRoadEncodeIdx(idx); }
    if (name == "qRoadEncodeData") { evt.setQRoadEncodeIdx(idx); }
    if (name == "roadEncodeData") { evt.setRoadEncodeIdx(idx); }
    auto new_msg = bmsg.toBytes();
    logger_log(&s->logger, (uint8_t *)new_msg.begin(), new_msg.size(), true);   // always in qlog?
    bytes_count += new_msg.size();

    // this frees the message
    delete msg;
  }

  return bytes_count;
}


