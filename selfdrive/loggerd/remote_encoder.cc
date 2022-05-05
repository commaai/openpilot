#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/remote_encoder.h"

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re) {
  const LogCameraInfo &cam_info = (name == "driverEncodeData") ? cameras_logged[1] :
    ((name == "wideRoadEncodeData") ? cameras_logged[2] :
    ((name == "qRoadEncodeData") ? qcam_info : cameras_logged[0]));
  if (!cam_info.record) return 0; // TODO: handle this by not subscribing

  int bytes_count = 0;

  // TODO: AlignedBuffer is making a copy and allocing
  //AlignedBuffer aligned_buf;
  //capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg->getData(), msg->getSize()));
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (name == "driverEncodeData") ? event.getDriverEncodeData() :
    ((name == "wideRoadEncodeData") ? event.getWideRoadEncodeData() :
    ((name == "qRoadEncodeData") ? event.getQRoadEncodeData() : event.getRoadEncodeData()));
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  // rotation happened, process the queue (happens before the current message)
  if (re.logger_segment != s->rotate_segment) {
    re.logger_segment = s->rotate_segment;
    for (auto &qmsg: re.q) {
      bytes_count += handle_encoder_msg(s, qmsg, name, re);
    }
    re.q.clear();
  }

  if (!re.writer) {
    // only create on iframe
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      if (re.dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGD("%s: dropped %d non iframe packets before init", name.c_str(), re.dropped_frames);
        re.dropped_frames = 0;
      }
      re.writer.reset(new VideoWriter(s->segment_path,
        cam_info.filename, !cam_info.is_h265,
        cam_info.frame_width, cam_info.frame_height,
        cam_info.fps, cam_info.is_h265, false));
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
    if (re.writer) {
      // encoder is on the next segment, this segment is over so we close the videowriter
      re.writer.reset();
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d", re.segment, idx.getSegmentNum(), s->ready_to_rotate.load(), s->max_waiting);
    }
    // queue up all the new segment messages, they go in after the rotate
    re.q.push_back(msg);
  } else {
    auto data = edata.getData();
    re.writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof()/1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);

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
    delete msg;
    bytes_count += new_msg.size();
  }

  return bytes_count;
}


