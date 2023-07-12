#include "system/loggerd/encoder_writer.h"

size_t EncoderWriter::write(LoggerState *logger, Message *msg) {
  std::unique_ptr<Message> m(msg);
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)m->getData(), m->getSize() / sizeof(capnp::word)));
  auto event = cmsg.getRoot<cereal::Event>();
  auto idx = (event.*(info.get_encode_data_func))().getIdx();

  remote_encoder_segment = idx.getSegmentNum();
  if (current_encoder_segment == -1) {
    current_encoder_segment = remote_encoder_segment;
    LOGD("%s: has encoderd offset %d", info.publish_name, current_encoder_segment);
  }

  size_t written = 0;
  if (current_encoder_segment == remote_encoder_segment) {
    if (!q.empty()) {
      for (auto &qmsg : q) {
        capnp::FlatArrayMessageReader msg_reader({(capnp::word *)qmsg->getData(), qmsg->getSize() / sizeof(capnp::word)});
        written += write_encoder_data(logger, msg_reader.getRoot<cereal::Event>());
      }
      q.clear();
    }
    written = write_encoder_data(logger, event);
  } else {
    // rotate to the next segment to sync with remote encoders.
    if (!marked_ready_to_rotate) {
      marked_ready_to_rotate = true;
      ++ready_to_rotate;
    }
    q.push_back(std::move(m));
  }
  return written;
}

void EncoderWriter::rotate(const std::string &path) {
  video_writer.reset();
  current_encoder_segment = remote_encoder_segment;
  segment_path = path;
  marked_ready_to_rotate = false;
}

size_t EncoderWriter::write_encoder_data(LoggerState *logger, const cereal::Event::Reader event) {
  auto edata = (event.*(info.get_encode_data_func))();
  const auto idx = edata.getIdx();

  if (info.record) {
    write_video(edata, idx);
  }
  // put it in log stream as the idx packet
  MessageBuilder msg;
  auto evt = msg.initEvent(event.getValid());
  evt.setLogMonoTime(event.getLogMonoTime());
  (evt.*(info.set_encode_idx_func))(idx);
  auto bytes = msg.toBytes();
  logger_log(logger, (uint8_t *)bytes.begin(), bytes.size(), true);  // always in qlog?
  return bytes.size();
}

size_t EncoderWriter::write_video(const cereal::EncodeData::Reader &edata, const cereal::EncodeIndex::Reader &idx) {
  size_t written = 0;
  const bool is_key_frame = idx.getFlags() & V4L2_BUF_FLAG_KEYFRAME;

  if (!video_writer) {
    if (is_key_frame) { // only create on iframe
      if (dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGW("%s: dropped %d non iframe packets before init", info.publish_name, dropped_frames);
        dropped_frames = 0;
      }
      video_writer.reset(new VideoWriter(segment_path.c_str(), info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
                                         info.frame_width, info.frame_height, info.fps, idx.getType()));
      auto header = edata.getHeader();
      video_writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof() / 1000, true, false);
      written += header.size();
    } else {
      // this is a sad case when we aren't recording, but don't have an iframe
      // nothing we can do but drop the frame
      ++dropped_frames;
    }
  }

  if (video_writer) {
    auto data = edata.getData();
    video_writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof() / 1000, false, is_key_frame);
    written += data.size();
  }
  return written;
}
