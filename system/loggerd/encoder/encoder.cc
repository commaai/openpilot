#include <cassert>
#include "system/loggerd/encoder/encoder.h"

VideoEncoder::~VideoEncoder() {}

void VideoEncoder::publisher_init() {
  // publish
  service_name = this->publish_name;
  pm.reset(new PubMaster({service_name}));
}

void VideoEncoder::publisher_publish(VideoEncoder *e, int segment_num, uint32_t idx, VisionIpcBufExtra &extra,
                                     unsigned int flags, kj::ArrayPtr<capnp::byte> header, kj::ArrayPtr<capnp::byte> dat) {
  // broadcast packet
  MessageBuilder msg;
  auto event = msg.initEvent(true);
  auto edat = (e->type == DriverCam) ? event.initDriverEncodeData() :
    ((e->type == WideRoadCam) ? event.initWideRoadEncodeData() :
    (e->in_width == e->out_width ? event.initRoadEncodeData() : event.initQRoadEncodeData()));
  auto edata = edat.initIdx();
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  edat.setUnixTimestampNanos((uint64_t)ts.tv_sec*1000000000 + ts.tv_nsec);
  edata.setFrameId(extra.frame_id);
  edata.setTimestampSof(extra.timestamp_sof);
  edata.setTimestampEof(extra.timestamp_eof);
  edata.setType(e->codec);
  edata.setEncodeId(e->cnt++);
  edata.setSegmentNum(segment_num);
  edata.setSegmentId(idx);
  edata.setFlags(flags);
  edata.setLen(dat.size());
  edat.setData(dat);
  if (flags & V4L2_BUF_FLAG_KEYFRAME) edat.setHeader(header);

  auto words = new kj::Array<capnp::word>(capnp::messageToFlatArray(msg));
  auto bytes = words->asBytes();
  e->pm->send(e->service_name, bytes.begin(), bytes.size());
  delete words;
}
