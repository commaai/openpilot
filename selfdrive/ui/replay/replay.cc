#include "selfdrive/ui/replay/replay.h"

#include <QApplication>
#include <QDebug>

#include <capnp/dynamic.h>
#include "cereal/services.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/params.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/replay/util.h"

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_, bool dcam, bool ecam, QString data_dir, QObject *parent)
    : sm(sm_), load_dcam(dcam), load_ecam(ecam), QObject(parent) {
  std::vector<const char *> s;
  auto event_struct = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_struct.getUnionFields().size());
  for (const auto &it : services) {
    if ((allow.size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      uint16_t which = event_struct.getFieldByName(it.name).getProto().getDiscriminantValue();
      sockets_[which] = it.name;
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }
  route_ = std::make_unique<Route>(route, data_dir);
  events_ = new std::vector<Event *>();
  // doSeek & queueSegment are always executed in the same thread
  connect(this, &Replay::seekTo, this, &Replay::doSeek);
  connect(this, &Replay::segmentChanged, this, &Replay::queueSegment);
}

Replay::~Replay() {
  qDebug() << "shutdown: in progress...";
  exit_ = true;
  updating_events_ = true;
  if (stream_thread_) {
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
  }

  delete pm;
  delete events_;
  segments_.clear();
  camera_server_.reset(nullptr);
  qDebug() << "shutdown: done";
}

bool Replay::load() {
  if (!route_->load()) {
    qDebug() << "failed to load route" << route_->name() << "from server";
    return false;
  }

  for (auto &[n, f] : route_->segments()) {
    if ((!f.rlog.isEmpty() || !f.qlog.isEmpty()) && (!f.road_cam.isEmpty() || !f.qcamera.isEmpty())) {
      segments_[n] = nullptr;
    }
  }
  if (segments_.empty()) {
    qDebug() << "no valid segments in route" << route_->name();
    return false;
  }
  qDebug() << "load route" << route_->name() << "with" << segments_.size() << "valid segments";
  return true;
}

void Replay::start(int seconds) {
  seekTo(seconds, false);
}

void Replay::updateEvents(const std::function<bool()> &lambda) {
  // set updating_events to true to force stream thread relase the lock and wait for evnets_udpated.
  updating_events_ = true;
  {
    std::unique_lock lk(stream_lock_);
    events_updated_ = lambda();
    updating_events_ = false;
  }
  stream_cv_.notify_one();
}

void Replay::doSeek(int seconds, bool relative) {
  updateEvents([&]() {
    if (relative) {
      seconds += currentSeconds();
    }
    qInfo() << "seeking to" << seconds;
    const int max_segment_number = segments_.rbegin()->first;
    cur_mono_time_ = route_start_ts_ + std::clamp(seconds, 0, (max_segment_number + 1) * 60) * 1e9;
    current_segment_ = std::min(seconds / 60, max_segment_number);
    return false;
  });
  queueSegment();
}

void Replay::pause(bool pause) {
  updateEvents([=]() {
    qDebug() << (pause ? "paused..." : "resuming");
    if (pause) {
      qInfo() << "at " << currentSeconds() << "s";
    }
    paused_ = pause;
    return true;
  });
}

void Replay::setCurrentSegment(int n) {
  if (current_segment_.exchange(n) != n) {
    emit segmentChanged();
  }
}

void Replay::segmentLoadFinished(bool success) {
  if (!success) {
    Segment *seg = qobject_cast<Segment *>(sender());
    qInfo() << "failed to load segment " << seg->seg_num << ", removing it from current replay list";
    segments_.erase(seg->seg_num);
  }
  queueSegment();
}

void Replay::queueSegment() {
  // get the current segment window
  SegmentMap::iterator begin, cur, end;
  begin = cur = end = segments_.lower_bound(current_segment_);
  for (int i = 0; i < BACKWARD_SEGS && begin != segments_.begin(); ++i) {
    --begin;
  }
  for (int i = 0; i <= FORWARD_SEGS && end != segments_.end(); ++i) {
    ++end;
  }

  // load segments
  for (auto it = begin; it != end; ++it) {
    auto &[n, seg] = *it;
    if (!seg) {
      seg = std::make_unique<Segment>(n, route_->at(n), load_dcam, load_ecam);
      QObject::connect(seg.get(), &Segment::loadFinished, this, &Replay::segmentLoadFinished);
      qInfo() << "loading segment" << n << "...";
    }
  }

  // merge segments
  mergeSegments(begin, end);

  // free segments out of current semgnt window.
  for (auto it = segments_.begin(); it != begin; ++it) {
    it->second.reset(nullptr);
  }
  for (auto it = end; it != segments_.end(); ++it) {
    it->second.reset(nullptr);
  }

  // start stream thread
  if (stream_thread_ == nullptr && cur != segments_.end() && cur->second->isLoaded()) {
    startStream(cur->second.get());
  }
}

void Replay::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  // segments must be merged in sequence.
  std::vector<int> segments_need_merge;
  for (auto it = begin; it != end && it->second->isLoaded(); ++it) {
    segments_need_merge.push_back(it->first);
  }

  if (segments_need_merge != segments_merged_) {
    qDebug() << "merge segments" << segments_need_merge;
    // merge & sort events
    std::vector<Event *> *new_events = new std::vector<Event *>();
    new_events->reserve(std::accumulate(segments_need_merge.begin(), segments_need_merge.end(), 0,
                                        [=](int v, int n) { return v + segments_[n]->log->events.size(); }));
    for (int n : segments_need_merge) {
      auto &e = segments_[n]->log->events;
      auto middle = new_events->insert(new_events->end(), e.begin(), e.end());
      std::inplace_merge(new_events->begin(), middle, new_events->end(), Event::lessThan());
    }
    // update events
    auto prev_events = events_;
    updateEvents([&]() {
      events_ = new_events;
      segments_merged_ = segments_need_merge;
      return true;
    });
    delete prev_events;
  } else {
    updateEvents([=]() { return true; });
  }
}

void Replay::startStream(const Segment *cur_segment) {
  const auto &events = cur_segment->log->events;

  // get route start time from initData
  auto it = std::find_if(events.begin(), events.end(), [](auto e) { return e->which == cereal::Event::Which::INIT_DATA; });
  route_start_ts_ = it != events.end() ? (*it)->mono_time : events[0]->mono_time;
  cur_mono_time_ += route_start_ts_;

  // write CarParams
  it = std::find_if(events.begin(), events.end(), [](auto e) { return e->which == cereal::Event::Which::CAR_PARAMS; });
  if (it != events.end()) {
    auto bytes = (*it)->bytes();
    Params().put("CarParams", (const char *)bytes.begin(), bytes.size());
  } else {
    qInfo() << "failed to read CarParams from current segment";
  }

  // start camera server
  std::pair<int, int> cameras[MAX_CAMERAS] = {};
  for (auto type : ALL_CAMERAS) {
    if (auto &fr = cur_segment->frames[type]) {
      cameras[type] = {fr->width, fr->height};
    }
  }
  camera_server_ = std::make_unique<CameraServer>();
  camera_server_->start(cameras);

  // start stream thread
  stream_thread_ = QThread::create(&Replay::stream, this);
  stream_thread_->start();
}

void Replay::publishMessage(const Event *e) {
  if (sm == nullptr) {
    auto bytes = e->bytes();
    int ret = pm->send(sockets_[e->which], (capnp::byte *)bytes.begin(), bytes.size());
    if (ret == -1) {
      qDebug() << "stop publishing" << sockets_[e->which] << "due to multiple publishers error";
      sockets_[e->which] = nullptr;
    }
  } else {
    sm->update_msgs(nanos_since_boot(), {{sockets_[e->which], e->event}});
  }
}

void Replay::publishFrame(const Event *e) {
  static const std::map<cereal::Event::Which, CameraType> cam_types{
      {cereal::Event::ROAD_ENCODE_IDX, RoadCam},
      {cereal::Event::DRIVER_ENCODE_IDX, DriverCam},
      {cereal::Event::WIDE_ROAD_ENCODE_IDX, WideRoadCam},
  };
  auto eidx = capnp::AnyStruct::Reader(e->event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
  if (std::find(segments_merged_.begin(), segments_merged_.end(), eidx.getSegmentNum()) == segments_merged_.end()) {
    // eidx's segment is not loaded
    return;
  }
  CameraType cam = cam_types.at(e->which);
  auto &fr = segments_[eidx.getSegmentNum()]->frames[cam];
  if (fr && eidx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
    camera_server_->pushFrame(cam, fr.get(), eidx);
  }
}

void Replay::stream() {
  float last_print = 0;
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;

  std::unique_lock lk(stream_lock_);

  while (true) {
    stream_cv_.wait(lk, [=]() { return exit_ || (events_updated_ && !paused_); });
    events_updated_ = false;
    if (exit_) break;

    Event cur_event(cur_which, cur_mono_time_);
    auto eit = std::upper_bound(events_->begin(), events_->end(), &cur_event, Event::lessThan());
    if (eit == events_->end()) {
      qDebug() << "waiting for events...";
      continue;
    }

    uint64_t evt_start_ts = cur_mono_time_;
    uint64_t loop_start_ts = nanos_since_boot();

    for (auto end = events_->end(); !updating_events_ && eit != end; ++eit) {
      const Event *evt = (*eit);
      cur_which = evt->which;
      cur_mono_time_ = evt->mono_time;
      const int current_ts = currentSeconds();
      if (last_print > current_ts || (current_ts - last_print) > 5.0) {
        last_print = current_ts;
        qInfo() << "at " << current_ts << "s";
      }
      setCurrentSegment(current_ts / 60);

      // migration for pandaState -> pandaStates to keep UI working for old segments
      if (cur_which == cereal::Event::Which::PANDA_STATE_D_E_P_R_E_C_A_T_E_D) {
        MessageBuilder msg;
        auto ps = msg.initEvent().initPandaStates(1);
        ps[0].setIgnitionLine(true);
        ps[0].setPandaType(cereal::PandaState::PandaType::DOS);
        pm->send(sockets_[cereal::Event::Which::PANDA_STATES], msg);
      }

      if (cur_which < sockets_.size() && sockets_[cur_which] != nullptr) {
        // keep time
        long etime = cur_mono_time_ - evt_start_ts;
        long rtime = nanos_since_boot() - loop_start_ts;
        long behind_ns = etime - rtime;
        // if behind_ns is greater than 1 second, it means that an invalid segemnt is skipped by seeking/replaying
        if (behind_ns >= 1 * 1e9) {
          // reset start times
          evt_start_ts = cur_mono_time_;
          loop_start_ts = nanos_since_boot();
        } else if (behind_ns > 0) {
          precise_nano_sleep(behind_ns);
        }

        if (evt->frame) {
          publishFrame(evt);
        } else {
          publishMessage(evt);
        }
      }
    }
    // wait for frame to be sent before unlock.(frameReader may be deleted after unlock)
    camera_server_->waitFinish();

    if (eit == events_->end() && (current_segment_ == segments_.rbegin()->first)) {
      qInfo() << "reaches the end of route, restart from beginning";
      emit seekTo(0, false);
    }
  }
}
