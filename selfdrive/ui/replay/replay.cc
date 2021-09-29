#include "selfdrive/ui/replay/replay.h"

#include <QApplication>
#include <QDebug>
#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

inline void precise_nano_sleep(long sleep_ns) {
  const long estimate_ns = 1 * 1e6;  // 1ms
  struct timespec req = {.tv_nsec = estimate_ns};
  uint64_t start_sleep = nanos_since_boot();
  while (sleep_ns > estimate_ns) {
    nanosleep(&req, nullptr);
    uint64_t end_sleep = nanos_since_boot();
    sleep_ns -= (end_sleep - start_sleep);
    start_sleep = end_sleep;
  }
  // spin wait
  if (sleep_ns > 0) {
    while ((nanos_since_boot() - start_sleep) <= sleep_ns) {/**/}
  }
}

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_, bool dcam, bool ecam, QObject *parent)
    : sm(sm_), load_dcam(dcam), load_ecam(ecam), QObject(parent) {
  std::vector<const char*> s;
  for (const auto &it : services) {
    if ((allow.size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      socks.insert(it.name);
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }

  route_ = std::make_unique<Route>(route);
  events_ = new std::vector<Event *>();
  // queueSegment is always executed in the main thread
  connect(this, &Replay::segmentChanged, this, &Replay::queueSegment);
}

Replay::~Replay() {
  // TODO: quit stream thread and free resources.
}

void Replay::start(int seconds){
  // load route
  if (!route_->load() || route_->size() == 0) {
    qDebug() << "failed load route" << route_->name() << "from server";
    return;
  }

  qDebug() << "load route" << route_->name() << route_->size() << "segments, start from" << seconds;
  segments_.resize(route_->size());
  seekTo(seconds);

  // start stream thread
  thread = new QThread;
  QObject::connect(thread, &QThread::started, [=]() { stream(); });
  thread->start();
}

void Replay::updateEvents(const std::function<bool()>& lambda) {
  // set updating_events to true to force stream thread relase the lock and wait for evnets_udpated.
  updating_events_ = true;
  {
    std::unique_lock lk(lock_);
    events_updated_ = lambda();
    updating_events_ = false;
  }
  stream_cv_.notify_one();
}

void Replay::seekTo(int seconds, bool relative) {
  if (segments_.empty()) return;

  updateEvents([&]() {
    if (relative) {
      seconds += ((cur_mono_time_ - route_start_ts_) * 1e-9);
    }
    seconds = std::clamp(seconds, 0, (int)segments_.size() * 60 - 1);
    qInfo() << "seeking to " << seconds;

    int segment = seconds / 60;
    bool segment_changed = (segment != current_segment_);

    cur_mono_time_ = route_start_ts_ + seconds * 1e9;
    setCurrentSegment(segment);
    bool segment_loaded = std::find(segments_merged_.begin(), segments_merged_.end(), segment) != segments_merged_.end();
    // return false if segment changed and not loaded yet
    return !segment_changed || segment_loaded;
  });
}

void Replay::pause(bool pause) {
  updateEvents([=]() {
    qDebug() << (pause ? "paused..." : "resuming");
    paused_ = pause;
    return true;
  });
}

void Replay::setCurrentSegment(int n) {
  if (current_segment_.exchange(n) != n) {
    emit segmentChanged(n);
  }
}

// maintain the segment window
void Replay::queueSegment() {
  // fetch segments forward
  int cur_seg = current_segment_.load();
  int end_idx = cur_seg;
  for (int i = cur_seg, fwd = 0; i < segments_.size() && fwd <= FORWARD_SEGS; ++i) {
    if (!segments_[i]) {
      segments_[i] = std::make_unique<Segment>(i, route_->at(i), load_dcam, load_ecam);
      QObject::connect(segments_[i].get(), &Segment::loadFinished, this, &Replay::queueSegment);
    }
    end_idx = i;
    // skip invalid segment
    if (segments_[i]->isValid()) {
      ++fwd;
    } else if (i == cur_seg) {
      ++cur_seg;
    }
  }

  // merge segments
  mergeSegments(std::min(cur_seg, (int)segments_.size() - 1), end_idx);
}

void Replay::mergeSegments(int cur_seg, int end_idx) {
  // segments must be merged in sequence.
  std::vector<int> segments_need_merge;
  const int begin_idx = std::max(cur_seg - BACKWARD_SEGS, 0);
  for (int i = begin_idx; i <= end_idx; ++i) {
    if (segments_[i] && segments_[i]->isLoaded()) {
      segments_need_merge.push_back(i);
    } else if (i >= cur_seg) {
      // segment is valid,but still loading. can't skip it to merge the next one.
      // otherwise the stream thread may jump to the next segment.
      break;
    }
  }

  if (segments_need_merge != segments_merged_) {
    qDebug() << "merge segments" << segments_need_merge;

    // merge & sort events
    std::vector<Event *> *new_events = new std::vector<Event *>();
    for (int n : segments_need_merge) {
      auto &log = segments_[n]->log;
      auto middle = new_events->insert(new_events->end(), log->events.begin(), log->events.end());
      std::inplace_merge(new_events->begin(), middle, new_events->end(), Event::lessThan());
    }

    // update events
    auto prev_events = events_;
    updateEvents([&]() {
      if (route_start_ts_ == 0) {
        // get route start time from initData
        auto it = std::find_if(new_events->begin(), new_events->end(), [=](auto e) { return e->which == cereal::Event::Which::INIT_DATA; });
        if (it != new_events->end()) {
          route_start_ts_ = (*it)->mono_time;
          cur_mono_time_ = route_start_ts_;
        }
      }

      events_ = new_events;
      segments_merged_ = segments_need_merge;
      return true;
    });
    delete prev_events;
  }

  // free segments out of current semgnt window.
  for (int i = 0; i < segments_.size(); i++) {
    if ((i < begin_idx || i > end_idx) && segments_[i]) {
      segments_[i].reset(nullptr);
    }
  }
}

void Replay::stream() {
  float last_print = 0;
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;

  std::unique_lock lk(lock_);

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

    qDebug() << "unlogging at" << (int)((cur_mono_time_ - route_start_ts_) * 1e-9);
    uint64_t evt_start_ts = cur_mono_time_;
    uint64_t loop_start_ts = nanos_since_boot();

    for (auto end = events_->end(); !updating_events_ && eit != end; ++eit) {
      const Event *evt = (*eit);
      cur_which = evt->which;
      cur_mono_time_ = evt->mono_time;

      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(evt->event).which()) {
        type = e_->getProto().getName();
      }

      if (socks.find(type) != socks.end()) {
        int current_ts = (cur_mono_time_ - route_start_ts_) / 1e9;
        if ((current_ts - last_print) > 5.0) {
          last_print = current_ts;
          qInfo() << "at " << current_ts << "s";
        }
        setCurrentSegment(current_ts / 60);

        // keep time
        long etime = cur_mono_time_ - evt_start_ts;
        long rtime = nanos_since_boot() - loop_start_ts;
        long behind_ns = etime - rtime;
        if (behind_ns > 0) {
          precise_nano_sleep(behind_ns);
        }

        // publish frame
        if (evt->frame) {
          // TODO: publish all frames
          if (evt->which == cereal::Event::ROAD_ENCODE_IDX) {
            auto idx = evt->event.getRoadEncodeIdx();
            auto &seg = segments_[idx.getSegmentNum()];

            if (seg && seg->isLoaded() && idx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
              auto &frm = seg->frames[RoadCam];

              if (vipc_server == nullptr) {
                cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
                cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

                vipc_server = new VisionIpcServer("camerad", device_id, context);
                vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_BACK, UI_BUF_COUNT,
                                            true, frm->width, frm->height);
                vipc_server->start_listener();
              }

              uint8_t *dat = frm->get(idx.getSegmentId());
              if (dat) {
                VisionIpcBufExtra extra = {};
                VisionBuf *buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
                memcpy(buf->addr, dat, frm->getRGBSize());
                vipc_server->send(buf, &extra, false);
              }
            }
          }

        // publish msg
        } else {
          if (sm == nullptr) {
            auto bytes = evt->bytes();
            pm->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
          } else {
            sm->update_msgs(nanos_since_boot(), {{type, evt->event}});
          }
        }
      }
    }
  }
}
