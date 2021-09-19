#include "selfdrive/ui/replay/replay.h"

#include <QApplication>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_, QObject *parent) : sm(sm_), QObject(parent) {
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
  events = new std::vector<Event *>();
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
  segments.resize(route_->size());
  seekTo(seconds);

  // start stream thread
  thread = new QThread;
  QObject::connect(thread, &QThread::started, [=]() { stream(); });
  thread->start();
}

void Replay::seekTo(int seconds) {
  if (segments.empty()) return;

  updating_events = true;

  std::unique_lock lk(lock);
  seconds = std::clamp(seconds, 0, (int)segments.size() * 60);
  qInfo() << "seeking to " << seconds;
  seek_ts = seconds;
  setCurrentSegment(std::clamp(seconds / 60, 0, (int)segments.size() - 1));
  updating_events = false;
}

void Replay::relativeSeek(int seconds) {
  seekTo(current_ts + seconds);
}

void Replay::pause(bool pause) {
  updating_events = true;
  std::unique_lock lk(lock);
  qDebug() << (pause ? "paused..." : "resuming");
  paused_ = pause;
  updating_events = false;
  stream_cv_.notify_one();
}

void Replay::setCurrentSegment(int n) {
  if (current_segment.exchange(n) != n) {
    emit segmentChanged(n);
  }
}

// maintain the segment window
void Replay::queueSegment() {
  assert(QThread::currentThreadId() == qApp->thread()->currentThreadId());

  // fetch segments forward
  int cur_seg = current_segment.load();
  int end_idx = cur_seg;
  for (int i = cur_seg, fwd = 0; i < segments.size() && fwd <= FORWARD_SEGS; ++i) {
    if (!segments[i]) {
      segments[i] = std::make_unique<Segment>(i, route_->at(i));
      QObject::connect(segments[i].get(), &Segment::loadFinished, this, &Replay::queueSegment);
    }
    end_idx = i;
    // skip invalid segment
    fwd += segments[i]->isValid();
  }

  // merge segments
  mergeSegments(cur_seg, end_idx);
}

void Replay::mergeSegments(int cur_seg, int end_idx) {
  // segments must be merged in sequence.
  std::vector<int> segments_need_merge;
  const int begin_idx = std::max(cur_seg - BACKWARD_SEGS, 0);
  for (int i = begin_idx; i <= end_idx; ++i) {
    if (segments[i] && segments[i]->isLoaded()) {
      segments_need_merge.push_back(i);
    } else if (i >= cur_seg) {
      // segment is valid,but still loading. can't skip it to merge the next one.
      // otherwise the stream thread may jump to the next segment.
      break;
    }
  }

  if (segments_need_merge != segments_merged) {
    qDebug() << "merge segments" << segments_need_merge;
    segments_merged = segments_need_merge;

    std::vector<Event *> *new_events = new std::vector<Event *>();
    std::unordered_map<uint32_t, EncodeIdx> *new_eidx = new std::unordered_map<uint32_t, EncodeIdx>[MAX_CAMERAS];
    for (int n : segments_need_merge) {
      auto &log = segments[n]->log;
      // merge & sort events
      auto middle = new_events->insert(new_events->end(), log->events.begin(), log->events.end());
      std::inplace_merge(new_events->begin(), middle, new_events->end(), Event::lessThan());
      for (CameraType cam_type : ALL_CAMERAS) {
        new_eidx[cam_type].insert(log->eidx[cam_type].begin(), log->eidx[cam_type].end());
      }
    }

    // update logs
    // set updating_events to true to force stream thread relase the lock
    updating_events = true;
    lock.lock();

    if (route_start_ts == 0) {
      // get route start time from initData
      auto it = std::find_if(new_events->begin(), new_events->end(), [=](auto e) { return e->which == cereal::Event::Which::INIT_DATA; });
      if (it != new_events->end()) {
        route_start_ts = (*it)->mono_time;
      }
    }

    auto prev_events = std::exchange(events, new_events);
    auto prev_eidx = std::exchange(eidx, new_eidx);
    updating_events = false;

    lock.unlock();

    // free segments
    delete prev_events;
    delete[] prev_eidx;
    for (int i = 0; i < segments.size(); i++) {
      if ((i < begin_idx || i > end_idx) && segments[i]) {
        segments[i].reset(nullptr);
      }
    }
  }
}

void Replay::stream() {
  bool waiting_printed = false;
  uint64_t cur_mono_time = 0;
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;

  while (true) {
    std::unique_lock lk(lock);
    stream_cv_.wait(lk, [=]() { return paused_ == false; });

    uint64_t evt_start_ts = seek_ts != -1 ? route_start_ts + (seek_ts * 1e9) : cur_mono_time;
    Event cur_event(cur_which, evt_start_ts);
    auto eit = std::upper_bound(events->begin(), events->end(), &cur_event, Event::lessThan());
    if (eit == events->end()) {
      lock.unlock();
      if (std::exchange(waiting_printed, true) == false) {
        qDebug() << "waiting for events...";
      }
      QThread::msleep(50);
      continue;
    }
    waiting_printed = false;
    seek_ts = -1;
    uint64_t loop_start_ts = nanos_since_boot();
    qDebug() << "unlogging at" << int((evt_start_ts - route_start_ts) / 1e9);

    for (/**/; !updating_events && eit != events->end(); ++eit) {
      const Event *evt = (*eit);
      cur_which = evt->which;
      cur_mono_time = evt->mono_time;
      current_ts = (cur_mono_time - route_start_ts) / 1e9;

      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(evt->event).which()) {
        type = e_->getProto().getName();
      }

      if (socks.find(type) != socks.end()) {
        if (std::abs(current_ts - last_print) > 5.0) {
          last_print = current_ts;
          qInfo() << "at " << int(last_print) << "s";
        }

        setCurrentSegment(current_ts / 60);
        // keep time
        long etime = cur_mono_time - evt_start_ts;
        long rtime = nanos_since_boot() - loop_start_ts;
        long us_behind = ((etime - rtime) * 1e-3) + 0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
        }

        // publish frame
        // TODO: publish all frames
        if (evt->which == cereal::Event::ROAD_CAMERA_STATE) {
          auto it_ = eidx[RoadCam].find(evt->event.getRoadCameraState().getFrameId());
          if (it_ != eidx[RoadCam].end()) {
            EncodeIdx &e = it_->second;
            auto &seg = segments[e.segmentNum]; 
            if (seg && seg->isLoaded()) {
              auto &frm = seg->frames[RoadCam];
              if (vipc_server == nullptr) {
                cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
                cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

                vipc_server = new VisionIpcServer("camerad", device_id, context);
                vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_BACK, UI_BUF_COUNT,
                                            true, frm->width, frm->height);
                vipc_server->start_listener();
              }

              uint8_t *dat = frm->get(e.frameEncodeId);
              if (dat) {
                VisionIpcBufExtra extra = {};
                VisionBuf *buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
                memcpy(buf->addr, dat, frm->getRGBSize());
                vipc_server->send(buf, &extra, false);
              }
            }
          }
        }

        // publish msg
        if (sm == nullptr) {
          auto bytes = evt->bytes();
          pm->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          sm->update_msgs(nanos_since_boot(), {{type, evt->event}});
        }
      }
    }
    lk.unlock();
    usleep(0);
  }
}
