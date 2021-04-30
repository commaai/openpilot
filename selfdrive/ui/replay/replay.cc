#include "replay.hpp"

int getch(void) {
  int ch;
  struct termios oldt;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);

  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return ch;
}

Replay::Replay(QString route_, int seek_) : route(route_), seek(seek_) {
  ctx = Context::create();
  seek_request = seek*1e9;

  QStringList block = QString(getenv("BLOCK")).split(",");
  qDebug() << "blocklist" << block;

  QStringList allow = QString(getenv("ALLOW")).split(",");
  qDebug() << "allowlist" << allow;

  for (const auto& it : services) {
    std::string name = it.name;
    if (allow[0].size() > 0 && !allow.contains(name.c_str())) {
      qDebug() << "not allowing" << name.c_str();
      continue;
    }

    if (block.contains(name.c_str())) {
      qDebug() << "blocking" << name.c_str();
      continue;
    }

    PubSocket *sock = PubSocket::create(ctx, name);
    if (sock == NULL) {
      qDebug() << "FAILED" << name.c_str();
      continue;
    }

    qDebug() << name.c_str();

    socks.insert(name, sock);
  }

  current_segment = -window_padding - 1;
  bool create_jwt = true;

#if !defined(QCOM) && !defined(QCOM2)
  create_jwt = false;
#endif

  http = new HttpRequest(this, "https://api.commadotai.com/v1/route/" + route + "/files", "", create_jwt);
  QObject::connect(http, SIGNAL(receivedResponse(QString)), this, SLOT(parseResponse(QString)));
}

void Replay::parseResponse(QString response){
  response = response.trimmed();
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());

  if (doc.isNull()) {
    qDebug() << "JSON Parse failed";
    return;
  }

  camera_paths = doc["cameras"].toArray();
  log_paths = doc["logs"].toArray();

  seek_queue.enqueue(seek);
}

void Replay::addSegment(int i){
  if (lrs.find(i) != lrs.end()) {
    return;
  }

  QThread* lr_thread = new QThread;

  QString log_fn = this->log_paths.at(i).toString();
  lrs.insert(i, new LogReader(log_fn, &events, &events_lock, &eidx));

  lrs[i]->moveToThread(lr_thread);
  QObject::connect(lr_thread, SIGNAL (started()), lrs[i], SLOT (process()));
  lr_thread->start();

  QString camera_fn = this->camera_paths.at(i).toString();
  frs.insert(i, new FrameReader(qPrintable(camera_fn)));
}

void Replay::trimSegment(int seg_num){
  lrs.remove(seg_num);
  frs.remove(seg_num);

  auto eit = events.begin();
  while(eit != events.end()){
    if(std::abs(eit.key()/1e9 - getCurrentTime()/1e9) > window_padding*60.0){
      eit = events.erase(eit);
      continue;
    }
    eit++;
  }
}

void Replay::stream(SubMaster *sm){
  thread = new QThread;
  this->moveToThread(thread);
  QObject::connect(thread, &QThread::started, [=](){
    process(sm);
  });
  thread->start();

  seek_thread = new QThread;
  QObject::connect(seek_thread, &QThread::started, [=](){
    seekThread();
  });
  seek_thread->start();

  queue_thread = new QThread;
  QObject::connect(queue_thread, &QThread::started, [=](){
    seekRequestThread();
  });
  queue_thread->start();
}

void Replay::seekTime(int seek_){
  // TODO: see if eidx also needs to be cleared
  if(!seeking){
    if(seek >= 0){
      setSeekRequest(seek_*1e9);
    }

    if(seek_/60 != current_segment) {
      int last_segment = current_segment;
      current_segment = seek_/60;

      for(int i = 0 ; i < 2*window_padding + 1 ; i++) {
        // add segments that don't overlap
        int seek_ind = seek_/60 - window_padding + i;
        if(((last_segment + window_padding < seek_ind) || (last_segment - window_padding > seek_ind)) && (seek_ind >= 0)) {
          addSegment(seek_ind);
        }
        // remove current segments that don't overlap
        int cur_ind = last_segment - window_padding + i;
        if(((seek_/60 + window_padding < cur_ind) || (seek_/60 - window_padding > cur_ind)) && (cur_ind >= 0)) {
          trimSegment(cur_ind);
        }
      }
    }
  }
}

void Replay::seekRequestThread() {
  while(1) {
    if(seek_queue.size() > 0 && !seeking) {
      seekTime(seek_queue.dequeue());
    }
  }
}

void Replay::seekThread(){
  char c;
  while(1){
    c = getch();
    if(c == '\n'){
      printf("Enter seek request: ");
      std::string request;
      std::cin >> request;

      if(request[0] == 's') {
        request.erase(request.begin());
        seek = std::stoi(request)*60;
      } else {
        seek = std::stoi(request);
      }

      seek_queue.enqueue(seek);
      getch(); // remove \n from entering seek
    } else if (c == 'm') {
      seek_queue.enqueue((getRelativeCurrentTime() + 60*1e9)/1e9);
    } else if (c == 'M') {
      seek_queue.enqueue((getRelativeCurrentTime() - 60*1e9)/1e9);
    } else if (c == 's') {
      seek_queue.enqueue((getRelativeCurrentTime() + 10*1e9)/1e9);
    } else if (c == 'S') {
      seek_queue.enqueue((getRelativeCurrentTime() - 10*1e9)/1e9);
    } else if (c == 'G') {
      seek_queue.enqueue(0);
    } else if (c == ' ') {
      togglePause();
    }
  }
}

void Replay::process(SubMaster *sm) {

  active = true;

  qDebug() << "hello from unlogger thread";
  while (events.size() == 0) {
    qDebug() << "waiting for events";
    QThread::sleep(1);
  }
  qDebug() << "got events";

  route_t0 = events.firstKey();

  QElapsedTimer timer; timer.start();

  // loops
  while (active) {
    uint64_t t0 = (events.begin()+1).key();
    uint64_t t0r = timer.nsecsElapsed();
    qDebug() << "unlogging at" << t0;

    auto eit = events.lowerBound(t0);
    while(eit.key() - t0 > 1e9){
      eit = events.lowerBound(t0);
    }

    while ((eit != events.end()) && active) {
      while (paused) {
        QThread::usleep(1000);
        t0 = (*eit).getLogMonoTime();
        t0r = timer.nsecsElapsed();
      }

      if (seeking) {
        t0 = seek_request + route_t0 + 1;
        tc = t0;
        qDebug() << "seeking to" << t0;
        t0r = timer.nsecsElapsed();
        eit = events.lowerBound(t0);
        seek_request = 0;
        if ((eit == events.end()) || (eit.key() - t0 > 1e9)) {
          qWarning() << "seek off end";
          while((eit == events.end()) || (eit.key() - t0 > 1e9)) {
            qDebug() << "stuck";
            eit = events.lowerBound(t0);
            QThread::sleep(1);
          }
        }
        seeking = false;
      }

      float time_to_end = (current_segment + 2)*60.0 - getRelativeCurrentTime()/1e9;
      if (loading_segment && (time_to_end > 80.0)){
        loading_segment = false;
      }

      cereal::Event::Reader e = (*eit);

      capnp::DynamicStruct::Reader e_ds = static_cast<capnp::DynamicStruct::Reader>(e);
      std::string type;
      KJ_IF_MAYBE(e_, e_ds.which()){
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      auto it = socks.find(type);
      tc = tm;
      if (it != socks.end()) {
        long etime = tm-t0;

        float timestamp = (tm - route_t0)/1e9;
        if(std::abs(timestamp-last_print) > 5.0){
          last_print = timestamp;
          printf("at %f\n", last_print);
        }

        long rtime = timer.nsecsElapsed() - t0r;
        long us_behind = ((etime-rtime)*1e-3)+0.5;
        if (us_behind > 0) {
          if (us_behind > 1e6) {
            qWarning() << "OVER ONE SECOND BEHIND, HACKING" << us_behind;
            us_behind = 0;
            t0 = tm;
            t0r = timer.nsecsElapsed();
          }
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        if (type == "roadCameraState") {
          auto fr = e.getRoadCameraState();

          auto it_ = eidx.find(fr.getFrameId());
          if (it_ != eidx.end()) {
            auto pp = *it_;
            //qDebug() << fr.getRoadCameraStateId() << pp;

            if (frs.find(pp.first) != frs.end()) {
              auto frm = frs[pp.first];
              auto data = frm->get(pp.second);

              if (vipc_server == nullptr) {
                cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
                cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

                vipc_server = new VisionIpcServer("camerad", device_id, context);
                vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_BACK, 4, true, frm->width, frm->height);

                vipc_server->start_listener();
              }

              VisionBuf *buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
              memcpy(buf->addr, data, frm->getRGBSize());
              VisionIpcBufExtra extra = {};

              vipc_server->send(buf, &extra, false);
            }
          }
        }

        if (sm == nullptr){
          capnp::MallocMessageBuilder msg;
          msg.setRoot(e);
          auto words = capnp::messageToFlatArray(msg);
          auto bytes = words.asBytes();

          (*it)->send((char*)bytes.begin(), bytes.size());
        } else{
          std::vector<std::pair<std::string, cereal::Event::Reader>> messages;
          messages.push_back({type, e});
          sm->update_msgs(nanos_since_boot(), messages);
        }
      }
      ++eit;
      if ((time_to_end < 60.0) && !loading_segment){
        loading_segment = true;
        seek_queue.enqueue((current_segment+1)*60 + 1);
      }
    }
  }
}
