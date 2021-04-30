#include "replay.hpp"
#include <termios.h>
#include <iostream>

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
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
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

  seekTime(seek);
}

void Replay::addSegment(int i){
  if (lrs.find(i) != lrs.end()) {
    return;
  }

  QThread* thread = new QThread;

  QString log_fn = this->log_paths.at(i).toString();
  lrs.insert(i, new LogReader(log_fn, &events, &events_lock, &unlogger->eidx, i));

  lrs[i]->moveToThread(thread);
  QObject::connect(thread, SIGNAL (started()), lrs[i], SLOT (process()));
  thread->start();

  QString camera_fn = this->camera_paths.at(i).toString();
  frs.insert(i, new FrameReader(qPrintable(camera_fn)));
}

void Replay::trimSegment(int seg_num){
  lrs.remove(seg_num);
  frs.remove(seg_num);

  auto eit = events.begin();
  while(eit != events.end()){
    if(std::abs(eit.key()/1e9 - unlogger->getCurrentTime()/1e9) > window_padding*60.0){
      eit = events.erase(eit);
      continue;
    }
    eit++;
  }
}

void Replay::stream(SubMaster *sm){
  thread = new QThread;
  unlogger->moveToThread(thread);
  QObject::connect(thread, &QThread::started, [=](){
    unlogger->process(sm);
  });
  thread->start();

  seek_thread = new QThread;
  QObject::connect(seek_thread, &QThread::started, [=](){
    seekThread();
  });
  seek_thread->start();

  QObject::connect(unlogger, &Unlogger::loadSegment, [=](){
    addSegment(current_segment + window_padding + 1);
    trimSegment(current_segment - window_padding);
    current_segment++;
  });
}

void Replay::seekTime(int seek_){
  if(!unlogger->isSeeking()){
    unlogger->setSeekRequest(seek_*1e9);

    if(seek_/60 != current_segment) {
      for(int i = 0 ; i < 2*window_padding + 1 ; i++) {
        // add segments that don't overlap
        int seek_ind = seek_/60 - window_padding + i;
        if(((current_segment + window_padding < seek_ind) || (current_segment - window_padding > seek_ind)) && (seek_ind >= 0)) {
          addSegment(seek_ind);
        }
        // remove current segments that don't overlap
        int cur_ind = current_segment - window_padding + i;
        if(((seek_/60 + window_padding < cur_ind) || (seek_/60 - window_padding > cur_ind)) && (cur_ind >= 0)) {
          trimSegment(cur_ind);
        }
      }
      current_segment = seek_/60;
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

      seekTime(seek);
      getch(); // remove \n from entering seek
    } else if (c == 'm') {
      seekTime((unlogger->getRelativeCurrentTime() + 60*1e9)/1e9);
    } else if (c == 'M') {
      seekTime((unlogger->getRelativeCurrentTime() - 60*1e9)/1e9);
    } else if (c == 's') {
      seekTime((unlogger->getRelativeCurrentTime() + 10*1e9)/1e9);
    } else if (c == 'S') {
      seekTime((unlogger->getRelativeCurrentTime() - 10*1e9)/1e9);
    } else if (c == 'G') {
      seekTime(0);
    } else if (c == ' ') {
      unlogger->togglePause();
    }
  }
}
