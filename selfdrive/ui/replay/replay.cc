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
  current_segment = seek/60;
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

  // add first window:
  for(int i = 0 ; i < 3 ; i++) {
    int ind = current_segment - 1 + i;
    if ((ind < camera_paths.size()) && (ind >= 0)) {
      addSegment(ind);
    }
  }
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

void Replay::trimSegment(){
	printf("BEFORE TRIM : %d\n", events.size());
	auto eit = events.begin();
	while(eit != events.end()){
		if(std::abs((*eit).first - current_segment) > 1){
			eit = events.erase(eit);
			continue;
		}
		eit++;
	}
	printf("AFTER TRIM : %d\n", events.size());
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
    updateSeek();
  });
  seek_thread->start();

  QObject::connect(unlogger, &Unlogger::loadSegment, [=](){
    addSegment(current_segment + 2);
    current_segment++;
    //trimSegment();
  });
  QObject::connect(unlogger, &Unlogger::trimSegments, [=](){
    //trimSegment();
  });
}

void Replay::stopStream(){
  unlogger->setActive(false);
  thread->quit();
}

void Replay::seekTime(int seek_){
  if(std::abs(seek_/60 - current_segment) > 1){
    stopStream();
    events.clear();
    lrs.clear();
    frs.clear();

    // Add 3 segment window for seek
    for(int i = 0 ; i < 3 ; i++){
      int ind = seek_/60 - 1 + i;
      if((ind < camera_paths.size()) && (ind >= 0)){
        addSegment(ind);
      }
    }
    thread->start();
    current_segment = seek_/60;
  }
  unlogger->setSeekRequest(seek_*1e9);
}

void Replay::updateSeek(){
	char c;
  while(1){
    c = getch();
    if(c == '\n'){
			printf("Enter seek request: ");
			std::cin >> seek;
      seekTime(seek);
      getch(); // remove \n from entering seek
    } else if (c == 'm') {
      unlogger->setSeekRequest(unlogger->getRelativeCurrentTime() + 60*1e9);
    } else if (c == 'M') {
      unlogger->setSeekRequest(unlogger->getRelativeCurrentTime() - 60*1e9);
    } else if (c == 's') {
      unlogger->setSeekRequest(unlogger->getRelativeCurrentTime() + 10*1e9);
    } else if (c == 'S') {
      unlogger->setSeekRequest(unlogger->getRelativeCurrentTime() - 10*1e9);
    } else if (c == 'G') {
      seekTime(0);
    } else if (c == ' ') {
      unlogger->togglePause();
    }
  }
}
