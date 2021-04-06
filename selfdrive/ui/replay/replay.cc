#include "replay.hpp"

Replay::Replay(QString route_, int seek, int use_api_) : route(route_), use_api(use_api_){
  QThread* thread = new QThread;
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
  unlogger->moveToThread(thread);
  QObject::connect(thread, SIGNAL (started()), unlogger, SLOT (process()));
  thread->start();

  if (use_api) {
    QString settings;
    QFile file;
    file.setFileName("routes.json");
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    settings = file.readAll();
    file.close();

    QJsonDocument sd = QJsonDocument::fromJson(settings.toUtf8());
    qWarning() << sd.isNull(); // <- print false :)
    QJsonObject sett2 = sd.object();

    this->camera_paths = sett2.value("camera").toArray();
    this->log_paths = sett2.value("logs").toArray();
  }

	// add the first segment
  addSegment(seek/60);
}

bool Replay::addSegment(int i){
  if (lrs.find(i) == lrs.end()) {
    QString fn = QString("http://data.comma.life/%1/%2/rlog.bz2").arg(route).arg(i);

    QThread* thread = new QThread;
    if (!use_api) {
      lrs.insert(i, new LogReader(fn, &events, &events_lock, &unlogger->eidx));
    } else {
      QString log_fn = this->log_paths.at(i).toString();
      lrs.insert(i, new LogReader(log_fn, &events, &events_lock, &unlogger->eidx));

    }

    lrs[i]->moveToThread(thread);
    QObject::connect(thread, SIGNAL (started()), lrs[i], SLOT (process()));
    thread->start();

    QString frn = QString("http://data.comma.life/%1/%2/fcamera.hevc").arg(route).arg(i);

    if (!use_api) {
      frs.insert(i, new FrameReader(qPrintable(frn)));
    } else {
      QString camera_fn = this->camera_paths.at(i).toString();
      frs.insert(i, new FrameReader(qPrintable(camera_fn)));
    }
    return true;
  }
  return false;
}

