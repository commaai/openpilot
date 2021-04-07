#pragma once

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "zmq.hpp"
#include "clutil.h"
#include "Unlogger.hpp"
#include "FileReader.hpp"
#include "FrameReader.hpp"
#include "visionipc_server.h"
#include "cameras/camera_frame_stream.h"

class Replay : public QObject {

public:
	Replay(QString route_, int seek, int use_api);
  void replay();
	bool addSegment(int i);
	QJsonArray camera_paths;
	QJsonArray log_paths;
	int use_api;

protected:
	Unlogger *unlogger;

private:
	QString route;

	QReadWriteLock events_lock;
	Events events;

	QMap<int, LogReader*> lrs;
	QMap<int, FrameReader*> frs;

  VisionIpcServer *vipc_server;
};

