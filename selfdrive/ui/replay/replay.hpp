#pragma once

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "Unlogger.hpp"
#include "FrameReader.hpp"
#include "FileReader.hpp"

class Replay : public QObject {

public:
	Replay(QString route_, int seek, int use_api);
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
};

