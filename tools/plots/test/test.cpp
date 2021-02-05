#include <QApplication>
#include <QLabel>
#include <QWidget>
#include <iostream>
#include <QObject>

#include "../../nui/FileReader.cpp"

int main(int argc, char *argv[ ])
{
	QApplication app(argc, argv);

	Events events_test;
	QReadWriteLock events_lock_test;
  	QMap<int, QPair<int, int> >eidx_test;

	const QString f_url = "http://data.comma.life/3a5d6ac1c23e5536/2019-10-29--10-06-58/0/rlog.bz2";
	const QString f_file = "./rlog.bz2";
	
	LogReader* testing_log = new LogReader(f_url, &events_test, &events_lock_test, &eidx_test);

	QThread* thread = new QThread;
	testing_log->moveToThread(thread);
	QObject::connect(thread, SIGNAL (started()), testing_log, SLOT (process()));
	QObject::connect(testing_log, SIGNAL (done()), thread, SLOT (quit()));
	thread->start();
	
	// hack to wait for thread to finish
	sleep(3);

	for(auto e : events_test){
		if(e.which() != cereal::Event::CAR_STATE)
			continue;

		auto cs = e.getCarState();

	}

	return app.exec();
}
