#include "../../clib/FrameReader.hpp"
#include "TestFrameReader.hpp"

void TestFrameReader::frameread() {
  QElapsedTimer t;
  t.start();
  FrameReader fr("3a5d6ac1c23e5536/2019-10-29--10-06-58/2/fcamera.hevc");
  fr.get(2);
  //QThread::sleep(10);
  qDebug() << t.nsecsElapsed()*1e-9 << "seconds";
}

QTEST_MAIN(TestFrameReader)

