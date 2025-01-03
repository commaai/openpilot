#include <sys/resource.h>

#include <QApplication>
#include <QtWidgets>
#include <QGraphicsScene>

#include "system/hardware/hw.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/notouch.h"
#include "selfdrive/ui/qt/qt_window.h"

const QString VIDEOS_PATH = "../assets/videos/out";
const QString GST_VIDEO_CMD = QString("gst-launch-1.0 -v multifilesrc location=\"%1\" loop=true ! decodebin ! videorate ! queue2 ! video/x-raw,framerate=20/1 ! queue2 ! videoconvert ! queue2 ! videoflip method=clockwise ! queue2 ! autovideosink");
//const QString GST_VIDEO_CMD = QString("gst-launch-1.0 filesrc location=\"%1\" ! decodebin ! autovideosink");

void TransparentWidget::mousePressEvent(QMouseEvent *event) {
  qDebug() << "transparent mouse event" << (*event).type();
}

void MainWindowNoTouch::mousePressEvent(QMouseEvent *event) {
  qDebug() << "mouse event" << (*event).type();
}

MainWindowNoTouch::MainWindowNoTouch(QWidget *parent) : QWidget(parent) {
  setpriority(PRIO_PROCESS, 0, -5);
  Hardware::set_brightness(100);

  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);

  process = new QProcess(this);

  QStringList filters;
  filters << "*.mp4" << "*.mkv" << "*.hevc";
  QDir videos_path = QDir(VIDEOS_PATH);
  videos_path.setNameFilters(filters);
  QString content_name;
  QStringList videos = videos_path.entryList();
  QStringList entries = videos;
  entries.prepend("Random");
  while ((content_name = MultiOptionDialog::getSelection(tr("Select content"), entries, "", this)).isEmpty()) {
    qDebug() << "No content selected!";
  }

  if (content_name == "Random") {
    content_name = videos.at(QRandomGenerator::global()->bounded(videos.size()));
  }

  content_name = VIDEOS_PATH + "/" + content_name;
  qDebug() << "Selected:" << content_name;

  // play video
  QTimer::singleShot(0, [=]() {
    std::system(QString("while true; do %1; done").arg(GST_VIDEO_CMD.arg(content_name)).toStdString().c_str());
  });

//  QString command = "/bin/bash";
//  QStringList args;
//  args << "-c" << "while true; do " + GST_VIDEO_CMD.arg(content_name) + "; done";
//  process->start(command, args);

//  QTimer::singleShot(5000, [&]() {
////  QObject::connect(process, &QProcess::started, [=](){
//    // transparent touch capture widget
//    TransparentWidget *w = new TransparentWidget();
//    w->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::X11BypassWindowManagerHint);
//    //  w->setStyleSheet("background-color:black;");
//    w->setAttribute(Qt::WA_TranslucentBackground);
//    w->setGeometry(QRect(QPoint(0, 0), QPoint(2160, 1080)));
//    w->show();
//    // transparent touch capture widget
//  });

  // no outline to prevent the focus rectangle
  setStyleSheet(R"(
    * {
      font-family: Inter;
      outline: none;
    }
  )");
  setAttribute(Qt::WA_NoSystemBackground);
}

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication a(argc, argv);
  MainWindowNoTouch w;
  setMainWindow(&w);

  return a.exec();
}
