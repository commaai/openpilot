#include <QDebug>
#include <QTimer>
#include <QPushButton>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/offroad/networking.h"
#include "selfdrive/ui/qt/setup/updater.h"

#define UPDATER_PATH "/data/openpilot/selfdrive/hardware/tici/agnos.py"
#define MANIFEST_PATH "/data/openpilot/selfdrive/hardware/tici/agnos.json"

void run(const char* cmd) {
  int err = std::system(cmd);
  assert(err == 0);
}

Updater::Updater(QWidget *parent) : QStackedWidget(parent) {

  // initial prompt screen
  prompt = new QWidget;
  {
    QVBoxLayout *layout = new QVBoxLayout(prompt);
    layout->setContentsMargins(100, 100, 100, 100);

    QLabel *title = new QLabel("Update Required");
    title->setStyleSheet("font-size: 80px; font-weight: bold;");
    layout->addWidget(title);

    QLabel *desc = new QLabel("An operating system update is required. Connect your device to WiFi for the fastest update experience. The download size is approximately 1GB.");
    desc->setWordWrap(true);
    desc->setStyleSheet("font-size: 65px;");
    layout->addWidget(desc);

    QHBoxLayout *hlayout = new QHBoxLayout;
    layout->addLayout(hlayout);

    QPushButton *connect = new QPushButton("Connect to WiFi");
    QObject::connect(connect, &QPushButton::clicked, [=]() {
      setCurrentWidget(wifi);
    });
    hlayout->addWidget(connect);

    QPushButton *install = new QPushButton("Install");
    QObject::connect(install, &QPushButton::clicked, this, &Updater::installUpdate);
    hlayout->addWidget(install);
  }

  // wifi connection screen
  wifi = new QWidget;
  {
    QVBoxLayout *layout = new QVBoxLayout(wifi);
    layout->setContentsMargins(100, 100, 100, 100);

    Networking *networking = new Networking(this, false);
    networking->setStyleSheet("Networking { background-color: #292929; border-radius: 13px; }");
    layout->addWidget(networking, 1);

    QHBoxLayout *hlayout = new QHBoxLayout;
    layout->addLayout(hlayout);

    QPushButton *back = new QPushButton("Back");
    QObject::connect(back, &QPushButton::clicked, [=]() {
      setCurrentWidget(prompt);
    });
    hlayout->addWidget(back);

    QPushButton *install = new QPushButton("Install");
    QObject::connect(install, &QPushButton::clicked, this, &Updater::installUpdate);
    hlayout->addWidget(install);
  }

  addWidget(prompt);
  addWidget(wifi);

  setStyleSheet(R"(
    * {
      color: white;
    }
    Updater {
      color: white;
      background-color: black;
    }
    QPushButton {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #333333;
    }
  )");
}

void Updater::installUpdate() {
  // set widget
  proc.start(UPDATER_PATH, {MANIFEST_PATH});
}

void Updater::updateProgress(int percent) {
  bar->setValue(percent);
  val->setText(QString("%1%").arg(percent));
  repaint();
}

void Updater::readProgress() {
  auto line = QString(proc.readAllStandardError());
  qDebug() << "got " << line;
}

void Updater::updateFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  qDebug() << "finished with " << exitCode;
  assert(exitCode == 0);
}

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Updater updater;
  setMainWindow(&updater);
  return a.exec();
}
