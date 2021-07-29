#include <QDebug>
#include <QTimer>
#include <QPushButton>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/setup/updater.h"

#define UPDATER_PATH "/data/openpilot/selfdrive/hardware/tici/agnos.py"
#define MANIFEST_PATH "/data/openpilot/selfdrive/hardware/tici/agnos.json"

void run(const char* cmd) {
  int err = std::system(cmd);
  assert(err == 0);
}

Updater::Updater(QWidget *parent) : QStackedWidget(parent) {

  // prompt layout
  QWidget *w = new QWidget;
  QVBoxLayout *layout = new QVBoxLayout(w);
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

  QPushButton *wifi = new QPushButton("Connect to WiFi");
  hlayout->addWidget(wifi);

  QPushButton *install = new QPushButton("Install");
  QObject::connect(install, &QPushButton::clicked, this, &Updater::installUpdate);
  hlayout->addWidget(install);

  addWidget(w);

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

QWidget* Updater::buildProgressWidget() {

  QWidget *w = new QWidget;
  QVBoxLayout *layout = new QVBoxLayout(w);
  layout->setContentsMargins(150, 290, 150, 150);
  layout->setSpacing(0);

  QLabel *title = new QLabel("Installing...");
  title->setStyleSheet("font-size: 90px; font-weight: 600;");
  layout->addWidget(title, 0, Qt::AlignTop);

  layout->addSpacing(170);

  bar = new QProgressBar();
  bar->setRange(0, 100);
  bar->setTextVisible(false);
  bar->setFixedHeight(72);
  layout->addWidget(bar, 0, Qt::AlignTop);

  layout->addSpacing(30);

  val = new QLabel("0%");
  val->setStyleSheet("font-size: 70px; font-weight: 300;");
  layout->addWidget(val, 0, Qt::AlignTop);

  layout->addStretch();

  QObject::connect(&proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Updater::updateFinished);
  QObject::connect(&proc, &QProcess::readyReadStandardError, this, &Updater::readProgress);

  w->setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
    }
    QProgressBar {
      border: none;
      background-color: #292929;
    }
    QProgressBar::chunk {
      background-color: #364DEF;
    }
  )");
  return w;
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
