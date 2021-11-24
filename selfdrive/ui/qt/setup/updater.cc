#include <QDebug>
#include <QTimer>
#include <QVBoxLayout>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/setup/updater.h"

#ifndef QCOM
#include "selfdrive/ui/qt/offroad/networking.h"
#endif

Updater::Updater(const QString &updater_path, const QString &manifest_path, QWidget *parent)
  : updater(updater_path), manifest(manifest_path), QStackedWidget(parent) {

  assert(updater.size());
  assert(manifest.size());

  // initial prompt screen
  prompt = new QWidget;
  {
    QVBoxLayout *layout = new QVBoxLayout(prompt);
    layout->setContentsMargins(100, 250, 100, 100);

    QLabel *title = new QLabel("Update Required");
    title->setStyleSheet("font-size: 80px; font-weight: bold;");
    layout->addWidget(title);

    layout->addSpacing(75);

    QLabel *desc = new QLabel("An operating system update is required. Connect your device to Wi-Fi for the fastest update experience. The download size is approximately 1GB.");
    desc->setWordWrap(true);
    desc->setStyleSheet("font-size: 65px;");
    layout->addWidget(desc);

    layout->addStretch();

    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setSpacing(30);
    layout->addLayout(hlayout);

    QPushButton *connect = new QPushButton("Connect to Wi-Fi");
    connect->setObjectName("navBtn");
    QObject::connect(connect, &QPushButton::clicked, [=]() {
#ifndef QCOM
      setCurrentWidget(wifi);
#else
      HardwareEon::launch_wifi();
#endif
    });
    hlayout->addWidget(connect);

    QPushButton *install = new QPushButton("Install");
    install->setObjectName("navBtn");
    install->setStyleSheet("background-color: #465BEA;");
    QObject::connect(install, &QPushButton::clicked, this, &Updater::installUpdate);
    hlayout->addWidget(install);
  }

  // wifi connection screen
  wifi = new QWidget;
  {
    QVBoxLayout *layout = new QVBoxLayout(wifi);
    layout->setContentsMargins(100, 100, 100, 100);

#ifndef QCOM
    Networking *networking = new Networking(this, false);
    networking->setStyleSheet("Networking { background-color: #292929; border-radius: 13px; }");
    layout->addWidget(networking, 1);
#endif

    QPushButton *back = new QPushButton("Back");
    back->setObjectName("navBtn");
    back->setStyleSheet("padding-left: 60px; padding-right: 60px;");
    QObject::connect(back, &QPushButton::clicked, [=]() {
      setCurrentWidget(prompt);
    });
    layout->addWidget(back, 0, Qt::AlignLeft);
  }

  // progress screen
  progress = new QWidget;
  {
    QVBoxLayout *layout = new QVBoxLayout(progress);
    layout->setContentsMargins(150, 330, 150, 150);
    layout->setSpacing(0);

    text = new QLabel("Loading...");
    text->setStyleSheet("font-size: 90px; font-weight: 600;");
    layout->addWidget(text, 0, Qt::AlignTop);

    layout->addSpacing(100);

    bar = new QProgressBar();
    bar->setRange(0, 100);
    bar->setTextVisible(false);
    bar->setFixedHeight(72);
    layout->addWidget(bar, 0, Qt::AlignTop);

    layout->addStretch();

    reboot = new QPushButton("Reboot");
    reboot->setObjectName("navBtn");
    reboot->setStyleSheet("padding-left: 60px; padding-right: 60px;");
    QObject::connect(reboot, &QPushButton::clicked, [=]() {
      Hardware::reboot();
    });
    layout->addWidget(reboot, 0, Qt::AlignLeft);
    reboot->hide();

    layout->addStretch();
  }

  addWidget(prompt);
  addWidget(wifi);
  addWidget(progress);

  setStyleSheet(R"(
    * {
      color: white;
      outline: none;
      font-family: Inter;
    }
    Updater {
      color: white;
      background-color: black;
    }
    QPushButton#navBtn {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #333333;
    }
    QProgressBar {
      border: none;
      background-color: #292929;
    }
    QProgressBar::chunk {
      background-color: #364DEF;
    }
  )");
}

void Updater::installUpdate() {
  setCurrentWidget(progress);
  QObject::connect(&proc, &QProcess::readyReadStandardOutput, this, &Updater::readProgress);
  QObject::connect(&proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Updater::updateFinished);
  proc.setProcessChannelMode(QProcess::ForwardedErrorChannel);
  proc.start(updater, {"--swap", manifest});
}

void Updater::readProgress() {
  auto lines = QString(proc.readAllStandardOutput());
  for (const QString &line : lines.trimmed().split("\n")) {
    auto parts = line.split(":");
    if (parts.size() == 2) {
      text->setText(parts[0]);
      bar->setValue((int)parts[1].toDouble());
    } else {
      qDebug() << line;
    }
  }
  update();
}

void Updater::updateFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  qDebug() << "finished with " << exitCode;
  if (exitCode == 0) {
    Hardware::reboot();
  } else {
    text->setText("Update failed");
    reboot->show();
  }
}

bool Updater::eventFilter(QObject *obj, QEvent *event) {
#ifdef QCOM
  // filter out touches while in android activity
  const static QSet<QEvent::Type> filter_events({QEvent::MouseButtonPress, QEvent::MouseMove, QEvent::TouchBegin, QEvent::TouchUpdate, QEvent::TouchEnd});
  if (HardwareEon::launched_activity && filter_events.contains(event->type())) {
    HardwareEon::check_activity();
    if (HardwareEon::launched_activity) {
      return true;
    }
  }
#endif
  return false;
}

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Updater updater(argv[1], argv[2]);
  setMainWindow(&updater);
  a.installEventFilter(&updater);
  return a.exec();
}
