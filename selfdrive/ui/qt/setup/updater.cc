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

	// progress screen
  progress = new QWidget;
  {
		QVBoxLayout *layout = new QVBoxLayout(progress);
		layout->setContentsMargins(150, 330, 150, 150);
		layout->setSpacing(0);

		text = new QLabel("Installing...");
		text->setStyleSheet("font-size: 90px; font-weight: 600;");
		layout->addWidget(text, 0, Qt::AlignTop);

		layout->addSpacing(100);

		bar = new QProgressBar();
		bar->setRange(0, 100);
		bar->setTextVisible(false);
		bar->setFixedHeight(72);
		layout->addWidget(bar, 0, Qt::AlignTop);

		layout->addStretch();
  }

  addWidget(prompt);
  addWidget(wifi);
  addWidget(progress);

  setStyleSheet(R"(
    * {
      color: white;
			font-family: Inter;
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
  QObject::connect(&proc, &QProcess::readyReadStandardError, this, &Updater::readProgress);
  proc.start(UPDATER_PATH, {"--swap", MANIFEST_PATH});
}

void Updater::readProgress() {
  auto lines = QString(proc.readAllStandardError());
  auto line = lines.trimmed().split("\n").last();

	auto parts = line.split(":");
	if (parts.size() == 2) {
		text->setText(parts[0]);
		bar->setValue((int)parts[1].toDouble());
		repaint();
	}
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
