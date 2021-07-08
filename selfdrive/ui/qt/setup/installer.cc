#include "selfdrive/ui/qt/setup/installer.h"
#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

#include <QApplication>
#include <QGridLayout>
#include <QPainter>
#include <QString>
#include <QTransform>
#include <QDebug>
#include <QTimer>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

#ifndef BRANCH
#define BRANCH "master"
#endif

#ifdef MASTER
  const QVector<QString> stages = {"Receiving objects: ", "Resolving deltas: ", "Filtering content: "};
  const QVector<int> weights = {60, 2, 38};
#else
  const QVector<QString> stages = {"Receiving objects: ", "Resolving deltas: "};
  const QVector<int> weights = {95, 5};
#endif

#define GIT_URL "https://github.com/commaai/openpilot.git"
#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"

#define CONTINUE_PATH "/data/continue.sh"
#define BASEDIR "/data"

bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

int Installer::getProgress(const QString &line) {
  for (const QString &prefix : stages) {
    if (line.startsWith(prefix)) {
      currentStage = prefix;
      break;
    }
  }

  if (stages.contains(currentStage)) {
    const int startIdx = line.indexOf(currentStage);
    if (startIdx != -1) {
      int stageIdx = stages.indexOf(currentStage);
      const int weight = weights.at(stageIdx);
      QVector<int> offsets = weights.mid(0, stageIdx);

      float offset = std::accumulate(offsets.begin(), offsets.end(), 0);
      float value = line.mid(startIdx + currentStage.length(), 3).toFloat() / (100. / weight);
      return qRound(value + offset);
    }
  }
  return -1;
}

int Installer::freshClone() {
  int err;

  // Cleanup
  err = std::system("rm -rf " BASEDIR "/tmppilot " BASEDIR "/openpilot");
  if (err) return 1;

  // Clone
//  auto clone_pipe = popen("git clone " GIT_URL " -b " BRANCH " --depth=1 --progress --recurse-submodules " BASEDIR "/tmppilot 2>&1 >/dev/null", "r");
  auto clone_pipe = popen("git clone " GIT_URL " -b " BRANCH " --depth=1 --progress " BASEDIR "/tmppilot 2>&1 >/dev/null", "r");
  if (!clone_pipe) return 1;

  char c;
  QString buffer;  // TODO: use QT
  while (fscanf(clone_pipe, "%c", &c) != EOF) {
    if (c == '\r' || c == '\n') {
      qDebug() << "--->>>" << buffer << "|||\n";
      int progress = getProgress(buffer);
      if (progress != -1) {
        emit update(progress);
      }
      buffer.clear();
    } else if (c != '\n' && c != '\r') {
      buffer.push_back(c);
    }
  }

  err = pclose(clone_pipe);
  if (err) return 1;

  err = std::system("cd " BASEDIR "/tmppilot && git remote set-url origin --push " GIT_SSH_URL);
  if (err) return 1;

  err = std::system("mv " BASEDIR "/tmppilot " BASEDIR "/openpilot");
  if (err) return 1;

#ifdef INTERNAL
  err = std::system("mkdir -p " BASEDIR "/params/d/");
  if (err) return 1;

  std::map<std::string, std::string> params = {
    {"SshEnabled", "1"},
    {"RecordFrontLock", "1"},
    {"GithubSshKeys", SSH_KEYS},
  };
  for (const auto& [key, value] : params) {
    std::ofstream param;
    param.open("" BASEDIR "/params/d/" + key);
    param << value;
    param.close();
  }
#endif

  return 0;
}

int Installer::install() {
  int err;

  // Wait for valid time
  while (!time_valid()) {
    usleep(500 * 1000);
    std::cout << "Waiting for valid time\n";
  }

  std::cout << "Doing fresh clone\n";
  err = freshClone();
  if (err) return 1;

  // Write continue.sh
  err = std::system("cp " BASEDIR "/openpilot/installer/continue_openpilot.sh " CONTINUE_PATH);
  if (err == -1) return 1;

  return 0;
}

Installer::Installer(QWidget *parent) : QWidget(parent) {
  QGridLayout *main_layout = new QGridLayout(this);
  main_layout->setSpacing(0);
  main_layout->setMargin(200);

  progress_bar = new QProgressBar();
//  progress_bar->setRange(5, 100);
  progress_bar->setRange(0, 100);
  progress_bar->setTextVisible(false);
  progress_bar->setVisible(true);
  progress_bar->setFixedHeight(20);
  main_layout->addWidget(progress_bar, 1, 0, -1, -1, Qt::AlignHCenter);

  text = new QLabel("Downloading...");
  text->setVisible(true);
  main_layout->addWidget(text, 2, 0, Qt::AlignHCenter);

  setStyleSheet(R"(
    Installer {
      background-color: black;
    }
    * {
      background-color: transparent;
    }
    QLabel {
      color: white;
      font-size: 80px;
    }
    QProgressBar {
      background-color: #373737;
      width: 1000px;
      border solid white;
      border-radius: 10px;
    }
    QProgressBar::chunk {
      border-radius: 10px;
      background-color: white;
    }
  )");

  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, [=]() {
    QCoreApplication::processEvents(QEventLoop::AllEvents, 1);
    this->repaint();
    if (!started) {
      install();
      started = true;
    }
  });
  timer->start(100);  // TODO why is this needed?
};

void Installer::update(const int value) {
  qDebug() << "VALUE:" << value;
  progress_bar->setValue(value);
//  this->repaint();  // FIXME: doesn't work
//  QCoreApplication::processEvents(QEventLoop::AllEvents, 1000);
}

int main(int argc, char *argv[]) {
  setQtSurfaceFormat();

  Hardware::set_display_power(true);
  Hardware::set_brightness(65);

  QApplication a(argc, argv);
  Installer installer;
  setMainWindow(&installer);
  return a.exec();
}

