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

#define GIT_URL "https://github.com/commaai/openpilot.git"
#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"

#define CONTINUE_PATH "/data/continue.sh"

int get_progress(const std::string &line) {
  const std::string prefix = "Receiving objects: ";
  const int start_idx = line.find(prefix);
  if (start_idx != std::string::npos) {
    return std::stoi(line.substr(start_idx + prefix.length(), 3));
  }
  return -1;
}

bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

int Installer::fresh_clone() {
  int err;

  // Cleanup
  err = std::system("rm -rf /data/tmppilot /data/openpilot");
  if (err) return 1;

  // Clone
  auto clone_pipe = popen("git clone " GIT_URL " -b " BRANCH " --depth=1 --progress --recurse-submodules /data/tmppilot 2>&1 >/dev/null", "r");
  if (!clone_pipe) return 1;

  char c;
  std::string buffer;  // TODO: use QT
  while (fscanf(clone_pipe, "%c", &c) != EOF) {
    if (c == '\r' || c == '\n') {
//      std::cout << "--->>> " << buffer.c_str() << "|||\n";
      int progress = get_progress(buffer);
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

  err = std::system("cd /data/tmppilot && git remote set-url origin --push " GIT_SSH_URL);
  if (err) return 1;

  err = std::system("mv /data/tmppilot /data/openpilot");
  if (err) return 1;

#ifdef INTERNAL
  err = std::system("mkdir -p /data/params/d/");
  if (err) return 1;

  std::map<std::string, std::string> params = {
    {"SshEnabled", "1"},
    {"RecordFrontLock", "1"},
    {"GithubSshKeys", SSH_KEYS},
  };
  for (const auto& [key, value] : params) {
    std::ofstream param;
    param.open("/data/params/d/" + key);
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
  err = fresh_clone();
  if (err) return 1;

  // Write continue.sh
  err = std::system("cp /data/openpilot/installer/continue_openpilot.sh " CONTINUE_PATH);
  if (err == -1) return 1;

  return 0;
}

Installer::Installer(QWidget *parent) : QWidget(parent) {
  QGridLayout *main_layout = new QGridLayout(this);
  main_layout->setSpacing(0);
  main_layout->setMargin(200);

  progress_bar = new QProgressBar();
  progress_bar->setRange(5, 100);
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
  timer->start(100);
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

