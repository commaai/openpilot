#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/setup/installer.h"

#define GIT_URL "https://github.com/commaai/openpilot.git"
#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"

#define CONTINUE_PATH "/data/continue.sh"

bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

int fresh_clone() {
  int err;

  // Cleanup
  err = std::system("rm -rf /data/tmppilot /data/openpilot");
  if (err) return 1;

  // Clone
  err = std::system("git clone " GIT_URL " -b " BRANCH " --depth=1 --recurse-submodules /data/tmppilot");
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

int install() {
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
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setContentsMargins(150, 290, 150, 150);

  QLabel *title = new QLabel("Installing...");
  title->setStyleSheet("font-size: 90px; font-weight: 600;");
  layout->addWidget(title, 0, Qt::AlignTop);

  bar = new QProgressBar();
  bar->setRange(2, 100);
  bar->setTextVisible(false);
  bar->setFixedHeight(72);
  layout->addWidget(bar);
  
  val = new QLabel("0%");
  val->setStyleSheet("font-size: 70px; font-weight: 300;");
  layout->addWidget(val, 0, Qt::AlignTop);

  updateProgress(69);

  setStyleSheet(R"(
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
}

void Installer::updateProgress(int percent) {
  bar->setValue(percent);
  val->setText(QString("%1%").arg(percent));
}

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Installer installer;
  setMainWindow(&installer);
  return a.exec();
}
