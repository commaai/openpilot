#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <map>

#include <QDebug>
#include <QTimer>
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
  return (1900 + sys_time->tm_year) >= 2020;
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

  QTimer::singleShot(100, this, &Installer::doInstall);

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

void Installer::doInstall() {
  int err;

  // wait for valid time
  while (!time_valid()) {
    usleep(500 * 1000);
    qDebug() << "Waiting for valid time";
  }

  // cleanup
  err = std::system("rm -rf /data/tmppilot /data/openpilot");
  if (err) goto fail;

  // TODO: support using the dashcam cache
  // do install
  qDebug() << "Doing fresh clone\n";
  freshClone();

  return;
fail:
  val->setText("Failed");
}

void Installer::freshClone() {
  QObject::connect(&proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Installer::cloneFinished);
  QObject::connect(&proc, &QProcess::readyReadStandardError, this, &Installer::readProgress);
  QStringList args = {"clone", "--progress", GIT_URL, "-b", BRANCH, "--depth=1", "--recurse-submodules", "/data/tmppilot"};
  proc.start("git", args);
}

void Installer::readProgress() {
  qDebug() << "\n\ngot progress";
  qDebug() << proc.readAllStandardError();
}

void Installer::cloneFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  qDebug() << "finished " << exitCode;

  if (exitCode != 0) {
    // reboot button?
    // cleanup?
    val->setText("Failed");
    return;
  }

  // TODO: error handling

  // move into place
  std::system("mv /data/tmppilot /data/openpilot");

  // write continue.sh
  std::system("cp /data/openpilot/installer/continue_openpilot.sh " CONTINUE_PATH);

#ifdef INTERNAL
  std::system("mkdir -p /data/params/d/");

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

  std::system("cd /data/tmppilot && git remote set-url origin --push " GIT_SSH_URL);
#endif
}



int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Installer installer;
  setMainWindow(&installer);
  return a.exec();
}
