#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <map>

#include <QDebug>
#include <QDir>
#include <QTimer>
#include <QVBoxLayout>

#include "selfdrive/ui/installer/installer.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"

#define GIT_URL "https://github.com/commaai/openpilot.git"
#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"

#ifdef QCOM
  #define CONTINUE_PATH "/data/data/com.termux/files/continue.sh"
#else
  #define CONTINUE_PATH "/data/continue.sh"
#endif

// TODO: remove the other paths after a bit
const QList<QString> CACHE_PATHS = {"/data/openpilot.cache", "/system/comma/openpilot", "/usr/comma/openpilot"};

#define INSTALL_PATH "/data/openpilot"
#define TMP_INSTALL_PATH "/data/tmppilot"

extern const uint8_t str_continue[] asm("_binary_selfdrive_ui_installer_continue_" BRAND "_sh_start");
extern const uint8_t str_continue_end[] asm("_binary_selfdrive_ui_installer_continue_" BRAND "_sh_end");

bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2020;
}

void run(const char* cmd) {
  int err = std::system(cmd);
  assert(err == 0);
}

Installer::Installer(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *layout = new QVBoxLayout(this);
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

  QObject::connect(&proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Installer::cloneFinished);
  QObject::connect(&proc, &QProcess::readyReadStandardError, this, &Installer::readProgress);

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
  update();
}

void Installer::doInstall() {
  // wait for valid time
  while (!time_valid()) {
    usleep(500 * 1000);
    qDebug() << "Waiting for valid time";
  }

  // cleanup
  run("rm -rf " TMP_INSTALL_PATH " " INSTALL_PATH);

  // find the cache path
  QString cache;
  for (const QString &path : CACHE_PATHS) {
    if (QDir(path).exists()) {
      cache = path;
      break;
    }
  }

  // do the install
  if (!cache.isEmpty()) {
    cachedFetch(cache);
  } else {
    freshClone();
  }
}

void Installer::freshClone() {
  qDebug() << "Doing fresh clone";
  proc.start("git", {"clone", "--progress", GIT_URL, "-b", BRANCH,
                     "--depth=1", "--recurse-submodules", TMP_INSTALL_PATH});
}

void Installer::cachedFetch(const QString &cache) {
  qDebug() << "Fetching with cache: " << cache;

  run(QString("cp -rp %1 %2").arg(cache, TMP_INSTALL_PATH).toStdString().c_str());
  int err = chdir(TMP_INSTALL_PATH);
  assert(err == 0);
  run("git remote set-branches --add origin " BRANCH);

  updateProgress(10);

  proc.setWorkingDirectory(TMP_INSTALL_PATH);
  proc.start("git", {"fetch", "--progress", "origin", BRANCH});
}

void Installer::readProgress() {
  const QVector<QPair<QString, int>> stages = {
    // prefix, weight in percentage
    {"Receiving objects: ", 91},
    {"Resolving deltas: ", 2},
    {"Updating files: ", 7},
  };

  auto line = QString(proc.readAllStandardError());

  int base = 0;
  for (const QPair kv : stages) {
    if (line.startsWith(kv.first)) {
      auto perc = line.split(kv.first)[1].split("%")[0];
      int p = base + int(perc.toFloat() / 100. * kv.second);
      updateProgress(p);
      break;
    }
    base += kv.second;
  }
}

void Installer::cloneFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  qDebug() << "git finished with " << exitCode;
  assert(exitCode == 0);

  updateProgress(100);

  // ensure correct branch is checked out
  int err = chdir(TMP_INSTALL_PATH);
  assert(err == 0);
  run("git checkout " BRANCH);
  run("git reset --hard origin/" BRANCH);

  // move into place
  run("mv " TMP_INSTALL_PATH " " INSTALL_PATH);

#ifdef INTERNAL
  run("mkdir -p /data/params/d/");

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
  run("cd " INSTALL_PATH " && "
      "git submodule update --init && "
      "git remote set-url origin --push " GIT_SSH_URL " && "
      "git config remote.origin.fetch \"+refs/heads/*:refs/remotes/origin/*\"");
#endif

  // write continue.sh
  FILE *of = fopen("/data/continue.sh.new", "wb");
  assert(of != NULL);

  size_t num = str_continue_end - str_continue;
  size_t num_written = fwrite(str_continue, 1, num, of);
  assert(num == num_written);
  fclose(of);

  run("chmod +x /data/continue.sh.new");
  run("mv /data/continue.sh.new " CONTINUE_PATH);

#ifdef QCOM
  QTimer::singleShot(100, &QCoreApplication::quit);
#else
  // wait for the installed software's UI to take over
  QTimer::singleShot(60 * 1000, &QCoreApplication::quit);
#endif
}

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Installer installer;
  setMainWindow(&installer);
  return a.exec();
}
