#include <unistd.h>
#include <cstdlib>

#include <QLabel>
#include <QTimer>
#include <QWidget>
#include <QString>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif

#include "installer.hpp"

#ifndef BRANCH
#define BRANCH "master"
#endif

#define GIT_CLONE_COMMAND "git clone https://github.com/commaai/openpilot.git "

#define CONTINUE_PATH "/home/comma/continue.sh"

static bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

int fresh_clone() {
  int err;

  // Cleanup
  err = chdir("/tmp");
  if(err) return 1;
  err = system("rm -rf /tmp/openpilot");
  if(err) return 1;

  err = system(GIT_CLONE_COMMAND " -b " BRANCH " --depth=1 openpilot");
  if(err) return 1;

  return 0;

  // Cleanup old folder in /data
  err = system("rm -rf /data/openpilot");
  if(err) return 1;

  // this won't move if /data/openpilot exists
  err = system("mv /tmp/openpilot /data");
  if(err) return 1;

  return 0;
}

int Installer::install() {
  int err;

  // TODO: Disable SSH after install done

  // Wait for valid time
  while (!time_valid()) {
    usleep(500 * 1000);
    printf("Waiting for valid time\n");
  }

  printf("Doing fresh clone\n");
  err = fresh_clone();
  if(err) goto done;

  // Write continue.sh
  err = system("cp /data/openpilot/installer/continue_openpilot.sh " CONTINUE_PATH);
  if(err == -1) goto done;

done:
  emit done();
  return err != 0;
}

Installer::Installer(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  QLabel *title = new QLabel("Installing software...");
  title->setStyleSheet(R"(
    QLabel {
      font-size: 80px;
      font-weight: bold;
    }
  )");
  title->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(title, Qt::AlignCenter);

  setLayout(main_layout);
  setStyleSheet(R"(
    QWidget {
      color: white;
      background-color: black;
    }
  )");
}


int main(int argc, char *argv[]) {
#ifdef QCOM2
  int w = 2160, h = 1080;
#else
  int w = 1920, h = 1080;
#endif

  QApplication a(argc, argv);

  Installer installer;
  installer.setFixedSize(w, h);
  installer.show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", installer.windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  installer.showFullScreen();
#endif

  QObject::connect(&installer, SIGNAL(done()), &a, SLOT(quit()));
  QTimer::singleShot(1000, &installer, SLOT(install()));

  return a.exec();
}
