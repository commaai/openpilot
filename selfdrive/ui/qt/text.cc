#include <cstdlib>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QScreen>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>
#include <QDesktopWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif

#define MAX_TEXT_SIZE 2048

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w.windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w.showFullScreen();
#endif

  QRect screen = QGuiApplication::primaryScreen()->geometry();

  QWidget window = QWidget();
  window.setFixedSize(screen.width(), screen.height());

  QVBoxLayout *main_layout = new QVBoxLayout();

  QString text = "";
  for (int i = 1; i < argc; i++) {
    if (i > 1) {
      text.append(" ");
    }
    text.append(argv[i]);
  }

  QLabel *label = new QLabel(text);
  label->setAlignment(Qt::AlignTop);
  main_layout->addWidget(label);

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::released, [=]() {
    std::system("sudo reboot");
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, SIGNAL(released()), &a, SLOT(quit()));
#endif

  main_layout->addWidget(btn);

  window.setLayout(main_layout);
  window.setStyleSheet(R"(
    QWidget {
      background-color: black;
      margin: 60px;
    }
    QLabel {
      color: white;
      font-size: 60px;
    }
    QPushButton {
      color: white;
      font-size: 50px;
      padding: 60px;
      margin-left: 1500px;
      border-color: white;
      border-width: 2px;
      border-style: solid;
      border-radius: 10px;
    }
  )");

  window.show();
  return a.exec();
}
