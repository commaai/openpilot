#include <cstdlib>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QPixmap>
#include <QVBoxLayout>
#include <QProgressBar>
#include <QApplication>
#include <QDesktopWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif


int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget *window = new QWidget();

  // TODO: get size from QScreen, doesn't work on tici
#ifdef QCOM2
  int w = 2160, h = 1080;
#else
  int w = 1920, h = 1080;
#endif
  window->setFixedSize(w, h);

  QVBoxLayout *main_layout = new QVBoxLayout();

  QPixmap pix("../assets/img_spinner_comma.png");
  QLabel *comma_img = new QLabel();
  comma_img->setPixmap(pix);
  comma_img->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
  main_layout->addWidget(comma_img);

  // TODO: read this from stdin
  QLabel *text = new QLabel("building boardd");
  text->setAlignment(Qt::AlignHCenter);
  main_layout->addWidget(text);

  QProgressBar *bar = new QProgressBar();
  bar->setMinimum(5);
  bar->setMaximum(100);
  bar->setValue(50);
  bar->setTextVisible(false);
  main_layout->addWidget(bar);

  window->setLayout(main_layout);
  window->setStyleSheet(R"(
    QWidget {
      margin: 60px;
      background-color: black;
    }
    QLabel {
      color: white;
      font-size: 30px;
    }
    QProgressBar {
      color: white;
      border: none;
      margin: 100px;
    }
  )");
  window->show();


#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", window->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  window->showFullScreen();
#endif

  return a.exec();
}
