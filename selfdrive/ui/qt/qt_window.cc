#include "selfdrive/ui/qt/qt_window.h"

void setMainWindow(QWidget *w) {
  const bool wide = (QGuiApplication::primaryScreen()->size().width() >= WIDE_WIDTH) ^
                    (getenv("INVERT_WIDTH") != NULL);
  if constexpr (Hardware::PC()) {
    w->setMinimumSize(QSize(640, 480));
    w->setMaximumSize(QSize(WIDE_WIDTH, 1080));
    w->resize(QSize(1920, 1080));
  } else {
    w->setFixedSize(QSize(wide ? WIDE_WIDTH : 1920, 1080));
  }

  w->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w->showFullScreen();
#endif
}


extern "C" {
  void set_main_window(void *w) {
    setMainWindow((QWidget*)w);
  }
}
