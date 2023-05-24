#include "selfdrive/ui/qt/qt_window.h"

void setMainWindow(QWidget *w) {
  const QSize sz = QGuiApplication::primaryScreen()->size();
  if (Hardware::PC() && sz.width() <= 1920 && sz.height() <= 1080 && getenv("SCALE") == nullptr) {
    w->setMinimumSize(QSize(640, 480)); // allow resize smaller than fullscreen
    w->setMaximumSize(QSize(2160, 1080));
    w->resize(sz);
  } else {
    const float scale = util::getenv("SCALE", 1.0f);
    const bool wide = (sz.width() >= WIDE_WIDTH) ^ (getenv("INVERT_WIDTH") != NULL);
    w->setFixedSize(QSize(wide ? WIDE_WIDTH : 1920, 1080) * scale);
  }
  w->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w->showFullScreen();

  // ensure we have a valid eglDisplay, otherwise the ui will silently fail
  void *egl = native->nativeResourceForWindow("egldisplay", w->windowHandle());
  assert(egl != nullptr);
#endif
}


extern "C" {
  void set_main_window(void *w) {
    setMainWindow((QWidget*)w);
  }
}
