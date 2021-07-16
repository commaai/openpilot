#include "selfdrive/ui/qt/qt_window.h"

void setMainWindow(QWidget *w) {
  const bool wide = (QGuiApplication::primaryScreen()->size().width() >= WIDE_WIDTH) ^
                    (getenv("INVERT_WIDTH") != NULL);
  const float scale = util::getenv("SCALE", 1.0f);

  w->setFixedSize(QSize(wide ? WIDE_WIDTH : 1920, 1080) * scale);
  w->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w->showFullScreen();
#endif
}
