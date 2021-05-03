#include <string>

#include <QWidget>
#include <QApplication>
#include "common/vision_defs.h"
#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif


inline void setMainWindow(QWidget *w) {
  const float scale = getenv("SCALE") != NULL ? std::stof(getenv("SCALE")) : 1.0;
  w->setFixedSize(SCREEN_WIDTH*scale, SCREEN_HEIGHT*scale);
  w->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w->showFullScreen();
#endif
}
