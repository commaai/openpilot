#pragma once

#include <string>

#include <QApplication>
#include <QWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <wayland-client-protocol.h>
#include <QPlatformSurfaceEvent>
#endif

#include "selfdrive/hardware/hw.h"

const QString ASSET_PATH = ":/";

const bool WIDE_UI = Hardware::TICI() || getenv("WIDE_UI") != nullptr;
const int vwp_w = WIDE_UI ? 2160 : 1920;
const int vwp_h = 1080;

inline void setMainWindow(QWidget *w) {
  const float scale = util::getenv("SCALE", 1.0f);
  w->setFixedSize(vwp_w*scale, vwp_h*scale);
  w->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", w->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  w->showFullScreen();
#endif
}
