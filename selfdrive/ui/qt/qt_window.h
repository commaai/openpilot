#pragma once

#include <QApplication>
#include <QScreen>
#include <QWidget>
#include <string>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif

#include "system/hardware/hw.h"

const QString ASSET_PATH = ":/";
const QSize DEVICE_SCREEN_SIZE = {2160, 1080};

void setMainWindow(QWidget *w);
