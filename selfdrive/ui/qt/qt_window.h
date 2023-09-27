#pragma once

#include <string>

#include <QApplication>
#include <QScreen>
#include <QWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <wayland-client-protocol.h>
#include <QPlatformSurfaceEvent>
#endif

#include "system/hardware/hw.h"

const QString ASSET_PATH = ":/";
const QSize DEVICE_SCREEN_SIZE = {2160, 1080};

void setMainWindow(QWidget *w);
