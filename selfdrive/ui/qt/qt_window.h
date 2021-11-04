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

#include "selfdrive/hardware/hw.h"

const QString ASSET_PATH = ":/";

const int WIDE_WIDTH = 2160;

void setMainWindow(QWidget *w);
