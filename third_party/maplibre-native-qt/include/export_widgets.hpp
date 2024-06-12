// Copyright (C) 2023 MapLibre contributors

// SPDX-License-Identifier: BSD-2-Clause

#ifndef QMAPLIBRE_WIDGETS_EXPORT_H
#define QMAPLIBRE_WIDGETS_EXPORT_H

#include <QtCore/QtGlobal>

#if !defined(QT_MAPLIBRE_STATIC)
#if defined(QT_BUILD_MAPLIBRE_WIDGETS_LIB)
#define Q_MAPLIBRE_WIDGETS_EXPORT Q_DECL_EXPORT
#else
#define Q_MAPLIBRE_WIDGETS_EXPORT Q_DECL_IMPORT
#endif
#else
#define Q_MAPLIBRE_WIDGETS_EXPORT
#endif

#endif // QMAPLIBRE_WIDGETS_EXPORT_H
