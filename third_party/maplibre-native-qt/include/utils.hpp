// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#ifndef QMAPLIBRE_UTILS_H
#define QMAPLIBRE_UTILS_H

#include <QMapLibre/Export>
#include <QMapLibre/Types>

namespace QMapLibre {

enum NetworkMode {
    Online, // Default
    Offline,
};

Q_MAPLIBRE_CORE_EXPORT NetworkMode networkMode();
Q_MAPLIBRE_CORE_EXPORT void setNetworkMode(NetworkMode mode);

Q_MAPLIBRE_CORE_EXPORT double metersPerPixelAtLatitude(double latitude, double zoom);
Q_MAPLIBRE_CORE_EXPORT ProjectedMeters projectedMetersForCoordinate(const Coordinate &coordinate);
Q_MAPLIBRE_CORE_EXPORT Coordinate coordinateForProjectedMeters(const ProjectedMeters &projectedMeters);

} // namespace QMapLibre

#endif // QMAPLIBRE_UTILS_H
