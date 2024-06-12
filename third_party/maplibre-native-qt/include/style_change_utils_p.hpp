// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2017 Mapbox, Inc.

// SPDX-License-Identifier: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only

#pragma once

#include <QMapLibre/Types>

#include <QtLocation/private/qdeclarativecirclemapitem_p.h>
#include <QtLocation/private/qdeclarativegeomapitembase_p.h>
#include <QtLocation/private/qdeclarativepolygonmapitem_p.h>
#include <QtLocation/private/qdeclarativepolylinemapitem_p.h>
#include <QtLocation/private/qdeclarativerectanglemapitem_p.h>

namespace QMapLibre::StyleChangeUtils {

Feature featureFromMapRectangle(QDeclarativeRectangleMapItem *item);
Feature featureFromMapCircle(QDeclarativeCircleMapItem *item);
Feature featureFromMapPolygon(QDeclarativePolygonMapItem *item);
Feature featureFromMapPolyline(QDeclarativePolylineMapItem *item);
Feature featureFromMapItem(QDeclarativeGeoMapItemBase *item);

QString featureId(QDeclarativeGeoMapItemBase *item);
std::vector<FeatureProperty> featureLayoutPropertiesFromMapPolyline(QDeclarativePolylineMapItem *item);
std::vector<FeatureProperty> featureLayoutPropertiesFromMapItem(QDeclarativeGeoMapItemBase *item);
std::vector<FeatureProperty> featurePaintPropertiesFromMapRectangle(QDeclarativeRectangleMapItem *item);
std::vector<FeatureProperty> featurePaingPropertiesFromMapCircle(QDeclarativeCircleMapItem *item);
std::vector<FeatureProperty> featurePaintPropertiesFromMapPolygon(QDeclarativePolygonMapItem *item);
std::vector<FeatureProperty> featurePaintPropertiesFromMapPolyline(QDeclarativePolylineMapItem *item);
std::vector<FeatureProperty> featurePaintPropertiesFromMapItem(QDeclarativeGeoMapItemBase *item);
std::vector<FeatureProperty> featurePropertiesFromMapItem(QDeclarativeGeoMapItemBase *item);

} // namespace QMapLibre::StyleChangeUtils
