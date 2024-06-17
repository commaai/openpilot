// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2017 The Qt Company Ltd.
// Copyright (C) 2017 Mapbox, Inc.

// SPDX-License-Identifier: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only

#pragma once

#include "qgeomap.hpp"

#include <QMapLibre/Settings>
#include <QMapLibre/StyleParameter>

#include <QtLocation/private/qgeomap_p_p.h>

#include <QtCore/QHash>
#include <QtCore/QList>
#include <QtCore/QRectF>
#include <QtCore/QSharedPointer>
#include <QtCore/QTimer>
#include <QtCore/QVariant>

namespace QMapLibre {

class Map;
class StyleChange;

class QGeoMapMapLibrePrivate : public QGeoMapPrivate {
    Q_DECLARE_PUBLIC(QGeoMapMapLibre)

public:
    explicit QGeoMapMapLibrePrivate(QGeoMappingManagerEngine *engine);
    ~QGeoMapMapLibrePrivate() override;

    QSGNode *updateSceneGraph(QSGNode *oldNode, QQuickWindow *window);

    QGeoMap::ItemTypes supportedMapItemTypes() const override;
    void addMapItem(QDeclarativeGeoMapItemBase *item) override;
    void removeMapItem(QDeclarativeGeoMapItemBase *item) override;

    void addStyleParameter(StyleParameter *parameter);
    void removeStyleParameter(StyleParameter *parameter);
    void clearStyleParameters();

    /* Data members */
    enum SyncState : int {
        NoSync = 0,
        ViewportSync = 1 << 0,
        CameraDataSync = 1 << 1,
        MapTypeSync = 1 << 2,
        VisibleAreaSync = 1 << 3
    };
    Q_DECLARE_FLAGS(SyncStates, SyncState);

    Settings m_settings;
    QString m_mapItemsBefore;

    QList<StyleParameter *> m_mapParameters;

    QTimer m_refresh;
    bool m_shouldRefresh = true;
    bool m_warned = false;
    bool m_threadedRendering = false;
    bool m_styleLoaded = false;

    SyncStates m_syncState = NoSync;

    std::vector<std::unique_ptr<StyleChange>> m_styleChanges;

protected:
    void changeViewportSize(const QSize &size) override;
    void changeCameraData(const QGeoCameraData &data) override;
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    void changeActiveMapType(const QGeoMapType &mapType) override;
#else
    void changeActiveMapType(const QGeoMapType mapType) override;
#endif

    void setVisibleArea(const QRectF &visibleArea) override;
    QRectF visibleArea() const override;

private:
    Q_DISABLE_COPY(QGeoMapMapLibrePrivate);

    void syncStyleChanges(Map *map);
    void threadedRenderingHack(QQuickWindow *window, Map *map);

    QRectF m_visibleArea;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(QGeoMapMapLibrePrivate::SyncStates)

} // namespace QMapLibre
