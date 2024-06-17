// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2017 The Qt Company Ltd.
// Copyright (C) 2017 Mapbox, Inc.

// SPDX-License-Identifier: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only

#pragma once

#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>
#include <QtQuick/QSGSimpleTextureNode>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QtOpenGL/QOpenGLFramebufferObject>
#else
#include <QtGui/QOpenGLFramebufferObject>
#endif

#include <QMapLibre/Map>

namespace QMapLibre {

class QGeoMapMapLibre;

class TextureNode : public QSGSimpleTextureNode {
public:
    TextureNode(const Settings &setting, const QSize &size, qreal pixelRatio, QGeoMapMapLibre *geoMap);

    [[nodiscard]] Map *map() const;

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    void resize(const QSize &size, qreal pixelRatio, QQuickWindow *window);
#else
    void resize(const QSize &size, qreal pixelRatio);
#endif
    void render(QQuickWindow *);

private:
    std::unique_ptr<Map> m_map{};
    std::unique_ptr<QOpenGLFramebufferObject> m_fbo{};
};

} // namespace QMapLibre
