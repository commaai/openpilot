// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "map.hpp"
#include "map_observer_p.hpp"
#include "map_renderer_p.hpp"

#include <mbgl/actor/actor.hpp>
#include <mbgl/map/map.hpp>
#include <mbgl/renderer/renderer_frontend.hpp>
#include <mbgl/storage/resource_transform.hpp>
#include <mbgl/util/geo.hpp>

#include <QtCore/QObject>
#include <QtCore/QSize>

#include <atomic>
#include <memory>

namespace QMapLibre {

class MapPrivate : public QObject, public mbgl::RendererFrontend {
    Q_OBJECT

public:
    explicit MapPrivate(Map *map, const Settings &settings, const QSize &size, qreal pixelRatio);
    ~MapPrivate() override;

    // mbgl::RendererFrontend implementation.
    void reset() final {}
    void setObserver(mbgl::RendererObserver &observer) final;
    void update(std::shared_ptr<mbgl::UpdateParameters> parameters) final;

    // These need to be called on the same thread.
    void createRenderer();
    void destroyRenderer();
    void render();
    void setFramebufferObject(quint32 fbo, const QSize &size);

    using PropertySetter = std::optional<mbgl::style::conversion::Error> (mbgl::style::Layer::*)(
        const std::string &, const mbgl::style::conversion::Convertible &);
    [[nodiscard]] bool setProperty(const PropertySetter &setter,
                                   const QString &layerId,
                                   const QString &name,
                                   const QVariant &value) const;

    mbgl::EdgeInsets margins;
    std::unique_ptr<mbgl::Map> mapObj{};

public slots:
    void requestRendering();

signals:
    void needsRendering();

private:
    Q_DISABLE_COPY(MapPrivate)

    std::recursive_mutex m_mapRendererMutex;
    std::shared_ptr<mbgl::RendererObserver> m_rendererObserver{};
    std::shared_ptr<mbgl::UpdateParameters> m_updateParameters{};

    std::unique_ptr<MapObserver> m_mapObserver{};
    std::unique_ptr<MapRenderer> m_mapRenderer{};
    std::unique_ptr<mbgl::Actor<mbgl::ResourceTransform::TransformCallback>> m_resourceTransform{};

    Settings::GLContextMode m_mode;
    qreal m_pixelRatio;

    QString m_localFontFamily;

    std::atomic_flag m_renderQueued = ATOMIC_FLAG_INIT;
};

} // namespace QMapLibre
