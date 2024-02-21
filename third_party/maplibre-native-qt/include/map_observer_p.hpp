// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "map.hpp"

#include <mbgl/map/map_observer.hpp>
#include <mbgl/style/style.hpp>

#include <QtCore/QObject>

#include <exception>
#include <memory>

namespace QMapLibre {

class MapPrivate;

class MapObserver : public QObject, public mbgl::MapObserver {
    Q_OBJECT

public:
    explicit MapObserver(MapPrivate *ptr);
    ~MapObserver() override;

    // mbgl::MapObserver implementation.
    void onCameraWillChange(mbgl::MapObserver::CameraChangeMode mode) final;
    void onCameraIsChanging() final;
    void onCameraDidChange(mbgl::MapObserver::CameraChangeMode mode) final;
    void onWillStartLoadingMap() final;
    void onDidFinishLoadingMap() final;
    void onDidFailLoadingMap(mbgl::MapLoadError error, const std::string &what) final;
    void onWillStartRenderingFrame() final;
    void onDidFinishRenderingFrame(mbgl::MapObserver::RenderFrameStatus status) final;
    void onWillStartRenderingMap() final;
    void onDidFinishRenderingMap(mbgl::MapObserver::RenderMode mode) final;
    void onDidFinishLoadingStyle() final;
    void onSourceChanged(mbgl::style::Source &source) final;

signals:
    void mapChanged(Map::MapChange);
    void mapLoadingFailed(Map::MapLoadingFailure, const QString &reason);
    void copyrightsChanged(const QString &copyrightsHtml);

private:
    Q_DISABLE_COPY(MapObserver)

    MapPrivate *d_ptrRef;
};

} // namespace QMapLibre
