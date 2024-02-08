// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "settings.hpp"
#include "types.hpp"

#include <mbgl/util/tile_server_options.hpp>

#include <QtCore/QString>
#include <QtCore/QVector>

#include <functional>
#include <memory>

namespace mbgl {
class TileServerOptions;
} // namespace mbgl

namespace QMapLibre {

class SettingsPrivate {
public:
    SettingsPrivate();

    void setProviderTemplate(Settings::ProviderTemplate providerTemplate);
    void setProviderApiBaseUrl(const QString &url);

    Settings::GLContextMode m_contextMode{Settings::SharedGLContext};
    Settings::MapMode m_mapMode{Settings::Continuous};
    Settings::ConstrainMode m_constrainMode{Settings::ConstrainHeightOnly};
    Settings::ViewportMode m_viewportMode{Settings::DefaultViewport};
    Settings::ProviderTemplate m_providerTemplate{Settings::NoProvider};

    unsigned m_cacheMaximumSize;
    QString m_cacheDatabasePath;
    QString m_assetPath;
    QString m_apiKey;
    QString m_localFontFamily;
    QString m_clientName;
    QString m_clientVersion;

    Coordinate m_defaultCoordinate{};
    double m_defaultZoom{};

    Styles m_styles;

    std::function<std::string(const std::string &)> m_resourceTransform;

    bool m_customTileServerOptions{};
    mbgl::TileServerOptions m_tileServerOptions{};
};

} // namespace QMapLibre
