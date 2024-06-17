// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#ifndef QMAPLIBRE_SETTINGS_H
#define QMAPLIBRE_SETTINGS_H

#include <QMapLibre/Export>
#include <QMapLibre/Types>

#include <QtCore/QString>
#include <QtGui/QImage>

#include <functional>
#include <memory>

// TODO: this will be wrapped at some point
namespace mbgl {
class TileServerOptions;
} // namespace mbgl

namespace QMapLibre {

class SettingsPrivate;

class Q_MAPLIBRE_CORE_EXPORT Settings {
public:
    enum GLContextMode : bool {
        UniqueGLContext,
        SharedGLContext
    };

    enum MapMode {
        Continuous = 0,
        Static
    };

    enum ConstrainMode {
        NoConstrain = 0,
        ConstrainHeightOnly,
        ConstrainWidthAndHeight
    };

    enum ViewportMode {
        DefaultViewport = 0,
        FlippedYViewport
    };

    enum ProviderTemplate {
        NoProvider = 0,
        MapLibreProvider,
        MapTilerProvider,
        MapboxProvider
    };

    using ResourceTransformFunction = std::function<std::string(const std::string &)>;

    explicit Settings(ProviderTemplate provider = NoProvider);
    ~Settings();
    Settings(const Settings &s);
    Settings(Settings &&s) noexcept;
    Settings &operator=(const Settings &s);
    Settings &operator=(Settings &&s) noexcept;

    [[nodiscard]] GLContextMode contextMode() const;
    void setContextMode(GLContextMode);

    [[nodiscard]] MapMode mapMode() const;
    void setMapMode(MapMode);

    [[nodiscard]] ConstrainMode constrainMode() const;
    void setConstrainMode(ConstrainMode);

    [[nodiscard]] ViewportMode viewportMode() const;
    void setViewportMode(ViewportMode);

    [[nodiscard]] unsigned cacheDatabaseMaximumSize() const;
    void setCacheDatabaseMaximumSize(unsigned);

    [[nodiscard]] QString cacheDatabasePath() const;
    void setCacheDatabasePath(const QString &path);

    [[nodiscard]] QString assetPath() const;
    void setAssetPath(const QString &path);

    [[nodiscard]] QString apiKey() const;
    void setApiKey(const QString &key);

    [[nodiscard]] QString apiBaseUrl() const;
    void setApiBaseUrl(const QString &url);

    [[nodiscard]] QString localFontFamily() const;
    void setLocalFontFamily(const QString &family);

    [[nodiscard]] QString clientName() const;
    void setClientName(const QString &name);

    [[nodiscard]] QString clientVersion() const;
    void setClientVersion(const QString &version);

    [[nodiscard]] ResourceTransformFunction resourceTransform() const;
    void setResourceTransform(const ResourceTransformFunction &transform);

    void setProviderTemplate(ProviderTemplate providerTemplate);
    void setStyles(const Styles &styles);

    [[nodiscard]] const Styles &styles() const;
    [[nodiscard]] Styles providerStyles() const;

    [[nodiscard]] Coordinate defaultCoordinate() const;
    void setDefaultCoordinate(const Coordinate &coordinate);
    [[nodiscard]] double defaultZoom() const;
    void setDefaultZoom(double zoom);

    [[nodiscard]] bool customTileServerOptions() const;
    [[nodiscard]] const mbgl::TileServerOptions &tileServerOptions() const;

private:
    std::unique_ptr<SettingsPrivate> d_ptr;
};

} // namespace QMapLibre

#endif // QMAPLIBRE_SETTINGS_H
