// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2019 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#ifndef QMAPLIBRE_TYPES_H
#define QMAPLIBRE_TYPES_H

#include <QMapLibre/Export>

#include <QtCore/QPair>
#include <QtCore/QString>
#include <QtCore/QVariant>
#include <QtCore/QVector>
#include <QtGui/QColor>
#include <utility>

namespace QMapLibre {

using Coordinate = QPair<double, double>;
using CoordinateZoom = QPair<Coordinate, double>;
using ProjectedMeters = QPair<double, double>;

using Coordinates = QVector<Coordinate>;
using CoordinatesCollection = QVector<Coordinates>;

using CoordinatesCollections = QVector<CoordinatesCollection>;

struct Q_MAPLIBRE_CORE_EXPORT Style {
    enum Type { // Taken from Qt to be in sync with QtLocation
        NoMap = 0,
        StreetMap,
        SatelliteMapDay,
        SatelliteMapNight,
        TerrainMap,
        HybridMap,
        TransitMap,
        GrayStreetMap,
        PedestrianMap,
        CarNavigationMap,
        CycleMap,
        CustomMap = 100
    };

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    explicit Style(QString url_, QString name_ = QString())
        : url(std::move(url_)),
          name(std::move(name_)) {}
#else
    explicit Style(QString url_ = QString(), QString name_ = QString())
        : url(std::move(url_)),
          name(std::move(name_)) {}
#endif

    QString url;
    QString name;
    QString description;
    bool night{};
    Type type{CustomMap};
};

using Styles = QVector<Style>;

struct Q_MAPLIBRE_CORE_EXPORT Feature {
    enum Type {
        PointType = 1,
        LineStringType,
        PolygonType
    };

    /*! Class constructor. */
    explicit Feature(Type type_ = PointType,
                     CoordinatesCollections geometry_ = CoordinatesCollections(),
                     QVariantMap properties_ = QVariantMap(),
                     QVariant id_ = QVariant())
        : type(type_),
          geometry(std::move(geometry_)),
          properties(std::move(properties_)),
          id(std::move(id_)) {}

    Type type;
    CoordinatesCollections geometry;
    QVariantMap properties;
    QVariant id;
};

struct Q_MAPLIBRE_CORE_EXPORT FeatureProperty {
    enum Type {
        LayoutProperty = 1,
        PaintProperty,
    };

    /*! Class constructor. */
    explicit FeatureProperty(Type type_, QString name_, QVariant value_)
        : type(type_),
          name(std::move(name_)),
          value(std::move(value_)) {}

    Type type;
    QString name;
    QVariant value;
};

struct Q_MAPLIBRE_CORE_EXPORT ShapeAnnotationGeometry {
    enum Type {
        LineStringType = 1,
        PolygonType,
        MultiLineStringType,
        MultiPolygonType
    };

    /*! Class constructor. */
    explicit ShapeAnnotationGeometry(Type type_ = LineStringType,
                                     CoordinatesCollections geometry_ = CoordinatesCollections())
        : type(type_),
          geometry(std::move(geometry_)) {}

    Type type;
    CoordinatesCollections geometry;
};

struct Q_MAPLIBRE_CORE_EXPORT SymbolAnnotation {
    Coordinate geometry;
    QString icon;
};

struct Q_MAPLIBRE_CORE_EXPORT LineAnnotation {
    /*! Class constructor. */
    explicit LineAnnotation(ShapeAnnotationGeometry geometry_ = ShapeAnnotationGeometry(),
                            float opacity_ = 1.0f,
                            float width_ = 1.0f,
                            const QColor &color_ = Qt::black)
        : geometry(std::move(geometry_)),
          opacity(opacity_),
          width(width_),
          color(color_) {}

    ShapeAnnotationGeometry geometry;
    float opacity;
    float width;
    QColor color;
};

struct Q_MAPLIBRE_CORE_EXPORT FillAnnotation {
    /*! Class constructor. */
    explicit FillAnnotation(ShapeAnnotationGeometry geometry_ = ShapeAnnotationGeometry(),
                            float opacity_ = 1.0f,
                            const QColor &color_ = Qt::black,
                            QVariant outlineColor_ = QVariant())
        : geometry(std::move(geometry_)),
          opacity(opacity_),
          color(color_),
          outlineColor(std::move(outlineColor_)) {}

    ShapeAnnotationGeometry geometry;
    float opacity;
    QColor color;
    QVariant outlineColor;
};

using Annotation = QVariant;
using AnnotationID = quint32;
using AnnotationIDs = QVector<AnnotationID>;

struct Q_MAPLIBRE_CORE_EXPORT CameraOptions {
    QVariant center;  // Coordinate
    QVariant anchor;  // QPointF
    QVariant zoom;    // double
    QVariant bearing; // double
    QVariant pitch;   // double
};

// This struct is a 1:1 copy of mbgl::CustomLayerRenderParameters.
struct Q_MAPLIBRE_CORE_EXPORT CustomLayerRenderParameters {
    double width;
    double height;
    double latitude;
    double longitude;
    double zoom;
    double bearing;
    double pitch;
    double fieldOfView;
};

class Q_MAPLIBRE_CORE_EXPORT CustomLayerHostInterface {
public:
    virtual ~CustomLayerHostInterface() = default;
    virtual void initialize() = 0;
    virtual void render(const CustomLayerRenderParameters &) = 0;
    virtual void deinitialize() = 0;
};

} // namespace QMapLibre

Q_DECLARE_METATYPE(QMapLibre::Coordinate);
Q_DECLARE_METATYPE(QMapLibre::Coordinates);
Q_DECLARE_METATYPE(QMapLibre::CoordinatesCollection);
Q_DECLARE_METATYPE(QMapLibre::CoordinatesCollections);
Q_DECLARE_METATYPE(QMapLibre::Feature);

Q_DECLARE_METATYPE(QMapLibre::SymbolAnnotation);
Q_DECLARE_METATYPE(QMapLibre::ShapeAnnotationGeometry);
Q_DECLARE_METATYPE(QMapLibre::LineAnnotation);
Q_DECLARE_METATYPE(QMapLibre::FillAnnotation);

#endif // QMAPLIBRE_TYPES_H
