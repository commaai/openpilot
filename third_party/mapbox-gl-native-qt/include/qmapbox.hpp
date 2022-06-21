#ifndef QMAPBOX_H
#define QMAPBOX_H

#include <QColor>
#include <QPair>
#include <QString>
#include <QVariant>
#include <QVector>

// This header follows the Qt coding style: https://wiki.qt.io/Qt_Coding_Style

#if !defined(QT_MAPBOXGL_STATIC)
#  if defined(QT_BUILD_MAPBOXGL_LIB)
#    define Q_MAPBOXGL_EXPORT Q_DECL_EXPORT
#  else
#    define Q_MAPBOXGL_EXPORT Q_DECL_IMPORT
#  endif
#else
#  define Q_MAPBOXGL_EXPORT
#endif

namespace QMapbox {

typedef QPair<double, double> Coordinate;
typedef QPair<Coordinate, double> CoordinateZoom;
typedef QPair<double, double> ProjectedMeters;

typedef QVector<Coordinate> Coordinates;
typedef QVector<Coordinates> CoordinatesCollection;

typedef QVector<CoordinatesCollection> CoordinatesCollections;

struct Q_MAPBOXGL_EXPORT Feature {
    enum Type {
        PointType = 1,
        LineStringType,
        PolygonType
    };

    /*! Class constructor. */
    Feature(Type type_ = PointType, const CoordinatesCollections& geometry_ = CoordinatesCollections(),
            const QVariantMap& properties_ = QVariantMap(), const QVariant& id_ = QVariant())
        : type(type_), geometry(geometry_), properties(properties_), id(id_) {}

    Type type;
    CoordinatesCollections geometry;
    QVariantMap properties;
    QVariant id;
};

struct Q_MAPBOXGL_EXPORT ShapeAnnotationGeometry {
    enum Type {
        LineStringType = 1,
        PolygonType,
        MultiLineStringType,
        MultiPolygonType
    };

    /*! Class constructor. */
    ShapeAnnotationGeometry(Type type_ = LineStringType, const CoordinatesCollections& geometry_ = CoordinatesCollections())
        : type(type_), geometry(geometry_) {}

    Type type;
    CoordinatesCollections geometry;
};

struct Q_MAPBOXGL_EXPORT SymbolAnnotation {
    Coordinate geometry;
    QString icon;
};

struct Q_MAPBOXGL_EXPORT LineAnnotation {
    /*! Class constructor. */
    LineAnnotation(const ShapeAnnotationGeometry& geometry_ = ShapeAnnotationGeometry(), float opacity_ = 1.0f,
            float width_ = 1.0f, const QColor& color_ = Qt::black)
        : geometry(geometry_), opacity(opacity_), width(width_), color(color_) {}

    ShapeAnnotationGeometry geometry;
    float opacity;
    float width;
    QColor color;
};

struct Q_MAPBOXGL_EXPORT FillAnnotation {
    /*! Class constructor. */
    FillAnnotation(const ShapeAnnotationGeometry& geometry_ = ShapeAnnotationGeometry(), float opacity_ = 1.0f,
            const QColor& color_ = Qt::black, const QVariant& outlineColor_ = QVariant())
        : geometry(geometry_), opacity(opacity_), color(color_), outlineColor(outlineColor_) {}

    ShapeAnnotationGeometry geometry;
    float opacity;
    QColor color;
    QVariant outlineColor;
};

typedef QVariant Annotation;
typedef quint32 AnnotationID;
typedef QVector<AnnotationID> AnnotationIDs;

enum NetworkMode {
    Online, // Default
    Offline,
};

Q_MAPBOXGL_EXPORT QVector<QPair<QString, QString> >& defaultStyles();

Q_MAPBOXGL_EXPORT NetworkMode networkMode();
Q_MAPBOXGL_EXPORT void setNetworkMode(NetworkMode);

// This struct is a 1:1 copy of mbgl::CustomLayerRenderParameters.
struct Q_MAPBOXGL_EXPORT CustomLayerRenderParameters {
    double width;
    double height;
    double latitude;
    double longitude;
    double zoom;
    double bearing;
    double pitch;
    double fieldOfView;
};

class Q_MAPBOXGL_EXPORT CustomLayerHostInterface {
public:
    virtual ~CustomLayerHostInterface() = default;
    virtual void initialize() = 0;
    virtual void render(const CustomLayerRenderParameters&) = 0;
    virtual void deinitialize() = 0;
};

Q_MAPBOXGL_EXPORT double metersPerPixelAtLatitude(double latitude, double zoom);
Q_MAPBOXGL_EXPORT ProjectedMeters projectedMetersForCoordinate(const Coordinate &);
Q_MAPBOXGL_EXPORT Coordinate coordinateForProjectedMeters(const ProjectedMeters &);

} // namespace QMapbox

Q_DECLARE_METATYPE(QMapbox::Coordinate);
Q_DECLARE_METATYPE(QMapbox::Coordinates);
Q_DECLARE_METATYPE(QMapbox::CoordinatesCollection);
Q_DECLARE_METATYPE(QMapbox::CoordinatesCollections);
Q_DECLARE_METATYPE(QMapbox::Feature);

Q_DECLARE_METATYPE(QMapbox::SymbolAnnotation);
Q_DECLARE_METATYPE(QMapbox::ShapeAnnotationGeometry);
Q_DECLARE_METATYPE(QMapbox::LineAnnotation);
Q_DECLARE_METATYPE(QMapbox::FillAnnotation);

#endif // QMAPBOX_H
