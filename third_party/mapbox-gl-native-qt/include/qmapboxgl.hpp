#ifndef QMAPBOXGL_H
#define QMAPBOXGL_H

#include <QImage>
#include <QMapbox>
#include <QMargins>
#include <QObject>
#include <QPointF>
#include <QSize>
#include <QString>
#include <QStringList>

#include <functional>

class QMapboxGLPrivate;

// This header follows the Qt coding style: https://wiki.qt.io/Qt_Coding_Style

class Q_MAPBOXGL_EXPORT QMapboxGLSettings
{
public:
    QMapboxGLSettings();

    enum GLContextMode {
        UniqueGLContext = 0,
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

    GLContextMode contextMode() const;
    void setContextMode(GLContextMode);

    MapMode mapMode() const;
    void setMapMode(MapMode);

    ConstrainMode constrainMode() const;
    void setConstrainMode(ConstrainMode);

    ViewportMode viewportMode() const;
    void setViewportMode(ViewportMode);

    unsigned cacheDatabaseMaximumSize() const;
    void setCacheDatabaseMaximumSize(unsigned);

    QString cacheDatabasePath() const;
    void setCacheDatabasePath(const QString &);

    QString assetPath() const;
    void setAssetPath(const QString &);

    QString accessToken() const;
    void setAccessToken(const QString &);

    QString apiBaseUrl() const;
    void setApiBaseUrl(const QString &);

    QString localFontFamily() const;
    void setLocalFontFamily(const QString &);

    std::function<std::string(const std::string &)> resourceTransform() const;
    void setResourceTransform(const std::function<std::string(const std::string &)> &);

private:
    GLContextMode m_contextMode;
    MapMode m_mapMode;
    ConstrainMode m_constrainMode;
    ViewportMode m_viewportMode;

    unsigned m_cacheMaximumSize;
    QString m_cacheDatabasePath;
    QString m_assetPath;
    QString m_accessToken;
    QString m_apiBaseUrl;
    QString m_localFontFamily;
    std::function<std::string(const std::string &)> m_resourceTransform;
};

struct Q_MAPBOXGL_EXPORT QMapboxGLCameraOptions {
    QVariant center;  // Coordinate
    QVariant anchor;  // QPointF
    QVariant zoom;    // double
    QVariant bearing; // double
    QVariant pitch;   // double
};

class Q_MAPBOXGL_EXPORT QMapboxGL : public QObject
{
    Q_OBJECT
    Q_PROPERTY(double latitude READ latitude WRITE setLatitude)
    Q_PROPERTY(double longitude READ longitude WRITE setLongitude)
    Q_PROPERTY(double zoom READ zoom WRITE setZoom)
    Q_PROPERTY(double bearing READ bearing WRITE setBearing)
    Q_PROPERTY(double pitch READ pitch WRITE setPitch)
    Q_PROPERTY(QString styleJson READ styleJson WRITE setStyleJson)
    Q_PROPERTY(QString styleUrl READ styleUrl WRITE setStyleUrl)
    Q_PROPERTY(double scale READ scale WRITE setScale)
    Q_PROPERTY(QMapbox::Coordinate coordinate READ coordinate WRITE setCoordinate)
    Q_PROPERTY(QMargins margins READ margins WRITE setMargins)

public:
    enum MapChange {
        MapChangeRegionWillChange = 0,
        MapChangeRegionWillChangeAnimated,
        MapChangeRegionIsChanging,
        MapChangeRegionDidChange,
        MapChangeRegionDidChangeAnimated,
        MapChangeWillStartLoadingMap,
        MapChangeDidFinishLoadingMap,
        MapChangeDidFailLoadingMap,
        MapChangeWillStartRenderingFrame,
        MapChangeDidFinishRenderingFrame,
        MapChangeDidFinishRenderingFrameFullyRendered,
        MapChangeWillStartRenderingMap,
        MapChangeDidFinishRenderingMap,
        MapChangeDidFinishRenderingMapFullyRendered,
        MapChangeDidFinishLoadingStyle,
        MapChangeSourceDidChange
    };

    enum MapLoadingFailure {
        StyleParseFailure,
        StyleLoadFailure,
        NotFoundFailure,
        UnknownFailure
    };

    // Determines the orientation of the map.
    enum NorthOrientation {
        NorthUpwards, // Default
        NorthRightwards,
        NorthDownwards,
        NorthLeftwards,
    };

    QMapboxGL(QObject* parent = 0,
              const QMapboxGLSettings& = QMapboxGLSettings(),
              const QSize& size = QSize(),
              qreal pixelRatio = 1);
    virtual ~QMapboxGL();

    QString styleJson() const;
    QString styleUrl() const;

    void setStyleJson(const QString &);
    void setStyleUrl(const QString &);

    double latitude() const;
    void setLatitude(double latitude);

    double longitude() const;
    void setLongitude(double longitude);

    double scale() const;
    void setScale(double scale, const QPointF &center = QPointF());

    double zoom() const;
    void setZoom(double zoom);

    double minimumZoom() const;
    double maximumZoom() const;

    double bearing() const;
    void setBearing(double degrees);
    void setBearing(double degrees, const QPointF &center);

    double pitch() const;
    void setPitch(double pitch);
    void pitchBy(double pitch);

    NorthOrientation northOrientation() const;
    void setNorthOrientation(NorthOrientation);

    QMapbox::Coordinate coordinate() const;
    void setCoordinate(const QMapbox::Coordinate &);
    void setCoordinateZoom(const QMapbox::Coordinate &, double zoom);

    void jumpTo(const QMapboxGLCameraOptions&);

    void setGestureInProgress(bool inProgress);

    void setTransitionOptions(qint64 duration, qint64 delay = 0);

    void addAnnotationIcon(const QString &name, const QImage &sprite);

    QMapbox::AnnotationID addAnnotation(const QMapbox::Annotation &);
    void updateAnnotation(QMapbox::AnnotationID, const QMapbox::Annotation &);
    void removeAnnotation(QMapbox::AnnotationID);

    bool setLayoutProperty(const QString &layer, const QString &property, const QVariant &value);
    bool setPaintProperty(const QString &layer, const QString &property, const QVariant &value);

    bool isFullyLoaded() const;

    void moveBy(const QPointF &offset);
    void scaleBy(double scale, const QPointF &center = QPointF());
    void rotateBy(const QPointF &first, const QPointF &second);

    void resize(const QSize &size);

    double metersPerPixelAtLatitude(double latitude, double zoom) const;
    QMapbox::ProjectedMeters projectedMetersForCoordinate(const QMapbox::Coordinate &) const;
    QMapbox::Coordinate coordinateForProjectedMeters(const QMapbox::ProjectedMeters &) const;
    QPointF pixelForCoordinate(const QMapbox::Coordinate &) const;
    QMapbox::Coordinate coordinateForPixel(const QPointF &) const;

    QMapbox::CoordinateZoom coordinateZoomForBounds(const QMapbox::Coordinate &sw, QMapbox::Coordinate &ne) const;
    QMapbox::CoordinateZoom coordinateZoomForBounds(const QMapbox::Coordinate &sw, QMapbox::Coordinate &ne, double bearing, double pitch);

    void setMargins(const QMargins &margins);
    QMargins margins() const;

    void addSource(const QString &sourceID, const QVariantMap& params);
    bool sourceExists(const QString &sourceID);
    void updateSource(const QString &sourceID, const QVariantMap& params);
    void removeSource(const QString &sourceID);

    void addImage(const QString &name, const QImage &sprite);
    void removeImage(const QString &name);

    void addCustomLayer(const QString &id,
        QScopedPointer<QMapbox::CustomLayerHostInterface>& host,
        const QString& before = QString());
    void addLayer(const QVariantMap &params, const QString& before = QString());
    bool layerExists(const QString &id);
    void removeLayer(const QString &id);

    QVector<QString> layerIds() const;

    void setFilter(const QString &layer, const QVariant &filter);
    QVariant getFilter(const QString &layer) const;
    // When rendering on a different thread,
    // should be called on the render thread.
    void createRenderer();
    void destroyRenderer();
    void setFramebufferObject(quint32 fbo, const QSize &size);

public slots:
    void render();
    void connectionEstablished();

    // Commit changes, load all the resources
    // and renders the map when completed.
    void startStaticRender();

signals:
    void needsRendering();
    void mapChanged(QMapboxGL::MapChange);
    void mapLoadingFailed(QMapboxGL::MapLoadingFailure, const QString &reason);
    void copyrightsChanged(const QString &copyrightsHtml);

    void staticRenderFinished(const QString &error);

private:
    Q_DISABLE_COPY(QMapboxGL)

    QMapboxGLPrivate *d_ptr;
};

Q_DECLARE_METATYPE(QMapboxGL::MapChange);
Q_DECLARE_METATYPE(QMapboxGL::MapLoadingFailure);

#endif // QMAPBOXGL_H
