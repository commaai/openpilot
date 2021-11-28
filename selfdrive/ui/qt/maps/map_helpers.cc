#include "selfdrive/ui/qt/maps/map_helpers.h"

#include <QJsonDocument>
#include <QJsonObject>

#include "selfdrive/common/params.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"

QString get_mapbox_token() {
  // Valid for 4 weeks since we can't swap tokens on the fly
  return MAPBOX_TOKEN.isEmpty() ? CommaApi::create_jwt({}, 4 * 7 * 24 * 3600) : MAPBOX_TOKEN;
}

QMapboxGLSettings get_mapbox_settings() {
  QMapboxGLSettings settings;

  if (!Hardware::PC()) {
    settings.setCacheDatabasePath(MAPS_CACHE_PATH);
  }
  settings.setApiBaseUrl(MAPS_HOST);
  settings.setAccessToken(get_mapbox_token());

  return settings;
}

QGeoCoordinate to_QGeoCoordinate(const QMapbox::Coordinate &in) {
  return QGeoCoordinate(in.first, in.second);
}

QMapbox::CoordinatesCollections model_to_collection(
  const cereal::LiveLocationKalman::Measurement::Reader &calibratedOrientationECEF,
  const cereal::LiveLocationKalman::Measurement::Reader &positionECEF,
  const cereal::ModelDataV2::XYZTData::Reader &line){

  Eigen::Vector3d ecef(positionECEF.getValue()[0], positionECEF.getValue()[1], positionECEF.getValue()[2]);
  Eigen::Vector3d orient(calibratedOrientationECEF.getValue()[0], calibratedOrientationECEF.getValue()[1], calibratedOrientationECEF.getValue()[2]);
  Eigen::Matrix3d ecef_from_local = euler2rot(orient);

  QMapbox::Coordinates coordinates;
  auto x = line.getX();
  auto y = line.getY();
  auto z = line.getZ();
  for (int i = 0; i < x.size(); i++) {
    Eigen::Vector3d point_ecef = ecef_from_local * Eigen::Vector3d(x[i], y[i], z[i]) + ecef;
    Geodetic point_geodetic = ecef2geodetic((ECEF){.x = point_ecef[0], .y = point_ecef[1], .z = point_ecef[2]});
    QMapbox::Coordinate coordinate(point_geodetic.lat, point_geodetic.lon);
    coordinates.push_back(coordinate);
  }

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

QMapbox::CoordinatesCollections coordinate_to_collection(QMapbox::Coordinate c) {
  QMapbox::Coordinates coordinates;
  coordinates.push_back(c);

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

QMapbox::CoordinatesCollections capnp_coordinate_list_to_collection(const capnp::List<cereal::NavRoute::Coordinate>::Reader& coordinate_list) {
  QMapbox::Coordinates coordinates;

  for (auto const &c: coordinate_list) {
    QMapbox::Coordinate coordinate(c.getLatitude(), c.getLongitude());
    coordinates.push_back(coordinate);
  }

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;

}

QMapbox::CoordinatesCollections coordinate_list_to_collection(QList<QGeoCoordinate> coordinate_list) {
  QMapbox::Coordinates coordinates;

  for (auto &c : coordinate_list) {
    QMapbox::Coordinate coordinate(c.latitude(), c.longitude());
    coordinates.push_back(coordinate);
  }

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

QList<QGeoCoordinate> polyline_to_coordinate_list(const QString &polylineString) {
  QList<QGeoCoordinate> path;
  if (polylineString.isEmpty())
      return path;

  QByteArray data = polylineString.toLatin1();

  bool parsingLatitude = true;

  int shift = 0;
  int value = 0;

  QGeoCoordinate coord(0, 0);

  for (int i = 0; i < data.length(); ++i) {
      unsigned char c = data.at(i) - 63;

      value |= (c & 0x1f) << shift;
      shift += 5;

      // another chunk
      if (c & 0x20)
          continue;

      int diff = (value & 1) ? ~(value >> 1) : (value >> 1);

      if (parsingLatitude) {
          coord.setLatitude(coord.latitude() + (double)diff/1e6);
      } else {
          coord.setLongitude(coord.longitude() + (double)diff/1e6);
          path.append(coord);
      }

      parsingLatitude = !parsingLatitude;

      value = 0;
      shift = 0;
  }

  return path;
}

static QGeoCoordinate sub(QGeoCoordinate v, QGeoCoordinate w) {
  return QGeoCoordinate(v.latitude() - w.latitude(), v.longitude() - w.longitude());
}

static QGeoCoordinate add(QGeoCoordinate v, QGeoCoordinate w) {
  return QGeoCoordinate(v.latitude() + w.latitude(), v.longitude() + w.longitude());
}

static QGeoCoordinate mul(QGeoCoordinate v, float c) {
  return QGeoCoordinate(c * v.latitude(), c * v.longitude());
}

static float dot(QGeoCoordinate v, QGeoCoordinate w) {
  return v.latitude() * w.latitude() + v.longitude() * w.longitude();
}

float minimum_distance(QGeoCoordinate a, QGeoCoordinate b, QGeoCoordinate p) {
  const QGeoCoordinate ap = sub(p, a);
  const QGeoCoordinate ab = sub(b, a);
  const float t = std::clamp(dot(ap, ab) / dot(ab, ab), 0.0f, 1.0f);
  const QGeoCoordinate projection = add(a, mul(ab, t));
  return projection.distanceTo(p);
}

float distance_along_geometry(QList<QGeoCoordinate> geometry, QGeoCoordinate pos) {
  if (geometry.size() <= 2) {
    return geometry[0].distanceTo(pos);
  }

  // 1. Find segment that is closest to current position
  // 2. Total distance is sum of distance to start of closest segment
  //    + all previous segments
  double total_distance = 0;
  double total_distance_closest = 0;
  double closest_distance = std::numeric_limits<double>::max();

  for (int i = 0; i < geometry.size() - 1; i++) {
    double d = minimum_distance(geometry[i], geometry[i+1], pos);
    if (d < closest_distance) {
      closest_distance = d;
      total_distance_closest = total_distance + geometry[i].distanceTo(pos);
    }
    total_distance += geometry[i].distanceTo(geometry[i+1]);
  }

  return total_distance_closest;
}

std::optional<QMapbox::Coordinate> coordinate_from_param(std::string param) {
  QString json_str = QString::fromStdString(Params().get(param));
  if (json_str.isEmpty()) return {};

  QJsonDocument doc = QJsonDocument::fromJson(json_str.toUtf8());
  if (doc.isNull()) return {};

  QJsonObject json = doc.object();
  if (json["latitude"].isDouble() && json["longitude"].isDouble()) {
    QMapbox::Coordinate coord(json["latitude"].toDouble(), json["longitude"].toDouble());
    return coord;
  } else {
    return {};
  }
}
