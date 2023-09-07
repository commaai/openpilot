#pragma once

#include <optional>
#include <string>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <QMapboxGL>
#include <QGeoCoordinate>

#include "common/util.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"
#include "cereal/messaging/messaging.h"

const QString MAPBOX_TOKEN = util::getenv("MAPBOX_TOKEN").c_str();
const QString MAPS_HOST = util::getenv("MAPS_HOST", MAPBOX_TOKEN.isEmpty() ? "https://maps.comma.ai" : "https://api.mapbox.com").c_str();
const QString MAPS_CACHE_PATH = "/data/mbgl-cache-navd.db";

QString get_mapbox_token();
QMapboxGLSettings get_mapbox_settings();
QGeoCoordinate to_QGeoCoordinate(const QMapbox::Coordinate &in);
QMapbox::CoordinatesCollections model_to_collection(
  const cereal::LiveLocationKalman::Measurement::Reader &calibratedOrientationECEF,
  const cereal::LiveLocationKalman::Measurement::Reader &positionECEF,
  const cereal::XYZTData::Reader &line);
QMapbox::CoordinatesCollections coordinate_to_collection(const QMapbox::Coordinate &c);
QMapbox::CoordinatesCollections capnp_coordinate_list_to_collection(const capnp::List<cereal::NavRoute::Coordinate>::Reader &coordinate_list);
QMapbox::CoordinatesCollections coordinate_list_to_collection(const QList<QGeoCoordinate> &coordinate_list);
QList<QGeoCoordinate> polyline_to_coordinate_list(const QString &polylineString);
std::optional<QMapbox::Coordinate> coordinate_from_param(const std::string &param);
std::pair<QString, QString> map_format_distance(float d, bool is_metric);
