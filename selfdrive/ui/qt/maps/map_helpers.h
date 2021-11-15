#pragma once

#include <optional>
#include <eigen3/Eigen/Dense>
#include <QMapboxGL>
#include <QGeoCoordinate>

#include "selfdrive/common/util.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"
#include "cereal/messaging/messaging.h"

const QString MAPBOX_TOKEN = util::getenv("MAPBOX_TOKEN").c_str();
const QString MAPS_HOST = util::getenv("MAPS_HOST", MAPBOX_TOKEN.isEmpty() ? "https://maps.comma.ai" : "https://api.mapbox.com").c_str();

QString get_mapbox_token();
QGeoCoordinate to_QGeoCoordinate(const QMapbox::Coordinate &in);
QMapbox::CoordinatesCollections model_to_collection(
  const cereal::LiveLocationKalman::Measurement::Reader &calibratedOrientationECEF,
  const cereal::LiveLocationKalman::Measurement::Reader &positionECEF,
  const cereal::ModelDataV2::XYZTData::Reader &line);
QMapbox::CoordinatesCollections coordinate_to_collection(QMapbox::Coordinate c);
QMapbox::CoordinatesCollections capnp_coordinate_list_to_collection(const capnp::List<cereal::NavRoute::Coordinate>::Reader &coordinate_list);
QMapbox::CoordinatesCollections coordinate_list_to_collection(QList<QGeoCoordinate> coordinate_list);

float minimum_distance(QGeoCoordinate a, QGeoCoordinate b, QGeoCoordinate p);
std::optional<QMapbox::Coordinate> coordinate_from_param(std::string param);
float distance_along_geometry(QList<QGeoCoordinate> geometry, QGeoCoordinate pos);
