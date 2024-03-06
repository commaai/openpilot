// Copyright (C) 2023 MapLibre contributors
// Copyright (C) 2018 Mapbox, Inc.

// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "geojson_p.hpp"
#include "types.hpp"

#include <mbgl/style/conversion/geojson.hpp>
#include <mbgl/style/conversion_impl.hpp>

#include <QtCore/QVariant>
#include <QtGui/QColor>

#include <optional>

namespace mbgl::style::conversion {

std::string convertColor(const QColor &color);

template <>
class ConversionTraits<QVariant> {
public:
    static bool isUndefined(const QVariant &value) { return value.isNull() || !value.isValid(); }

    static bool isArray(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        return QMetaType::canConvert(value.metaType(), QMetaType(QMetaType::QVariantList));
#else
        return value.canConvert(QVariant::List);
#endif
    }

    static std::size_t arrayLength(const QVariant &value) { return value.toList().size(); }

    static QVariant arrayMember(const QVariant &value, std::size_t i) { return value.toList()[static_cast<int>(i)]; }

    static bool isObject(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        return QMetaType::canConvert(value.metaType(), QMetaType(QMetaType::QVariantMap)) ||
               value.typeId() == QMetaType::QByteArray
#else
        return value.canConvert(QVariant::Map) || value.type() == QVariant::ByteArray
#endif
               || QString(value.typeName()) == QStringLiteral("QMapLibre::Feature") ||
               value.userType() == qMetaTypeId<QVector<QMapLibre::Feature>>() ||
               value.userType() == qMetaTypeId<QList<QMapLibre::Feature>>() ||
               value.userType() == qMetaTypeId<std::list<QMapLibre::Feature>>();
    }

    static std::optional<QVariant> objectMember(const QVariant &value, const char *key) {
        auto map = value.toMap();
        auto iter = map.constFind(key);

        if (iter != map.constEnd()) {
            return iter.value();
        }

        return {};
    }

    template <class Fn>
    static std::optional<Error> eachMember(const QVariant &value, Fn &&fn) {
        auto map = value.toMap();
        auto iter = map.constBegin();

        while (iter != map.constEnd()) {
            std::optional<Error> result = fn(iter.key().toStdString(), QVariant(iter.value()));
            if (result) {
                return result;
            }

            ++iter;
        }

        return {};
    }

    static std::optional<bool> toBool(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() == QMetaType::Bool) {
#else
        if (value.type() == QVariant::Bool) {
#endif
            return value.toBool();
        }

        return {};
    }

    static std::optional<float> toNumber(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() == QMetaType::Int || value.typeId() == QMetaType::Double ||
            value.typeId() == QMetaType::Long || value.typeId() == QMetaType::LongLong ||
            value.typeId() == QMetaType::ULong || value.typeId() == QMetaType::ULongLong) {
#else
        if (value.type() == QVariant::Int || value.type() == QVariant::Double || value.type() == QVariant::LongLong ||
            value.type() == QVariant::ULongLong) {
#endif
            return value.toFloat();
        }

        return {};
    }

    static std::optional<double> toDouble(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() == QMetaType::Int || value.typeId() == QMetaType::Double ||
            value.typeId() == QMetaType::Long || value.typeId() == QMetaType::LongLong ||
            value.typeId() == QMetaType::ULong || value.typeId() == QMetaType::ULongLong) {
#else
        if (value.type() == QVariant::Int || value.type() == QVariant::Double || value.type() == QVariant::LongLong ||
            value.type() == QVariant::ULongLong) {
#endif
            return value.toDouble();
        }

        return {};
    }

    static std::optional<std::string> toString(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() == QMetaType::QString) {
            return value.toString().toStdString();
        }

        if (value.typeId() == QMetaType::QColor) {
            return convertColor(value.value<QColor>());
        }
#else
        if (value.type() == QVariant::String) {
            return value.toString().toStdString();
        }

        if (value.type() == QVariant::Color) {
            return convertColor(value.value<QColor>());
        }
#endif
        return {};
    }

    static std::optional<Value> toValue(const QVariant &value) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() == QMetaType::Bool) {
            return {value.toBool()};
        }

        if (value.typeId() == QMetaType::QString) {
            return {value.toString().toStdString()};
        }

        if (value.typeId() == QMetaType::QColor) {
            return {convertColor(value.value<QColor>())};
        }

        if (value.typeId() == QMetaType::Int) {
            return {static_cast<int64_t>(value.toInt())};
        }

        if (QMetaType::canConvert(value.metaType(), QMetaType(QMetaType::Double))) {
            return {value.toDouble()};
        }
#else
        if (value.type() == QVariant::Bool) {
            return {value.toBool()};
        }

        if (value.type() == QVariant::String) {
            return {value.toString().toStdString()};
        }

        if (value.type() == QVariant::Color) {
            return {convertColor(value.value<QColor>())};
        }

        if (value.type() == QVariant::Int) {
            return {static_cast<int64_t>(value.toInt())};
        }

        if (value.canConvert(QVariant::Double)) {
            return {value.toDouble()};
        }
#endif
        return {};
    }

    static std::optional<GeoJSON> toGeoJSON(const QVariant &value, Error &error) {
        if (value.typeName() == QStringLiteral("QMapLibre::Feature")) {
            return GeoJSON{QMapLibre::GeoJSON::asFeature(value.value<QMapLibre::Feature>())};
        }

        if (value.userType() == qMetaTypeId<QVector<QMapLibre::Feature>>()) {
            return featureCollectionToGeoJSON(value.value<QVector<QMapLibre::Feature>>());
        }

        if (value.userType() == qMetaTypeId<QList<QMapLibre::Feature>>()) {
            return featureCollectionToGeoJSON(value.value<QList<QMapLibre::Feature>>());
        }

        if (value.userType() == qMetaTypeId<std::list<QMapLibre::Feature>>()) {
            return featureCollectionToGeoJSON(value.value<std::list<QMapLibre::Feature>>());
        }

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        if (value.typeId() != QMetaType::QByteArray) {
#else
        if (value.type() != QVariant::ByteArray) {
#endif
            error = {"JSON data must be in QByteArray"};
            return {};
        }

        const QByteArray data = value.toByteArray();
        return parseGeoJSON(std::string(data.constData(), data.size()), error);
    }

private:
    template <typename T>
    static GeoJSON featureCollectionToGeoJSON(const T &features) {
        mapbox::feature::feature_collection<double> collection;
        collection.reserve(static_cast<std::size_t>(features.size()));
        for (const auto &feature : features) {
            collection.push_back(QMapLibre::GeoJSON::asFeature(feature));
        }
        return GeoJSON{std::move(collection)};
    }
};

template <class T, class... Args>
std::optional<T> convert(const QVariant &value, Error &error, Args &&...args) {
    return convert<T>(Convertible(value), error, std::forward<Args>(args)...);
}

inline std::string convertColor(const QColor &color) {
    return QString::asprintf("rgba(%d,%d,%d,%lf)", color.red(), color.green(), color.blue(), color.alphaF())
        .toStdString();
}

} // namespace mbgl::style::conversion
