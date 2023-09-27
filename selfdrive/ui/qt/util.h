#pragma once

#include <optional>

#include <QDateTime>
#include <QLayout>
#include <QPainter>
#include <QPixmap>
#include <QSurfaceFormat>
#include <QWidget>

QString getVersion();
QString getBrand();
QString getBrandVersion();
QString getUserAgent();
std::optional<QString> getDongleId();
void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
void setQtSurfaceFormat();
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void initApp();
QWidget* topWidget (QWidget* widget);
QPixmap loadPixmap(const QString &fileName, const QSize &size = {}, Qt::AspectRatioMode aspectRatioMode = Qt::KeepAspectRatio);
