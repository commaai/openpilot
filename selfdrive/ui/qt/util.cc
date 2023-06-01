#include "selfdrive/ui/qt/util.h"

#include <QApplication>
#include <QFile>
#include <QHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLayoutItem>
#include <QStyleOption>
#include <QPainterPath>
#include <QTextStream>
#include <QtXml/QDomDocument>

#include "common/params.h"
#include "common/swaglog.h"
#include "system/hardware/hw.h"

QString getVersion() {
  static QString version =  QString::fromStdString(Params().get("Version"));
  return version;
}

QString getBrand() {
  return Params().getBool("Passive") ? QObject::tr("dashcam") : QObject::tr("openpilot");
}

QString getUserAgent() {
  return "openpilot-" + getVersion();
}

std::optional<QString> getDongleId() {
  std::string id = Params().get("DongleId");

  if (!id.empty() && (id != "UnregisteredDevice")) {
    return QString::fromStdString(id);
  } else {
    return {};
  }
}

QMap<QString, QString> getSupportedLanguages() {
  QFile f("translations/languages.json");
  f.open(QIODevice::ReadOnly | QIODevice::Text);
  QString val = f.readAll();

  QJsonObject obj = QJsonDocument::fromJson(val.toUtf8()).object();
  QMap<QString, QString> map;
  for (auto key : obj.keys()) {
    map[key] = obj[key].toString();
  }
  return map;
}

void configFont(QPainter &p, const QString &family, int size, const QString &style) {
  QFont f(family);
  f.setPixelSize(size);
  f.setStyleName(style);
  p.setFont(f);
}

void clearLayout(QLayout* layout) {
  while (layout->count() > 0) {
    QLayoutItem* item = layout->takeAt(0);
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

QString timeAgo(const QDateTime &date) {
  int diff = date.secsTo(QDateTime::currentDateTimeUtc());

  QString s;
  if (diff < 60) {
    s = "now";
  } else if (diff < 60 * 60) {
    int minutes = diff / 60;
    s = QObject::tr("%n minute(s) ago", "", minutes);
  } else if (diff < 60 * 60 * 24) {
    int hours = diff / (60 * 60);
    s = QObject::tr("%n hour(s) ago", "", hours);
  } else if (diff < 3600 * 24 * 7) {
    int days = diff / (60 * 60 * 24);
    s = QObject::tr("%n day(s) ago", "", days);
  } else {
    s = date.date().toString();
  }

  return s;
}

void setQtSurfaceFormat() {
  QSurfaceFormat fmt;
#ifdef __APPLE__
  fmt.setVersion(3, 2);
  fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  fmt.setRenderableType(QSurfaceFormat::OpenGL);
#else
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
#endif
  fmt.setSamples(16);
  QSurfaceFormat::setDefaultFormat(fmt);
}

void sigTermHandler(int s) {
  std::signal(s, SIG_DFL);
  qApp->quit();
}

void initApp(int argc, char *argv[], bool disable_hidpi) {
  Hardware::set_display_power(true);
  Hardware::set_brightness(65);

  // setup signal handlers to exit gracefully
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  if (disable_hidpi) {
#ifdef __APPLE__
    // Get the devicePixelRatio, and scale accordingly to maintain 1:1 rendering
    QApplication tmp(argc, argv);
    qputenv("QT_SCALE_FACTOR", QString::number(1.0 / tmp.devicePixelRatio() ).toLocal8Bit());
#endif
  }

  setQtSurfaceFormat();
}

void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  static std::map<QtMsgType, int> levels = {
    {QtMsgType::QtDebugMsg, CLOUDLOG_DEBUG},
    {QtMsgType::QtInfoMsg, CLOUDLOG_INFO},
    {QtMsgType::QtWarningMsg, CLOUDLOG_WARNING},
    {QtMsgType::QtCriticalMsg, CLOUDLOG_ERROR},
    {QtMsgType::QtSystemMsg, CLOUDLOG_ERROR},
    {QtMsgType::QtFatalMsg, CLOUDLOG_CRITICAL},
  };

  std::string file, function;
  if (context.file != nullptr) file = context.file;
  if (context.function != nullptr) function = context.function;

  auto bts = msg.toUtf8();
  cloudlog_e(levels[type], file.c_str(), context.line, function.c_str(), "%s", bts.constData());
}


QWidget* topWidget (QWidget* widget) {
  while (widget->parentWidget() != nullptr) widget=widget->parentWidget();
  return widget;
}

QPixmap loadPixmap(const QString &fileName, const QSize &size, Qt::AspectRatioMode aspectRatioMode) {
  if (size.isEmpty()) {
    return QPixmap(fileName);
  } else {
    return QPixmap(fileName).scaled(size, aspectRatioMode, Qt::SmoothTransformation);
  }
}

QRect getTextRect(QPainter &p, int flags, const QString &text) {
  QFontMetrics fm(p.font());
  QRect init_rect = fm.boundingRect(text);
  return fm.boundingRect(init_rect, flags, text);
}

void drawRoundedRect(QPainter &painter, const QRectF &rect, qreal xRadiusTop, qreal yRadiusTop, qreal xRadiusBottom, qreal yRadiusBottom){
  qreal w_2 = rect.width() / 2;
  qreal h_2 = rect.height() / 2;

  xRadiusTop = 100 * qMin(xRadiusTop, w_2) / w_2;
  yRadiusTop = 100 * qMin(yRadiusTop, h_2) / h_2;

  xRadiusBottom = 100 * qMin(xRadiusBottom, w_2) / w_2;
  yRadiusBottom = 100 * qMin(yRadiusBottom, h_2) / h_2;

  qreal x = rect.x();
  qreal y = rect.y();
  qreal w = rect.width();
  qreal h = rect.height();

  qreal rxx2Top = w*xRadiusTop/100;
  qreal ryy2Top = h*yRadiusTop/100;

  qreal rxx2Bottom = w*xRadiusBottom/100;
  qreal ryy2Bottom = h*yRadiusBottom/100;

  QPainterPath path;
  path.arcMoveTo(x, y, rxx2Top, ryy2Top, 180);
  path.arcTo(x, y, rxx2Top, ryy2Top, 180, -90);
  path.arcTo(x+w-rxx2Top, y, rxx2Top, ryy2Top, 90, -90);
  path.arcTo(x+w-rxx2Bottom, y+h-ryy2Bottom, rxx2Bottom, ryy2Bottom, 0, -90);
  path.arcTo(x, y+h-ryy2Bottom, rxx2Bottom, ryy2Bottom, 270, -90);
  path.closeSubpath();

  painter.drawPath(path);
}

QColor interpColor(float xv, std::vector<float> xp, std::vector<QColor> fp) {
  assert(xp.size() == fp.size());

  int N = xp.size();
  int hi = 0;

  while (hi < N and xv > xp[hi]) hi++;
  int low = hi - 1;

  if (hi == N && xv > xp[low]) {
    return fp[fp.size() - 1];
  } else if (hi == 0){
    return fp[0];
  } else {
    return QColor(
      (xv - xp[low]) * (fp[hi].red() - fp[low].red()) / (xp[hi] - xp[low]) + fp[low].red(),
      (xv - xp[low]) * (fp[hi].green() - fp[low].green()) / (xp[hi] - xp[low]) + fp[low].green(),
      (xv - xp[low]) * (fp[hi].blue() - fp[low].blue()) / (xp[hi] - xp[low]) + fp[low].blue(),
      (xv - xp[low]) * (fp[hi].alpha() - fp[low].alpha()) / (xp[hi] - xp[low]) + fp[low].alpha()
    );
  }
}

static QHash<QString, QByteArray> load_bootstrap_icons() {
  QHash<QString, QByteArray> icons;

  QFile f(":/bootstrap-icons.svg");
  if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QDomDocument xml;
    xml.setContent(&f);
    QDomNode n = xml.documentElement().firstChild();
    while (!n.isNull()) {
      QDomElement e = n.toElement();
      if (!e.isNull() && e.hasAttribute("id")) {
        QString svg_str;
        QTextStream stream(&svg_str);
        n.save(stream, 0);
        svg_str.replace("<symbol", "<svg");
        svg_str.replace("</symbol>", "</svg>");
        icons[e.attribute("id")] = svg_str.toUtf8();
      }
      n = n.nextSibling();
    }
  }
  return icons;
}

QPixmap bootstrapPixmap(const QString &id) {
  static QHash<QString, QByteArray> icons = load_bootstrap_icons();

  QPixmap pixmap;
  if (auto it = icons.find(id); it != icons.end()) {
    pixmap.loadFromData(it.value(), "svg");
  }
  return pixmap;
}
