#pragma once

#include <QLabel>
#include <QStackedWidget>
#include <QString>
#include <QTranslator>
#include <QWidget>

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  void selectLanguage();
  QWidget *low_voltage();
  QWidget *custom_software_warning();
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *downloading();
  QWidget *download_failed(QLabel *url, QLabel *body);

  QWidget *failed_widget;
  QWidget *downloading_widget;
  QWidget *custom_software_warning_widget;
  QWidget *software_selection_widget;
  QTranslator translator;

signals:
  void finished(const QString &url, const QString &error = "");

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
