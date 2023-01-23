#pragma once

#include <QComboBox>
#include <QDialog>
#include <QListWidget>

class SeriesSelector : public QDialog {
  Q_OBJECT

public:
  SeriesSelector(QWidget *parent);
  void addSeries(const QString &id, const QString& msg_name, const QString &sig_name);
  QList<QStringList> series();

private slots:
  void msgSelected(int index);
  void addSignal(QListWidgetItem *item);

private:
  QComboBox *msgs_combo;
  QListWidget *sig_list;
  QListWidget *chart_series;
};
