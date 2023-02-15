#pragma once

#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QListWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>

class RemoteRouteList : public QStackedWidget {
  Q_OBJECT
public:
  RemoteRouteList(QWidget *parent);

  QListWidget *list;

private:
  void getDevices();
  void getRouteList(const QString &dongleid);

  QPushButton *retry_btn;
  QLabel *msg_label;
  QComboBox *dongleid_cb;
};

class OpenRouteDialog : public QDialog {
  Q_OBJECT

public:
  OpenRouteDialog(QWidget *parent);
  void loadRoute();
  inline bool failedToLoad() const { return failed_to_load; }

private:
  QLineEdit *route_edit;
  QDialogButtonBox *btn_box;
  RemoteRouteList *route_list;
  bool failed_to_load = false;
};
