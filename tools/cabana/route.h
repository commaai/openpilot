#pragma once

#include <QComboBox>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QDialog>

class OpenRouteDialog : public QDialog {
  Q_OBJECT

public:
  OpenRouteDialog(QWidget *parent);
  void loadRoute();
  inline bool failedToLoad() const { return failed_to_load; }

private:
  QLineEdit *route_edit;
  QComboBox *choose_video_cb;
  QDialogButtonBox *btn_box;
  bool failed_to_load = false;
};
