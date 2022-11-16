#pragma once

#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>

#include "selfdrive/ui/qt/widgets/controls.h"

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class SignalForm : public QWidget {
  Q_OBJECT
public:
  SignalForm(QWidget *parent);
  QLineEdit *name, *unit, *comment, *val_desc, *offset, *factor, *min_val, *max_val;
  QLabel *lsb, *msb;
  QSpinBox *size;
  QComboBox *sign, *endianness;
  bool changed_by_user = false;

 signals:
  void changed();
};

class SignalEdit : public QWidget {
  Q_OBJECT

public:
  SignalEdit(int index, QWidget *parent = nullptr);
  void setSignal(const QString &msg_id, const Signal *sig);
  void setChartOpened(bool opened);
  void signalHovered(const Signal *sig);
  void updateForm(bool show);
  inline bool isFormVisible() const { return form->isVisible(); }
  const Signal *sig = nullptr;
  QString msg_id;

signals:
  void highlight(const Signal *sig);
  void showChart(const QString &name, const Signal *sig, bool show);
  void remove(const Signal *sig);
  void save(const Signal *sig, const Signal &new_sig);
  void showFormClicked();

protected:
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void saveSignal();

  SignalForm *form = nullptr;
  ElidedLabel *title;
  QLabel *color_label;
  QLabel *icon;
  int form_idx = 0;
  QToolButton *plot_btn;
  QTimer *save_timer;
};

class SignalFindDlg : public QDialog {
public:
  SignalFindDlg(const QString &id, const Signal *signal, QWidget *parent);
};
