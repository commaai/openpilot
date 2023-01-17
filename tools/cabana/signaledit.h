#pragma once

#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QToolButton>

#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class SignalForm : public QWidget {
  Q_OBJECT
public:
  SignalForm(QWidget *parent);
  void textBoxEditingFinished();

  QLineEdit *name, *unit, *comment, *val_desc, *offset, *factor, *min_val, *max_val;
  QSpinBox *size;
  QComboBox *sign, *endianness;
  QToolButton *expand_btn;

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
  const Signal *sig = nullptr;
  SignalForm *form = nullptr;
  QString msg_id;

signals:
  void highlight(const Signal *sig);
  void showChart(const QString &name, const Signal *sig, bool show, bool merge);
  void remove(const Signal *sig);
  void save(const Signal *sig, const Signal &new_sig);
  void showFormClicked(const Signal *sig);

protected:
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void saveSignal();

  ElidedLabel *title;
  QLabel *color_label;
  QLabel *icon;
  int form_idx = 0;
  QToolButton *plot_btn;
};
