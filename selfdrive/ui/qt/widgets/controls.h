#pragma once

#include <QFrame>

class QLabel;
class QHBoxLayout;
class QPushButton;
class QVBoxLayout;
class Toggle;

QFrame *horizontal_line(QWidget *parent = nullptr);
class AbstractControl : public QFrame {
  Q_OBJECT

public:
  void setIcon(const QString &icon);
  QString description() const;
  void setDescription(const QString &text);

signals:
  void showDescription();

protected:
 AbstractControl(const QString &title, const QString &desc = {},
                 QWidget *parent = nullptr);
 void hideEvent(QHideEvent *e) override;
 QSize minimumSizeHint() const override;

 QHBoxLayout *controls_layout = nullptr;

private:
  QLabel *icon_label = nullptr;
  QLabel *desc_label = nullptr;
  QPushButton *title_label = nullptr;
  QVBoxLayout *main_layout = nullptr;
  QHBoxLayout *hlayout = nullptr;
};

// widget to display a value
class LabelControl : public AbstractControl {
  Q_OBJECT

public:
 LabelControl(const QString &title, const QString &text = {},
              const QString &desc = {}, QWidget *parent = nullptr);
 void setText(const QString &text);

protected:
  QLabel *label;
};

// widget for a button with a label
class ButtonControl : public AbstractControl {
  Q_OBJECT

public:
 ButtonControl(const QString &title, const QString &text,
               const QString &desc = {}, QWidget *parent = nullptr);
 void setText(const QString &text);
 QString text() const;

signals:
  void released();

public slots:
  void setEnabled(bool enabled);

protected:
  QPushButton *btn;
};

class ToggleControl : public AbstractControl {
  Q_OBJECT

public:
 ToggleControl(const QString &title, const QString &desc = {},
               const QString &icon = {}, const bool state = false, QWidget *parent = nullptr);
 void setEnabled(bool enabled);

signals:
  void toggleFlipped(bool state);

protected:
  Toggle *toggle;
};

// widget to toggle params
class ParamControl : public ToggleControl {
  Q_OBJECT

public:
 ParamControl(const QString &param, const QString &title, const QString &desc,
              const QString &icon, QWidget *parent = nullptr);
};
