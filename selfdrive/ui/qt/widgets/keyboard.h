#pragma once

#include <vector>

#include <QPushButton>
#include <QAbstractButton>
#include <QFrame>
#include <QStackedLayout>
#include <QString>
#include <QWidget>

class KeyButton : public QPushButton {
  Q_OBJECT

public:
//  explicit KeyButton(QWidget* parent = nullptr, const QString& c);
  explicit KeyButton(QWidget *parent = 0);
  explicit KeyButton(const QString &text, QWidget *parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;
};

class KeyboardLayout : public QWidget {
  Q_OBJECT

public:
  explicit KeyboardLayout(QWidget* parent, const std::vector<QVector<QString>>& layout);
};

class Keyboard : public QFrame {
  Q_OBJECT

public:
  explicit Keyboard(QWidget *parent = 0);

private:
  QStackedLayout* main_layout;

private slots:
  void handleButton(QAbstractButton* m_button);
  void pressed(QAbstractButton* m_button);
  void released(QAbstractButton* m_button);

signals:
  void emitButton(const QString &s);
};
