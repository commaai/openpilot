#pragma once

#include <vector>

#include <QString>
#include <QWidget>
#include <QStackedLayout>
#include <QAbstractButton>

class KeyboardLayout : public QWidget {
  Q_OBJECT

public:
  explicit KeyboardLayout(QWidget *parent, std::vector<QVector<QString>> layout);
};

class Keyboard : public QWidget {
  Q_OBJECT

public:
  explicit Keyboard(QWidget *parent = 0);

private:
  QStackedLayout* main_layout;

private slots:
  void handleButton(QAbstractButton* m_button);

signals:
  void emitButton(QString s);
};
