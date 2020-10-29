#pragma once
#include "qt/keyboard.hpp"
#include <QWidget>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QLineEdit>


class InputField : public QWidget {
  Q_OBJECT

  public:
    explicit InputField(QWidget* parent = 0);
  
  private:
    Keyboard* k;
    QLineEdit* line;
    QVBoxLayout* l;
    bool eventFilter(QObject* object, QEvent* event);
  public slots:
    void getText(QString s);
  signals:
    void emitText(QString s);
};