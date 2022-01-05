#include <array>

#include <QWidget>

#include "selfdrive/ui/ui.h"

// Forward declaration
class QApplication;

class Text : public QWidget, public Wakeable {
  Q_OBJECT
  Q_INTERFACES(Wakeable)

public:
  explicit Text(char *argv[], QApplication &a, QWidget *parent = 0);

signals:
  void displayPowerChanged(bool on);
  void interactiveTimeout();

public slots:
  virtual void update(const UIState &s);
};