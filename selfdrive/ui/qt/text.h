#include <array>

#include <QWidget>
#include <QLabel>
#include <QPushButton>

#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

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

private:
  QLabel *label;
  ScrollView *scroll;
  QPushButton *btn;

};