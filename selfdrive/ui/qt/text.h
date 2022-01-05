#include <array>

#include <QWidget>

// Forward declaration
class QApplication;

class Text : public QWidget {
  Q_OBJECT

public:
  explicit Text(char *argv[], QApplication &a, QWidget *parent = 0);
};