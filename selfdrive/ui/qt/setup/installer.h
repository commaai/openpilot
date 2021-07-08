#include <array>

#include <QLabel>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QProgressBar>
#include <QSocketNotifier>
#include <QVariantAnimation>
#include <QWidget>

class Installer : public QWidget {
  Q_OBJECT

public:
  explicit Installer(QWidget *parent = 0);

private:
  QLabel *text;
  QProgressBar *progress_bar;
  QSocketNotifier *notifier;
  int install();
  int fresh_clone();
  bool started = false;

public slots:
  void update(const int value);
};
