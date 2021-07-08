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
  int freshClone();
  int getProgress(const QString &line);
  bool started = false;
  QString currentStage;

public slots:
  void update(const int value);
};
