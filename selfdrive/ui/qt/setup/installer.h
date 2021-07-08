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
  int install();
  int freshClone();
  int getProgress(const QString &line);
  bool started = false;

  const QVector<QString> stages = {"Receiving objects: ", "Resolving deltas: "};
  const QVector<int> weights = {95, 5};
  QString currentStage;

public slots:
  void update(const int value);
};
