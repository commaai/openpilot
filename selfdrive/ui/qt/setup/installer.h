#include <QLabel>
#include <QProgressBar>
#include <QWidget>

class Installer : public QWidget {
  Q_OBJECT

public:
  explicit Installer(QWidget *parent = 0);

private:
  QLabel *text;
  QProgressBar *progress_bar;
  int freshClone();
  int getProgress(const QString &line);
  void update(const int value);
  bool started = false;

  const QVector<QString> stages = {"Receiving objects: ", "Resolving deltas: ", "Updating files: "};
  const QVector<int> weights = {91, 2, 7};
  QString currentStage;

public slots:
  int install();
};
