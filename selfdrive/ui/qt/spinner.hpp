#include <QTimer>
#include <QLabel>
#include <QWidget>
#include <QPixmap>
#include <QProgressBar>
#include <QTransform>
#include <QSocketNotifier>

class Spinner : public QWidget {
  Q_OBJECT

public:
  explicit Spinner(QWidget *parent = 0);

private:
  QPixmap track_img;
  QTimer *rotate_timer;
  QLabel *comma, *track;
  QLabel *text;
  QProgressBar *progress_bar;
  QTransform transform;
  QSocketNotifier *notifier;

public slots:
  void rotate();
  void update(int n);
};
