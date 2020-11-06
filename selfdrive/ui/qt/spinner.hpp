#include <QTimer>
#include <QLabel>
#include <QWidget>
#include <QPixmap>
#include <QProgressBar>
#include <QTransform>

class Spinner : public QWidget {
  Q_OBJECT

public:
  explicit Spinner(QWidget *parent = 0);

private:
  QPixmap track_img;
  QTimer *timer;
  QLabel *comma, *track;
  QLabel *text;
  QProgressBar *progress_bar;
  QTransform transform;

public slots:
  void update();
};
