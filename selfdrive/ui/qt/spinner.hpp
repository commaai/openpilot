#include <array>

#include <QTimer>
#include <QLabel>
#include <QWidget>
#include <QPixmap>
#include <QProgressBar>
#include <QSocketNotifier>

constexpr int spinner_fps = 30;
constexpr QSize spinner_size = QSize(360, 360);

class Spinner : public QWidget {
  Q_OBJECT

public:
  explicit Spinner(QWidget *parent = 0);

private:
  int track_idx;
  QLabel *comma, *track;
  QLabel *text;
  QProgressBar *progress_bar;
  std::array<QPixmap, spinner_fps> track_imgs;

  QTimer *rotate_timer;
  QSocketNotifier *notifier;

public slots:
  void rotate();
  void update(int n);
};
