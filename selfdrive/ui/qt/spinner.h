#include <array>

#include <QLabel>
#include <QPixmap>
#include <QProgressBar>
#include <QSocketNotifier>
#include <QVariantAnimation>
#include <QWidget>

const int spinner_fps = 20;
const QSize spinner_size = QSize(360, 360);

class TrackWidget : public QWidget  {
  Q_OBJECT
public:
  TrackWidget(QWidget *parent = nullptr);

private:
  void paintEvent(QPaintEvent *event) override;
  std::array<QPixmap, 30> track_imgs;
  QVariantAnimation m_anim;
};

class Spinner : public QWidget {
  Q_OBJECT

public:
  explicit Spinner(QWidget *parent = 0);

private:
  QLabel *text;
  QProgressBar *progress_bar;
  QSocketNotifier *notifier;

public slots:
  void update(int n);
};
