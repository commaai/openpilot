#include <array>

#include <QLabel>
#include <QPixmap>
#include <QProgressBar>
#include <QSocketNotifier>
#include <QVariantAnimation>
#include <QWidget>

#include "selfdrive/ui/ui.h"

constexpr int spinner_fps = 30;
constexpr QSize spinner_size = QSize(360, 360);

class TrackWidget : public QWidget  {
  Q_OBJECT
public:
  TrackWidget(QWidget *parent = nullptr);

private:
  void paintEvent(QPaintEvent *event) override;
  std::array<QPixmap, spinner_fps> track_imgs;
  QVariantAnimation m_anim;
};

class Spinner : public QWidget, public Wakeable {
  Q_OBJECT
  Q_INTERFACES(Wakeable)

public:
  explicit Spinner(QWidget *parent = 0);

signals:
  void displayPowerChanged(bool on);
  void interactiveTimeout();

public slots:
  void update(int n);
  virtual void update(const UIState &s);

private:
  QLabel *text;
  QProgressBar *progress_bar;
  QSocketNotifier *notifier;
};
