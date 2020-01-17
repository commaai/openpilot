#include <QApplication>
#include <QWidget>
#include <QString>
#include <QTimer>
#include <QPushButton>
#include <QGraphicsScene>
#include <QPainter>
#include <QThread>
#include <QMouseEvent>
#include <QReadWriteLock>
#include <QLineEdit>

#include "FileReader.hpp"
#include "Unlogger.hpp"
#include "FrameReader.hpp"

class Window : public QWidget {
  public:
    Window(QString route_, int seek);
    bool addSegment(int i);
  protected:
    void keyPressEvent(QKeyEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;
    uint64_t ct;
    Unlogger *unlogger;
  private:
    int timeToPixel(uint64_t ns);
    uint64_t pixelToTime(int px);
    QString route;

    QReadWriteLock events_lock;
    Events events;
    int last_event_size = 0;

    QMap<int, LogReader*> lrs;
    QMap<int, FrameReader*> frs;

    // cache the bar
    QPixmap *px = NULL;
    int seg_add = 0;

    QLineEdit *timeLE;
};

Window::Window(QString route_, int seek) : route(route_) {
  timeLE = new QLineEdit(this);
  timeLE->setPlaceholderText("Placeholder Text");
  timeLE->move(50, 650);

  QThread* thread = new QThread;
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
  unlogger->moveToThread(thread);
  connect(thread, SIGNAL (started()), unlogger, SLOT (process()));
  connect(unlogger, SIGNAL (elapsed()), this, SLOT (update()));
  thread->start();

  this->setFocusPolicy(Qt::StrongFocus);

  // add the first segment
  addSegment(seek/60);
}

bool Window::addSegment(int i) {
  if (lrs.find(i) == lrs.end()) {
    QString fn = QString("%1/%2/rlog.bz2").arg(route).arg(i);

    QThread* thread = new QThread;
    lrs.insert(i, new LogReader(fn, &events, &events_lock, &unlogger->eidx));
    lrs[i]->moveToThread(thread);
    connect(thread, SIGNAL (started()), lrs[i], SLOT (process()));
    thread->start();
    //connect(lrs[i], SIGNAL (finished()), this, SLOT (update()));

    QString frn = QString("%1/%2/fcamera.hevc").arg(route).arg(i);
    frs.insert(i, new FrameReader(qPrintable(frn)));
    return true;
  }
  return false;
}

#define PIXELS_PER_SEC 0.5

int Window::timeToPixel(uint64_t ns) {
  // TODO: make this dynamic
  return int(ns*1e-9*PIXELS_PER_SEC+0.5);
}

uint64_t Window::pixelToTime(int px) {
  // TODO: make this dynamic
  //printf("%d\n", px);
  return ((px+0.5)/PIXELS_PER_SEC) * 1e9;
}

void Window::keyPressEvent(QKeyEvent *event) {
  printf("keypress: %x\n", event->key());
  if (event->key() == Qt::Key_Space) unlogger->togglePause();
}

void Window::mousePressEvent(QMouseEvent *event) {
  //printf("mouse event\n");
  if (event->button() == Qt::LeftButton) {
    uint64_t t0 = events.begin().key();
    uint64_t tt = pixelToTime(event->x());
    int seg = int((tt*1e-9)/60);
    printf("segment %d\n", seg);
    addSegment(seg);

    //printf("seek to %lu\n", t0+tt);
    unlogger->setSeekRequest(t0+tt);
  }
  this->update();
}

void Window::paintEvent(QPaintEvent *event) {
  if (events.size() == 0) return;

  QElapsedTimer timer;
  timer.start();

  uint64_t t0 = events.begin().key();
  uint64_t t1 = (events.end()-1).key();

  //p.drawRect(0, 0, 600, 100);

  // TODO: we really don't have to redraw this every time, only on updates to events
  int this_event_size = events.size();
  if (last_event_size != this_event_size) {
    if (px != NULL) delete px;
    px = new QPixmap(1920, 600);
    px->fill(QColor(0xd8, 0xd8, 0xd8));

    QPainter tt(px);
    tt.setBrush(Qt::cyan);

    int lt = -1;
    int lvv = 0;
    for (auto e : events) {
      auto type = e.which();
      //printf("%lld %d\n", e.getLogMonoTime()-t0, type);
      if (type == cereal::Event::CONTROLS_STATE) {
        auto controlsState = e.getControlsState();
        uint64_t t = (e.getLogMonoTime()-t0);
        float vEgo = controlsState.getVEgo();
        int enabled = controlsState.getState() == cereal::ControlsState::OpenpilotState::ENABLED;
        int rt = timeToPixel(t); // 250 ms per pixel
        if (rt != lt) {
          int vv = vEgo*8.0;
          if (lt != -1) {
            tt.setPen(Qt::red);
            tt.drawLine(lt, 300-lvv, rt, 300-vv);

            if (enabled) {
              tt.setPen(Qt::green); 
            } else {
              tt.setPen(Qt::blue); 
            }

            tt.drawLine(rt, 300, rt, 600);
          }
          lt = rt;
          lvv = vv;
        }
      }
    }
    tt.end();
    last_event_size = this_event_size;
    if (lrs.find(seg_add) != lrs.end() && lrs[seg_add]->is_done) {
      while (!addSegment(++seg_add));
    }
  }

  QPainter p(this);
  if (px != NULL) p.drawPixmap(0, 0, 1920, 600, *px);

  p.setBrush(Qt::cyan);

  uint64_t ct = unlogger->getCurrentTime();
  if (ct != 0) {
    addSegment((((ct-t0)*1e-9)/60)+1);
    int rrt = timeToPixel(ct-t0);
    p.drawRect(rrt-1, 0, 2, 600);

    timeLE->setText(QString("%1").arg((ct-t0)*1e-9, '8', 'f', 2));
  }

  p.end();

  if (timer.elapsed() > 50) {
    qDebug() << "paint in" << timer.elapsed() << "ms";
  }
}

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  QString route(argv[1]);
  int seek = QString(argv[2]).toInt();
  printf("seek: %d\n", seek);
  route = route.replace("|", "/");
  if (route == "") {
    printf("usage %s: <route>\n", argv[0]);
    exit(0);
    //route = "3a5d6ac1c23e5536/2019-10-29--10-06-58";
    //route = "0006c839f32a6f99/2019-02-18--06-21-29";
    //route = "02ec6bea180a4d36/2019-10-25--10-18-09";
  }

  Window window(route, seek);
  window.resize(1920, 800);
  window.setWindowTitle("nui unlogger");
  window.show();

  return app.exec();
}

