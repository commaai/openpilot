#include "tools/cabana/tools/routeinfo.h"
#include <QHeaderView>
#include <QScrollBar>
#include <QTableWidget>
#include <QVBoxLayout>
#include "tools/cabana/streams/replaystream.h"

RouteInfoDlg::RouteInfoDlg(QWidget *parent) : QDialog(parent) {
  auto *replay = qobject_cast<ReplayStream *>(can)->getReplay();
  setWindowTitle(tr("Route: %1").arg(QString::fromStdString(replay->route().name())));

  auto *table = new QTableWidget(replay->route().segments().size(), 7, this);
  table->setToolTip(tr("Click on a row to seek to the corresponding segment."));
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);
  table->setSelectionMode(QAbstractItemView::SingleSelection);
  table->setHorizontalHeaderLabels({"", "rlog", "fcam", "ecam", "dcam", "qlog", "qcam"});
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  table->verticalHeader()->setVisible(false);
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  int row = 0;
  for (const auto &[seg_num, seg] : replay->route().segments()) {
    table->setItem(row, 0, new QTableWidgetItem(QString::number(seg_num)));
    table->setItem(row, 1, new QTableWidgetItem(seg.rlog.empty() ? "--" : "Yes"));
    table->setItem(row, 2, new QTableWidgetItem(seg.road_cam.empty() ? "--" : "Yes"));
    table->setItem(row, 3, new QTableWidgetItem(seg.wide_road_cam.empty() ? "--" : "Yes"));
    table->setItem(row, 4, new QTableWidgetItem(seg.driver_cam.empty() ? "--" : "Yes"));
    table->setItem(row, 5, new QTableWidgetItem(seg.qlog.empty() ? "--" : "Yes"));
    table->setItem(row, 6, new QTableWidgetItem(seg.qcamera.empty() ? "--" : "Yes"));
    ++row;
  }
  table->setMinimumWidth(table->horizontalHeader()->length() + table->verticalScrollBar()->sizeHint().width());
  table->setMinimumHeight(table->rowHeight(0) * std::min(table->rowCount(), 13) + table->horizontalHeader()->height() + table->frameWidth() * 2);

  connect(table, &QTableWidget::itemClicked, [](QTableWidgetItem *item) { can->seekTo(item->row() * 60.0); });

  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->addWidget(table);
}
