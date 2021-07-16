#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QHBoxLayout>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"


class MainWindow : public QWidget {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);

private:
  bool eventFilter(QObject *obj, QEvent *event) override;
//  void openSettings();
//  void closeSettings();

//  Device device;
//  QUIState qs;

//  QStackedLayout *main_layout;
//  HomeWindow *homeWindow;
//  SettingsWindow *settingsWindow;
//  OnboardingWindow *onboardingWindow;
};

