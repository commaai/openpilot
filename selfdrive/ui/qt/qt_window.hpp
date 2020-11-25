#include <QWidget>

#ifdef QCOM2
  const int vwp_w = 2160, vwp_h = 1080;
#else
  const int vwp_w = 1920, vwp_h = 1080;
#endif

void setMainWindow(QWidget *w);
