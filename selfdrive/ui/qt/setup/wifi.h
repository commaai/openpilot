#include <QWidget>

class WifiSetup : public QWidget {
  Q_OBJECT

public:
  explicit WifiSetup(QWidget *parent = 0);

public slots:
  void finish();
};
