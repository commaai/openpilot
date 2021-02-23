#include <QString>
#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>

class WifiSetup : public QStackedWidget {
  Q_OBJECT

public:
  explicit WifiSetup(QWidget *parent = 0);

private:
  QWidget *network_setup();

  QWidget *build_page(QString title, QWidget *content);

public slots:
  void finish();
};
