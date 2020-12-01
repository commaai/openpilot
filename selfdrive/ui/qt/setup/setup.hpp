#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  QLineEdit *url_input;

  QWidget *getting_started();
  QWidget *software_selection();
  QWidget *downloading();
  QWidget *network_setup();

  QPushButton *continue_btn;

public slots:
  void nextPage();
};
