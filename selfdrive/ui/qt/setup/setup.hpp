#include <QString>
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
  QPushButton *continue_btn;

  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *custom_software();
  QWidget *downloading();

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
