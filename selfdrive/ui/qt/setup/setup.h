#include <QStackedWidget>
#include <QString>
#include <QWidget>

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *custom_software();
  QWidget *downloading();
  QWidget *download_failed();

  QWidget *build_page(QString title, QWidget *content, bool next, bool prev);

signals:
  void downloadFailed();

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
